"""Fast (<2 min) full-stack CUDA pre-flight verifying all 5 GPU models, VRAM budget, sharding & Layer 7 gate."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from src.egoannotate.config import load_config
from src.egoannotate.gpu_backends import GroundingDINOGPUDetector
from src.egoannotate.gpu_backends_v4 import (
    MoGe2MetricDepthGPU,
    Qwen3VLFastAnnotator,
    SAM21MaskPropagatorGPU,
)
from src.egoannotate.lerobot_sharding import pack_sharded_lerobot_v3
from src.egoannotate.perception_hands_rtmw import (
    _load_rtmw_wholebody_locked,
    release_rtmw_vram_session,
)
from src.egoannotate.pipeline_v3 import EgoAnnotateV3Pipeline
from src.egoannotate.policy_gate import validate_lerobot_v3_dataset
from src.egoannotate.privacy_v3 import probe_nvenc_available


def _peak_gb() -> float:
    return torch.cuda.max_memory_allocated(0) / (1024.0**3) if torch.cuda.is_available() else 0.0


def main() -> int:
    t_all = time.perf_counter()
    assert torch.cuda.is_available(), "CUDA GPU not available!"
    n_gpus = torch.cuda.device_count()
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024.0**3)
    print(
        f"[PREFLIGHT] GPU={torch.cuda.get_device_name(0)} | count={n_gpus} | "
        f"VRAM={total_vram:.1f} GB | NVENC={probe_nvenc_available()}",
        flush=True,
    )

    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    cv2.rectangle(frame, (180, 140), (440, 360), (40, 180, 70), -1)

    # 1. RTMW ONNXRuntime CUDAExecutionProvider + real FPS check
    t0 = time.perf_counter()
    wb = _load_rtmw_wholebody_locked("cuda")
    wb(frame)
    t_fps = time.perf_counter()
    for _ in range(10):
        wb(frame)
    fps = 10.0 / max(time.perf_counter() - t_fps, 1e-4)
    print(f"[PREFLIGHT] 1/6 RTMW CUDA OK | {fps:.1f} fps | init+bench={time.perf_counter()-t0:.1f}s", flush=True)
    assert fps >= 5.0, f"RTMW too slow ({fps:.1f} fps) - likely CPU fallback!"
    release_rtmw_vram_session()

    # 2. Qwen3-VL-4B-Instruct Pass 1 + Pass 2 + clean VRAM release
    torch.cuda.reset_peak_memory_stats(0)
    t0 = time.perf_counter()
    qwen = Qwen3VLFastAnnotator(device="cuda", mixed_precision="fp16")
    scene = qwen.discover_scene_and_objects([frame, frame])
    cap = qwen.describe_segment_keyframes([frame], "grasp")
    q_peak = _peak_gb()
    qwen.release_vram()
    print(
        f"[PREFLIGHT] 2/6 Qwen3-VL OK | peak_vram={q_peak:.2f} GB | "
        f"goal='{scene['task_goal'][:50]}' | cap='{cap[:50]}' | {time.perf_counter()-t0:.1f}s",
        flush=True,
    )
    assert q_peak < (total_vram - 0.8), f"Qwen3-VL peak VRAM too close to limit: {q_peak:.2f} GB"

    # 3. Grounding DINO batched FP16
    t0 = time.perf_counter()
    dino = GroundingDINOGPUDetector(prompts=["tool", "box", "cloth"], device="cuda", mixed_precision="fp16")
    _ = dino.detect_batch([frame, frame], batch_size=2)
    print(f"[PREFLIGHT] 3/6 Grounding DINO OK | {time.perf_counter()-t0:.1f}s", flush=True)

    # 4. SAM 2.1 FP16 -> float32 CPU post_process_masks check
    t0 = time.perf_counter()
    sam = SAM21MaskPropagatorGPU(device="cuda", mixed_precision="fp16")
    masks = sam.segment_boxes(frame, [[180.0, 140.0, 440.0, 360.0]])
    assert len(masks) == 1 and masks[0]["area_frac"] > 0.0
    print(
        f"[PREFLIGHT] 4/6 SAM 2.1 OK | backend={masks[0]['backend']} | "
        f"area={masks[0]['area_frac']:.3f} | {time.perf_counter()-t0:.1f}s",
        flush=True,
    )

    # 5. MoGe-2 metric depth + normals + intrinsics
    t0 = time.perf_counter()
    moge = MoGe2MetricDepthGPU(device="cuda", mixed_precision="fp16")
    d_out = moge.infer_batch([frame], batch_size=1)
    assert len(d_out) == 1 and d_out[0]["depth_m"].ndim == 2
    print(
        f"[PREFLIGHT] 5/6 Depth OK | backend={d_out[0]['backend']} | "
        f"metric={d_out[0]['is_metric']} | med={d_out[0]['median_depth_m']}m | {time.perf_counter()-t0:.1f}s",
        flush=True,
    )

    # 6. Full end-to-end 30-frame pipeline + atomic sharding + Layer 7 gate
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        vid = td / "smoke_60fps.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(str(vid), fourcc, 60.0, (320, 240))
        for i in range(60):
            fr = np.full((240, 320, 3), 100, dtype=np.uint8)
            cv2.circle(fr, (60 + i * 2, 120), 20, (50, 180, 90), -1)
            writer.write(fr)
        writer.release()

        cfg = load_config(
            profile="gpu_prod",
            overrides={
                "runtime": {"cache_dir": str(td / "cache")},
                "semantic": {"vlm_backend": "geometric_grounded"},
                "export": {
                    "render_qa_overlay": False,
                    "export_3d_viewer": False,
                    "generate_html_report": False,
                },
            },
        )
        ep_out = td / "ep_smoke"
        rep = EgoAnnotateV3Pipeline(cfg=cfg).run_video(vid, ep_out, episode_id="smoke", use_cache=False)
        gate = validate_lerobot_v3_dataset(ep_out / "lerobot_v3", run_smoke_train=True, smoke_epochs=2)
        assert gate["passed"], f"Layer 7 gate failed on smoke clip: {gate['errors']}"

        rep["quality"]["verdict"] = "PASS"
        (ep_out / "run_report.json").write_text(__import__("json").dumps(rep), encoding="utf-8")
        sh = pack_sharded_lerobot_v3([ep_out], td / "sharded", run_validation_gate=True)
        assert sh["total_episodes"] == 1 and sh["validation_gate"]["passed"]

    print(
        f"[PREFLIGHT] 6/6 E2E Pipeline + Sharding + Layer 7 Gate PASSED | "
        f"total_preflight={time.perf_counter()-t_all:.1f}s | pid={os.getpid()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
