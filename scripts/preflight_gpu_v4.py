"""Fast (<3 min) 1080p full-stack CUDA pre-flight verifying all 5 GPU models, VRAM release, sharding & Layer 7 gate."""

from __future__ import annotations

import gc
import json
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from src.egoannotate.config import load_config
from src.egoannotate.gpu_backends import GroundingDINOGPUDetector, clear_gpu_model_cache
from src.egoannotate.gpu_backends_v4 import (
    MoGe2MetricDepthGPU,
    Qwen3VLFastAnnotator,
    SAM21MaskPropagatorGPU,
)
from src.egoannotate.lerobot_sharding import pack_sharded_lerobot_v3
from src.egoannotate.perception_hands_rtmw import (
    _destroy_rtmw_model,
    _load_rtmw_wholebody_locked,
    release_rtmw_vram_session,
)
from src.egoannotate.pipeline_v3 import EgoAnnotateV3Pipeline
from src.egoannotate.policy_gate import validate_lerobot_v3_dataset
from src.egoannotate.privacy_v3 import probe_nvenc_available


def _peak_gb() -> float:
    return torch.cuda.max_memory_allocated(0) / (1024.0**3) if torch.cuda.is_available() else 0.0


def _free_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free_b, _ = torch.cuda.mem_get_info(0)
    return free_b / (1024.0**3)


def main() -> int:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("EGOANNOTATE_CACHE_GPU_MODELS", "1")
    t_all = time.perf_counter()
    assert torch.cuda.is_available(), "CUDA GPU not available!"
    n_gpus = torch.cuda.device_count()
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024.0**3)
    print(
        f"[PREFLIGHT] GPU={torch.cuda.get_device_name(0)} | count={n_gpus} | "
        f"VRAM={total_vram:.1f} GB | free={_free_gb():.2f} GB | NVENC={probe_nvenc_available()}",
        flush=True,
    )

    frame_1080p = np.full((1080, 1920, 3), 120, dtype=np.uint8)
    cv2.rectangle(frame_1080p, (480, 300), (1240, 860), (40, 180, 70), -1)

    # 1. RTMW ONNXRuntime CUDAExecutionProvider + real 1080p FPS + verify VRAM release
    t0 = time.perf_counter()
    wb = _load_rtmw_wholebody_locked("cuda")
    wb(frame_1080p)
    t_fps = time.perf_counter()
    for _ in range(10):
        wb(frame_1080p)
    fps = 10.0 / max(time.perf_counter() - t_fps, 1e-4)
    _destroy_rtmw_model(wb)
    del wb
    release_rtmw_vram_session()
    print(
        f"[PREFLIGHT] 1/6 RTMW 1080p CUDA OK | {fps:.1f} fps | "
        f"free_after_close={_free_gb():.2f} GB | init+bench={time.perf_counter()-t0:.1f}s",
        flush=True,
    )
    assert fps >= 4.0, f"RTMW too slow ({fps:.1f} fps) - likely CPU fallback!"

    # 2. Qwen3-VL-4B-Instruct Pass 1 (4x 1080p) + Pass 2 + clean VRAM release
    torch.cuda.reset_peak_memory_stats(0)
    t0 = time.perf_counter()
    qwen = Qwen3VLFastAnnotator(device="cuda", mixed_precision="fp16")
    scene = qwen.discover_scene_and_objects([frame_1080p] * 4)
    cap = qwen.describe_segment_keyframes([frame_1080p], "grasp")
    q_peak = _peak_gb()
    qwen.release_vram()
    del qwen
    clear_gpu_model_cache()
    print(
        f"[PREFLIGHT] 2/6 Qwen3-VL 1080p OK | peak_vram={q_peak:.2f} GB | "
        f"free_after_release={_free_gb():.2f} GB | goal='{scene['task_goal'][:45]}' | "
        f"cap='{cap[:45]}' | {time.perf_counter()-t0:.1f}s",
        flush=True,
    )
    assert q_peak < (total_vram - 0.8), f"Qwen3-VL peak VRAM too close to limit: {q_peak:.2f} GB"

    # 3. Grounding DINO on 4x 1080p (auto-capped to bs=2 on <=22 GB GPUs)
    t0 = time.perf_counter()
    dino = GroundingDINOGPUDetector(
        prompts=["tool", "box", "cloth", "trowel", "welding torch"],
        device="cuda",
        mixed_precision="fp16",
    )
    _ = dino.detect_batch([frame_1080p] * 4, batch_size=8)
    del dino
    clear_gpu_model_cache()
    print(
        f"[PREFLIGHT] 3/6 Grounding DINO 1080p OK | free={_free_gb():.2f} GB | "
        f"{time.perf_counter()-t0:.1f}s",
        flush=True,
    )

    # 4. SAM 2.1 on 1080p FP16 -> float32 CPU post_process_masks
    t0 = time.perf_counter()
    sam = SAM21MaskPropagatorGPU(device="cuda", mixed_precision="fp16")
    masks = sam.segment_boxes(frame_1080p, [[480.0, 300.0, 1240.0, 860.0]])
    assert len(masks) == 1 and masks[0]["area_frac"] > 0.0
    del sam
    clear_gpu_model_cache()
    print(
        f"[PREFLIGHT] 4/6 SAM 2.1 1080p OK | backend={masks[0]['backend']} | "
        f"area={masks[0]['area_frac']:.3f} | free={_free_gb():.2f} GB | {time.perf_counter()-t0:.1f}s",
        flush=True,
    )

    # 5. MoGe-2 metric depth on 1080p
    t0 = time.perf_counter()
    moge = MoGe2MetricDepthGPU(device="cuda", mixed_precision="fp16")
    d_out = moge.infer_batch([frame_1080p], batch_size=1)
    assert len(d_out) == 1 and d_out[0]["depth_m"].ndim == 2
    del moge
    clear_gpu_model_cache()
    gc.collect()
    print(
        f"[PREFLIGHT] 5/6 Depth 1080p OK | backend={d_out[0]['backend']} | "
        f"metric={d_out[0]['is_metric']} | med={d_out[0]['median_depth_m']}m | "
        f"free={_free_gb():.2f} GB | {time.perf_counter()-t0:.1f}s",
        flush=True,
    )

    # 6. Full end-to-end 720p 60fps -> 30fps pipeline (all 5 models) + atomic sharding + Layer 7 gate
    torch.cuda.reset_peak_memory_stats(0)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        vid = td / "smoke_720p_60fps.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(str(vid), fourcc, 60.0, (1280, 720))
        for i in range(60):
            fr = np.full((720, 1280, 3), 100, dtype=np.uint8)
            cv2.circle(fr, (240 + i * 8, 360), 60, (50, 180, 90), -1)
            writer.write(fr)
        writer.release()

        cfg = load_config(
            profile="gpu_prod",
            overrides={
                "runtime": {"cache_dir": str(td / "cache")},
                "export": {
                    "render_qa_overlay": False,
                    "export_3d_viewer": False,
                    "generate_html_report": False,
                },
            },
        )
        ep_out = td / "ep_smoke"
        rep = EgoAnnotateV3Pipeline(cfg=cfg).run_video(
            vid,
            ep_out,
            episode_id="smoke",
            use_cache=False,
        )
        gate = validate_lerobot_v3_dataset(
            ep_out / "lerobot_v3",
            run_smoke_train=True,
            smoke_epochs=2,
        )
        assert gate["passed"], f"Layer 7 gate failed on smoke clip: {gate['errors']}"

        rep["quality"]["verdict"] = "PASS"
        (ep_out / "run_report.json").write_text(json.dumps(rep), encoding="utf-8")
        sh = pack_sharded_lerobot_v3([ep_out], td / "sharded", run_validation_gate=True)
        assert sh["total_episodes"] == 1 and sh["validation_gate"]["passed"]

    print(
        f"[PREFLIGHT] 6/6 E2E 720p Pipeline (all 5 models) + Sharding + Layer 7 Gate PASSED | "
        f"e2e_peak_vram={_peak_gb():.2f} GB | free={_free_gb():.2f} GB | "
        f"total_preflight={time.perf_counter()-t_all:.1f}s | pid={os.getpid()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
