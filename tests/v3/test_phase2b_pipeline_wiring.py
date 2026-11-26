"""End-to-end Phase 2B wiring test on synthetic 60 FPS video."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.egoannotate.config import load_config
from src.egoannotate.pipeline_v3 import EgoAnnotateV3Pipeline


def test_pipeline_v4_runs_cfr_contracts_and_gap_free_hierarchy(tmp_path: Path) -> None:
    vid = tmp_path / "synthetic_60fps.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(vid), fourcc, 60.0, (160, 120))
    for i in range(60):
        img = np.full((120, 160, 3), 90, dtype=np.uint8)
        cv2.circle(img, (40 + i, 60), 12, (200, 180, 150), -1)
        writer.write(img)
    writer.release()

    cfg = load_config(
        profile="cpu_smoke",
        overrides={
            "runtime": {
                "device": "cpu",
                "target_fps": 30.0,
                "normalize_fps_cfr": True,
                "use_nvdec": False,
                "cache_dir": str(tmp_path / "cache"),
            },
            "perception": {"hand_backend": "rtmw_133_apache", "enable_upper_body": False},
            "retargeting": {"enabled": True, "embodiments": ["franka_panda"]},
            "export": {
                "formats": ["lerobot_v3", "parquet_json"],
                "render_qa_overlay": False,
                "export_3d_viewer": False,
                "generate_html_report": False,
            },
            "privacy": {"enable_face_blur": False, "strip_audio": True, "use_nvenc": False},
        },
    )
    out_dir = tmp_path / "out_ep"
    report = EgoAnnotateV3Pipeline(cfg=cfg).run_video(vid, out_dir, episode_id="ep_sync", use_cache=False)

    assert report["schema_version"] == "4.0.0"
    assert report["video"]["processing_fps"] == 30.0
    assert 28 <= report["video"]["processed_frames"] <= 32
    assert report["language"]["segments_count"] >= 1
    assert report["language"]["subtasks_count"] >= 1
    assert "contact_events" in report
    assert "depth_provenance" in report

    sidecar = json.loads((out_dir / "parquet_json" / "episode_ir.json").read_text(encoding="utf-8"))
    frames = sidecar["frames"]
    assert len(frames) == report["video"]["processed_frames"]
    assert len(frames[0]["canonical_delta_action"]) == 16
    assert "proprio_valid" in frames[0]
    segs = sidecar["segments"]
    assert segs[0]["start_frame"] == 0
    assert segs[-1]["end_frame"] == len(frames) - 1
