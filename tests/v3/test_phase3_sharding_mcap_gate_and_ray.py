"""Tests for Phase 3: Sharded LeRobot v3.0, MCAP, dataset_card.json, Layer 7 gate & Ray fallback."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from src.egoannotate.config import load_config
from src.egoannotate.lerobot_sharding import pack_sharded_lerobot_v3
from src.egoannotate.pipeline_v3 import EgoAnnotateV3Pipeline
from src.egoannotate.policy_gate import validate_lerobot_v3_dataset
from src.egoannotate.ray_pipeline import STAGE_ACTOR_POOLS, RayStreamingOrchestrator


def _make_clip(path: Path, n_frames: int = 30, shift: int = 0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (160, 120))
    for i in range(n_frames):
        img = np.full((120, 160, 3), 95, dtype=np.uint8)
        cv2.circle(img, (35 + shift + i, 60), 10, (210, 175, 140), -1)
        writer.write(img)
    writer.release()


def test_mcap_dataset_card_sharded_lerobot_v3_and_layer7_gate(tmp_path: Path) -> None:
    vid1 = tmp_path / "ep_alpha.mp4"
    vid2 = tmp_path / "ep_beta.mp4"
    _make_clip(vid1, n_frames=30, shift=0)
    _make_clip(vid2, n_frames=30, shift=10)

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
            "retargeting": {"enabled": True, "embodiments": ["franka_panda", "unitree_g1_upper"]},
            "export": {
                "formats": ["lerobot_v3", "parquet_json", "mcap"],
                "generate_dataset_card": True,
                "render_qa_overlay": False,
                "export_3d_viewer": False,
                "generate_html_report": False,
            },
            "privacy": {"enable_face_blur": False, "strip_audio": True, "use_nvenc": False},
        },
    )

    pipe = EgoAnnotateV3Pipeline(cfg=cfg)
    ep1_out = tmp_path / "runs" / "ep_alpha"
    ep2_out = tmp_path / "runs" / "ep_beta"
    r1 = pipe.run_video(vid1, ep1_out, episode_id="ep_alpha", use_cache=False)
    r2 = pipe.run_video(vid2, ep2_out, episode_id="ep_beta", use_cache=False)
    assert r1["exports"]["mcap"]["messages"] == 30
    assert (ep1_out / "ep_alpha.mcap").read_bytes().startswith(b"\x89MCAP0\r\n")
    assert (ep1_out / "dataset_card.json").exists()
    card = json.loads((ep1_out / "dataset_card.json").read_text(encoding="utf-8"))
    assert len(card["commercial_license_manifest"]) >= 5
    assert r2["exports"]["lerobot_v3"]["frames"] == 30

    gate_single = validate_lerobot_v3_dataset(ep1_out / "lerobot_v3", run_smoke_train=True, smoke_epochs=2)
    assert gate_single["passed"] is True
    assert gate_single["audio_stripped"] is True

    for d in (ep1_out, ep2_out):
        rep_p = d / "run_report.json"
        rep = json.loads(rep_p.read_text(encoding="utf-8"))
        rep["quality"]["verdict"] = "PASS"
        rep_p.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    sharded_dir = tmp_path / "sharded_dataset"
    shard_res = pack_sharded_lerobot_v3(
        [ep1_out, ep2_out],
        sharded_dir,
        max_frames_per_shard=1000,
        run_validation_gate=True,
    )
    assert shard_res["total_episodes"] == 2
    assert shard_res["total_frames"] == 60
    assert shard_res["validation_gate"]["passed"] is True
    assert (sharded_dir / "meta" / "tasks.parquet").exists()
    assert (sharded_dir / "meta" / "subtasks.parquet").exists()
    ep_tbl = pq.read_table(sharded_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    assert ep_tbl.num_rows == 2
    assert ep_tbl["dataset_to_index"].to_pylist() == [30, 60]
    data_tbl = pq.read_table(sharded_dir / "data" / "chunk-000" / "file-000.parquet")
    assert "retargeted.franka_panda.qpos" in data_tbl.column_names
    assert "retargeted.unitree_g1_upper.qpos" in data_tbl.column_names
    assert "action.canonical_delta" in data_tbl.column_names
    assert "observation.proprio_valid" in data_tbl.column_names


def test_ray_streaming_orchestrator_registers_pools(tmp_path: Path) -> None:
    orch = RayStreamingOrchestrator(tmp_path / "ray_out", profile="cpu_smoke")
    assert len(STAGE_ACTOR_POOLS) == 5
    res = orch.run_batch([], shard_lerobot=False)
    assert res["engine"] == "empty"
