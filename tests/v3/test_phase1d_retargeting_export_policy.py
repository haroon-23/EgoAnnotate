"""Phase 1D verification on the COMPLETE 5400-frame video: config-only retargeting, multi-format export & policy train."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
import pytest

from src.egoannotate.config import load_config
from src.egoannotate.pipeline_v3 import EgoAnnotateV3Pipeline
from src.egoannotate.policy_train_v3 import run_policy_training_smoke
from src.egoannotate.retargeting_v3 import retarget_embodiment_from_yaml
from src.egoannotate.schemas import EpisodeIR, FrameAnnotation, HandState

FULL_SAMPLE_VIDEO = Path("data/sample_raw/ad521ca5-0017-475b-9dd8-1fdb423a66b7.mp4")


def test_zero_code_onboarding_of_brand_new_robot_urdf_yaml(tmp_path: Path) -> None:
    """Prove a customer can onboard a brand-new 6-DoF robot strictly via URDF + YAML with zero code changes."""
    custom_urdf = tmp_path / "custom_6dof_arm.urdf"
    custom_urdf.write_text(
        """<?xml version="1.0"?>
<robot name="custom_6dof_arm">
  <link name="base_link"/>
  <link name="link1"/><link name="link2"/><link name="link3"/>
  <link name="link4"/><link name="link5"/><link name="ee_link"/>
  <joint name="j1" type="revolute"><parent link="base_link"/><child link="link1"/>
    <origin xyz="0 0 0.15" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3.0" upper="3.0" effort="50" velocity="2"/></joint>
  <joint name="j2" type="revolute"><parent link="link1"/><child link="link2"/>
    <origin xyz="0 0 0.12" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-1.8" upper="1.8" effort="50" velocity="2"/></joint>
  <joint name="j3" type="revolute"><parent link="link2"/><child link="link3"/>
    <origin xyz="0.30 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-2.4" upper="2.4" effort="50" velocity="2"/></joint>
  <joint name="j4" type="revolute"><parent link="link3"/><child link="link4"/>
    <origin xyz="0.25 0 0" rpy="0 0 0"/><axis xyz="1 0 0"/><limit lower="-3.0" upper="3.0" effort="20" velocity="2"/></joint>
  <joint name="j5" type="revolute"><parent link="link4"/><child link="link5"/>
    <origin xyz="0.08 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-2.0" upper="2.0" effort="20" velocity="2"/></joint>
  <joint name="j6" type="revolute"><parent link="link5"/><child link="ee_link"/>
    <origin xyz="0.06 0 0" rpy="0 0 0"/><axis xyz="1 0 0"/><limit lower="-3.0" upper="3.0" effort="20" velocity="2"/></joint>
</robot>
""",
        encoding="utf-8",
    )
    custom_yaml = tmp_path / "custom_6dof_arm.yaml"
    custom_yaml.write_text(
        f"""name: custom_6dof_arm
embodiment_class: single_arm_gripper
urdf_path: {custom_urdf}
arms:
  - side: right
    ee_link: ee_link
    workspace_center_m: [0.42, 0.0, 0.28]
    workspace_scale: 0.70
    table_z_m: 0.02
    max_joint_vel_radps: 2.5
gripper:
  dof: 1
  min_opening_m: 0.0
  max_opening_m: 0.08
""",
        encoding="utf-8",
    )

    frames = []
    for i in range(60):
        theta = i / 20.0
        rh = HandState(
            handedness="right",
            detected=True,
            confidence=0.95,
            visibility=1.0,
            wrist_pos_world_m=[0.35 + 0.05 * float(np.cos(theta)), 0.05 * float(np.sin(theta)), 0.25],
            gripper_aperture_m=0.04,
            finger_curls_5d=[0.3, 0.4, 0.4, 0.4, 0.4],
        )
        frames.append(FrameAnnotation(frame_idx=i, timestamp_s=i / 30.0, right_hand=rh))

    cfg = load_config(profile="cpu_smoke")
    traj = retarget_embodiment_from_yaml(custom_yaml, frames, [], cfg.retargeting)
    assert traj.qpos.shape == (60, 7)
    assert traj.metrics["joint_limit_compliance_pct"] == 100.0
    assert traj.metrics["mean_fk_pos_error_mm"] < 20.0
    assert traj.metrics["overall_feasibility"] >= 0.80


def test_complete_5400_frame_pipeline_retargeting_exports_and_policy_training() -> None:
    if not FULL_SAMPLE_VIDEO.exists():
        pytest.skip("Full Drive sample video not available locally")

    out_dir = Path("data/v3_certified_full_run")
    if out_dir.exists():
        shutil.rmtree(out_dir)

    cfg = load_config(profile="cpu_smoke")
    cfg.export.video_crf = 28
    pipe = EgoAnnotateV3Pipeline(cfg=cfg)
    report = pipe.run_video(FULL_SAMPLE_VIDEO, out_dir, episode_id="drive_full_180s", use_cache=True)

    # 1. Full-video certification
    assert report["video"]["is_full_video_processed"] is True
    assert report["video"]["processed_frames"] >= 5398
    assert report["privacy_audit"]["verified_unblurred_faces"] == 0
    assert report["privacy_audit"]["audio_stripped"] is True

    # 2. All 5 embodiment classes retargeted across all 5400 frames
    for emb in ("franka_panda", "dual_arm_franka", "unitree_g1_upper", "allegro_hand_right", "teleop_glove"):
        assert emb in report["retargeting"]
        m = report["retargeting"][emb]
        assert m["frames"] >= 5398
        assert m["joint_limit_compliance_pct"] >= 99.9
        assert m["overall_feasibility"] >= 0.65

    # 3. Verify LeRobot v3 Parquet + GR00T modality.json
    lr_dir = out_dir / "lerobot_v3"
    pq_tbl = pq.read_table(lr_dir / "data" / "chunk-000" / "file-000.parquet")
    assert pq_tbl.num_rows >= 5398
    assert (lr_dir / "meta" / "info.json").exists()
    assert (lr_dir / "meta" / "stats.json").exists()
    assert (lr_dir / "meta" / "modality.json").exists()
    info = json.loads((lr_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["total_frames"] == pq_tbl.num_rows
    assert (lr_dir / "videos" / "observation.images.ego" / "chunk-000" / "file-000.mp4").stat().st_size > 100_000

    # 4. Verify RLDS / OXE export
    rlds_npz = np.load(out_dir / "rlds" / "episode_000000.npz")
    assert rlds_npz["observation_state"].shape[0] == pq_tbl.num_rows
    assert bool(rlds_npz["is_first"][0]) is True
    assert bool(rlds_npz["is_terminal"][-1]) is True

    # 5. Verify Canonical HDF5 export
    with h5py.File(out_dir / "episode_canonical.h5", "r") as hf:
        assert int(hf.attrs["total_frames"]) == pq_tbl.num_rows
        assert bool(hf.attrs["is_full_video_processed"]) is True
        assert "franka_panda" in hf["retargeted"]
        assert "unitree_g1_upper" in hf["retargeted"]
        assert "teleop_glove" in hf["retargeted"]

    # 6. Verify Pydantic IR JSON round-trip
    ir_loaded = EpisodeIR.model_validate_json((out_dir / "parquet_json" / "episode_ir.json").read_text(encoding="utf-8"))
    assert len(ir_loaded.frames) == pq_tbl.num_rows

    # 7. Real PyTorch CPU policy training smoke test over all 5400 frames
    train_res = run_policy_training_smoke(lr_dir, chunk_size=8, epochs=6, batch_size=128)
    assert train_res["samples"] >= 5390
    assert train_res["loss_decreased"] is True
    assert train_res["finite_predictions"] is True

    print(
        f"\n[PHASE 1D COMPLETE 5400-FRAME PIPELINE]\n"
        f"  Frames: {pq_tbl.num_rows}/5400 (180.0s), Quality: {report['quality']['robot_readiness_score']:.3f} ({report['quality']['verdict']})\n"
        f"  Privacy: unblurred={report['privacy_audit']['verified_unblurred_faces']}, audio_stripped={report['privacy_audit']['audio_stripped']}\n"
        f"  Retargeting feasibility: "
        + ", ".join(f"{k}={v['overall_feasibility']:.3f} (FK={v['mean_fk_pos_error_mm']}mm)" for k, v in report["retargeting"].items())
        + f"\n  Policy Train (LeRobot v3): samples={train_res['samples']}, loss {train_res['initial_loss']:.4f} -> {train_res['final_loss']:.4f}\n"
        f"  Timings: {report['timings_s']}"
    )
