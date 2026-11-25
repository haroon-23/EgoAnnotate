"""Phase 3 validation: full 5400-frame loader conformance + real-video edge-case stress suite."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq
import pytest

from src.egoannotate.behavior import extract_grasp_events, segment_atomic_actions
from src.egoannotate.config import load_config
from src.egoannotate.geometry_world import MonocularWorldVO
from src.egoannotate.quality import score_episode_quality
from src.egoannotate.retargeting_v3 import retarget_embodiment_from_yaml
from src.egoannotate.schemas import EpisodeIR, FrameAnnotation, HandState, ScaleProvenance
from src.egoannotate.video_io import (
    VideoStreamReader,
    compute_frame_image_quality,
    estimate_camera_intrinsics,
)

FULL_SAMPLE_VIDEO = Path("data/sample_raw/ad521ca5-0017-475b-9dd8-1fdb423a66b7.mp4")
CERTIFIED_RUN_DIR = Path("data/v3_certified_full_run")


def test_full_5400_frame_lerobot_groot_rlds_hdf5_conformance() -> None:
    """Verify strict schema, dtype, finiteness, and ffprobe packet parity across all 4 export formats."""
    lr_dir = CERTIFIED_RUN_DIR / "lerobot_v3"
    pq_path = lr_dir / "data" / "chunk-000" / "file-000.parquet"
    mp4_path = lr_dir / "videos" / "observation.images.ego" / "chunk-000" / "file-000.mp4"
    if not pq_path.exists() or not mp4_path.exists():
        pytest.skip("Certified 5400-frame export directory not present")

    # 1. LeRobot v3 Parquet dtype & finiteness checks
    tbl = pq.read_table(pq_path)
    n_rows = tbl.num_rows
    assert n_rows >= 5398
    for col in ("observation.state", "action", "observation.tactile", "observation.camera_pose_world"):
        arr = np.asarray(tbl[col].to_pylist(), dtype=np.float32)
        assert arr.shape[0] == n_rows
        assert np.all(np.isfinite(arr)), f"Non-finite values found in {col}"

    # 2. GR00T modality.json slice bounds match exported tensor shapes exactly
    modality = json.loads((lr_dir / "meta" / "modality.json").read_text(encoding="utf-8"))
    state_dim = len(tbl["observation.state"][0].as_py())
    action_dim = len(tbl["action"][0].as_py())
    tactile_dim = len(tbl["observation.tactile"][0].as_py())
    assert max(v["end"] for v in modality["state"].values()) == state_dim
    assert max(v["end"] for v in modality["action"].values()) == action_dim
    assert max(v["end"] for v in modality["tactile"].values()) == tactile_dim

    # 3. ffprobe packet parity & zero audio streams on sanitized LeRobot MP4
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-of",
            "json",
            str(mp4_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    streams = json.loads(probe.stdout)["streams"]
    v_streams = [s for s in streams if s.get("codec_type") == "video"]
    a_streams = [s for s in streams if s.get("codec_type") == "audio"]
    assert len(v_streams) == 1
    assert len(a_streams) == 0, "Audio stream must be stripped (-an)"
    nb_frames = int(v_streams[0].get("nb_frames") or 0)
    assert abs(nb_frames - n_rows) <= 1, f"Video/Parquet frame mismatch: mp4={nb_frames} vs parquet={n_rows}"

    # 4. RLDS/OXE conformance
    rlds = np.load(CERTIFIED_RUN_DIR / "rlds" / "episode_000000.npz")
    assert rlds["observation_state"].shape == (n_rows, state_dim)
    assert rlds["action"].shape == (n_rows, action_dim)
    assert bool(rlds["is_first"][0]) is True
    assert bool(np.sum(rlds["is_first"]) == 1)
    assert bool(rlds["is_last"][-1]) is True
    assert bool(rlds["is_terminal"][-1]) is True
    assert np.all(np.isfinite(rlds["observation_state"]))

    # 5. Canonical HDF5 conformance across all 5 embodiments
    with h5py.File(CERTIFIED_RUN_DIR / "episode_canonical.h5", "r") as hf:
        assert int(hf.attrs["total_frames"]) == n_rows
        assert bool(hf.attrs["is_full_video_processed"]) is True
        for emb in ("franka_panda", "dual_arm_franka", "unitree_g1_upper", "allegro_hand_right", "teleop_glove"):
            qpos = np.asarray(hf["retargeted"][emb]["qpos"][:])
            acts = np.asarray(hf["retargeted"][emb]["actions"][:])
            assert qpos.shape[0] == n_rows
            assert acts.shape == qpos.shape
            assert np.all(np.isfinite(qpos)) and np.all(np.isfinite(acts))


def test_edge_cases_low_light_and_motion_blur_on_real_video_samples() -> None:
    """Apply severe low-light and directional motion blur to real frames sampled across the full 180s timeline."""
    if not FULL_SAMPLE_VIDEO.exists():
        pytest.skip("Full sample video not available locally")

    cfg = load_config(profile="cpu_smoke")
    reader = VideoStreamReader(FULL_SAMPLE_VIDEO)
    samples = reader._read_evenly_spaced_samples(count=12)
    assert len(samples) >= 6

    normal_frames: list[FrameAnnotation] = []
    dark_frames: list[FrameAnnotation] = []
    blur_frames: list[FrameAnnotation] = []

    # 31px horizontal motion-blur kernel
    ksize = 31
    mblur_k = np.zeros((ksize, ksize), dtype=np.float32)
    mblur_k[ksize // 2, :] = 1.0 / float(ksize)

    for idx, bgr in enumerate(samples):
        ts = idx / 30.0
        s_norm, l_norm = compute_frame_image_quality(bgr)
        dark = np.clip(bgr.astype(np.float32) * 0.08, 0, 255).astype(np.uint8)
        s_dark, l_dark = compute_frame_image_quality(dark)
        blurred = cv2.filter2D(bgr, -1, mblur_k)
        s_blur, l_blur = compute_frame_image_quality(blurred)

        rh = HandState(handedness="right", detected=True, confidence=0.85, visibility=0.95)
        normal_frames.append(
            FrameAnnotation(frame_idx=idx, timestamp_s=ts, sharpness_laplacian=s_norm, mean_luma=l_norm, right_hand=rh)
        )
        dark_frames.append(
            FrameAnnotation(frame_idx=idx, timestamp_s=ts, sharpness_laplacian=s_dark, mean_luma=l_dark, right_hand=rh)
        )
        blur_frames.append(
            FrameAnnotation(frame_idx=idx, timestamp_s=ts, sharpness_laplacian=s_blur, mean_luma=l_blur, right_hand=rh)
        )

    sp = ScaleProvenance(scale_meters_per_unit=1.0, scale_uncertainty_pct=12.0, is_calibrated_metric=False)
    q_norm = score_episode_quality(normal_frames, [], sp, cfg.quality)
    q_dark = score_episode_quality(dark_frames, [], sp, cfg.quality)
    q_blur = score_episode_quality(blur_frames, [], sp, cfg.quality)

    assert q_dark.dimensions.lighting_quality < q_norm.dimensions.lighting_quality
    assert "underexposed_low_light" in q_dark.flags
    assert q_blur.dimensions.motion_blur_quality < q_norm.dimensions.motion_blur_quality
    assert "severe_motion_blur" in q_blur.flags
    assert q_dark.robot_readiness_score < q_norm.robot_readiness_score
    assert q_blur.robot_readiness_score < q_norm.robot_readiness_score


def test_edge_case_occlusion_and_missing_vs_calibrated_intrinsics(tmp_path: Path) -> None:
    """Verify severe occlusion triggers quarantine and intrinsics calibration provenance updates honestly."""
    cfg = load_config(profile="cpu_smoke")

    # 1. Missing vs Calibrated intrinsics YAML
    est_auto = estimate_camera_intrinsics(1024, 576, sample_frames=None, calibration_yaml=None)
    assert est_auto.source == "auto_estimated"
    assert est_auto.uncertainty_pct == 15.0

    cal_yaml = tmp_path / "calibrated_cam.yaml"
    cal_yaml.write_text("fx: 720.0\nfy: 720.0\ncx: 512.0\ncy: 288.0\ndist_coeffs: [-0.04, 0.01, 0, 0, 0]\n", encoding="utf-8")
    est_cal = estimate_camera_intrinsics(1024, 576, sample_frames=None, calibration_yaml=cal_yaml)
    assert est_cal.source == "calibrated_yaml"
    assert est_cal.uncertainty_pct == 2.0

    vo = MonocularWorldVO(cfg.geometry, est_cal)
    prov_depth = vo.build_scale_provenance(est_cal, "depth_anything_v2_small_apache")
    assert prov_depth.scale_source == "calibrated_intrinsics_and_metric_depth"
    assert prov_depth.is_calibrated_metric is True
    assert prov_depth.scale_uncertainty_pct == 5.0

    # 2. Occluded / heavily truncated hand sequence
    occ_frames = [
        FrameAnnotation(
            frame_idx=i,
            timestamp_s=i / 30.0,
            sharpness_laplacian=140.0,
            mean_luma=120.0,
            right_hand=HandState(handedness="right", detected=(i % 5 == 0), confidence=0.45, visibility=0.4, truncated=True),
        )
        for i in range(60)
    ]
    q_occ = score_episode_quality(occ_frames, [], prov_depth, cfg.quality)
    assert "low_hand_visibility" in q_occ.flags
    assert "frequent_hand_truncation_or_occlusion" in q_occ.flags
    assert q_occ.dimensions.occlusion_score == 0.0


def test_edge_case_unimanual_vs_bimanual_on_real_5400_frame_ir() -> None:
    """Verify unimanual (left-only, right-only) vs bimanual coordination classification on real 5400-frame data."""
    ir_path = CERTIFIED_RUN_DIR / "parquet_json" / "episode_ir.json"
    if not ir_path.exists():
        pytest.skip("Certified 5400-frame IR JSON not present")

    cfg = load_config(profile="cpu_smoke")
    ir = EpisodeIR.model_validate_json(ir_path.read_text(encoding="utf-8"))
    assert len(ir.frames) >= 5398

    # Original real video contains bimanual manipulation
    bimanual_segs = [s for s in ir.segments if s.acting_hand == "bimanual"]
    assert len(bimanual_segs) >= 1

    # Mask left hand across a 300-frame real slice to verify strict right-only unimanual behavior
    right_only_slice = [f.model_copy(deep=True) for f in ir.frames[300:600]]
    for fr in right_only_slice:
        fr.left_hand = HandState(handedness="left", detected=False)
        for o in fr.objects:
            if o.in_hand in ("left", "both"):
                o.in_hand = "right" if o.in_hand == "both" else "none"
    g_right = extract_grasp_events(right_only_slice, cfg.behavior)
    s_right = segment_atomic_actions(right_only_slice, g_right, cfg.behavior)
    assert all(s.acting_hand in ("right", "none") for s in s_right)
    assert any(s.bimanual_coordination == "unimanual_right" for s in s_right)


def test_edge_case_mjcf_xml_config_only_retargeting(tmp_path: Path) -> None:
    """Verify pure config-driven MJCF (.xml) robot retargeting works alongside URDF."""
    mjcf = tmp_path / "mini_arm.xml"
    mjcf.write_text(
        """<mujoco model="mini_arm">
  <worldbody>
    <body name="link1" pos="0 0 0.20">
      <joint name="j1" type="hinge" axis="0 0 1" range="-2.8 2.8"/>
      <body name="link2" pos="0 0 0.15">
        <joint name="j2" type="hinge" axis="0 1 0" range="-1.8 1.8"/>
        <body name="link3" pos="0.30 0 0">
          <joint name="j3" type="hinge" axis="0 1 0" range="-2.2 2.2"/>
          <body name="ee_site" pos="0.25 0 0"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    cfg_yaml = tmp_path / "mini_arm.yaml"
    cfg_yaml.write_text(
        f"""name: mini_mjcf_arm
embodiment_class: single_arm_gripper
urdf_path: {mjcf}
arms:
  - side: right
    ee_link: ee_site
    workspace_center_m: [0.40, 0.0, 0.30]
    workspace_scale: 0.65
    table_z_m: 0.05
    max_joint_vel_radps: 2.5
gripper:
  dof: 1
  min_opening_m: 0.0
  max_opening_m: 0.08
""",
        encoding="utf-8",
    )

    frames = [
        FrameAnnotation(
            frame_idx=i,
            timestamp_s=i / 30.0,
            right_hand=HandState(
                handedness="right",
                detected=True,
                confidence=0.9,
                visibility=1.0,
                wrist_pos_world_m=[0.35 + 0.03 * float(np.sin(i / 10.0)), 0.02 * float(np.cos(i / 10.0)), 0.25],
                gripper_aperture_m=0.04,
                finger_curls_5d=[0.3, 0.3, 0.3, 0.3, 0.3],
            ),
        )
        for i in range(45)
    ]
    cfg = load_config(profile="cpu_smoke")
    traj = retarget_embodiment_from_yaml(cfg_yaml, frames, [], cfg.retargeting)
    assert traj.qpos.shape == (45, 4)
    assert traj.metrics["joint_limit_compliance_pct"] == 100.0
    assert traj.metrics["mean_fk_pos_error_mm"] < 25.0
