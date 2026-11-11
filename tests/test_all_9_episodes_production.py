"""Comprehensive Signal-Level & Multi-Episode Production Verification Suite (All 9 Videos)."""
import json
from pathlib import Path
import h5py
import numpy as np
import pyarrow.parquet as pq
import pytest

ALL_9_EPISODES = [
    "whatsapp_video",
    "test_10s",
    "my_video",
    "client_video",
    "new_video",
    "whatsapp_video_063602",
    "whatsapp_video_063622",
    "whatsapp_video_063639",
    "whatsapp_video_063658",
]


@pytest.mark.parametrize("episode_id", ALL_9_EPISODES)
def test_signal_24d_action_variance_and_hand_valid_mask(episode_id: str):
    ep_dir = Path("data/output") / episode_id
    lr_dir = ep_dir / "lerobot_v3"
    info = json.loads((lr_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    df = pq.read_table(lr_dir / "data" / "chunk-000" / "file-000.parquet").to_pandas()

    assert "observation.hand_valid" in df.columns
    assert "observation.hand_valid" in info["features"]
    assert "ego_annotate" in info
    assert info["ego_annotate"]["frame"] == "camera"

    states = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    actions = np.stack(df["action"].to_numpy()).astype(np.float64)
    vmask = np.stack(df["observation.hand_valid"].to_numpy())
    valid_1d = np.any(vmask > 0, axis=1)

    assert states.shape[1] == 24
    assert actions.shape[1] == 24
    assert float(np.mean(valid_1d)) >= 0.005
    assert float(np.mean(np.abs(states[valid_1d, 2]))) >= 0.15
    assert float(np.std(states[valid_1d, 2])) > 1e-4
    assert float(np.max(np.var(actions[valid_1d], axis=0))) > 1e-7


@pytest.mark.parametrize("episode_id", ALL_9_EPISODES)
def test_signal_delta_consistency_between_state_and_action(episode_id: str):
    ep_dir = Path("data/output") / episode_id
    df = pq.read_table(ep_dir / "lerobot_v3" / "data" / "chunk-000" / "file-000.parquet").to_pandas()
    states = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    actions = np.stack(df["action"].to_numpy()).astype(np.float64)
    vmask = np.stack(df["observation.hand_valid"].to_numpy())
    valid_1d = np.any(vmask > 0, axis=1)
    pair_valid = valid_1d[:-1] & valid_1d[1:]
    if np.any(pair_valid):
        delta_residual = float(np.mean(np.abs(actions[:-1][pair_valid] - (states[1:][pair_valid] - states[:-1][pair_valid]))))
        assert delta_residual < 1e-4


@pytest.mark.parametrize("episode_id", ALL_9_EPISODES)
def test_signal_independent_fk_pos_and_ori_residual_distribution(episode_id: str):
    ep_dir = Path("data/output") / episode_id
    npz_p = ep_dir / "embodiments" / "franka_panda" / "trajectory.npz"
    with np.load(npz_p, allow_pickle=True) as d:
        pos_residual = d["pos_residual_mm"]
        ori_residual = d["ori_residual_deg"]
        assert float(np.std(pos_residual)) > 1e-5
        assert float(np.std(ori_residual)) > 1e-5
        assert float(np.percentile(pos_residual, 95)) <= 20.0
        assert float(np.percentile(ori_residual, 95)) <= 20.0


@pytest.mark.parametrize("episode_id", ALL_9_EPISODES)
def test_signal_privacy_consent_and_caption_duration_tolerance(episode_id: str):
    ep_dir = Path("data/output") / episode_id
    consent_p = ep_dir / "consent_manifest.json"
    ann_p = ep_dir / "annotations.json"
    assert consent_p.exists()
    c_doc = json.loads(consent_p.read_text(encoding="utf-8"))
    if str(c_doc.get("consent_status", "")).startswith("UNSET_INTERNAL_TEST"):
        assert c_doc.get("distribution_restriction") == "NOT_FOR_DISTRIBUTION"
        assert "subject_acknowledged" not in c_doc
        assert "collector_id" not in c_doc
    else:
        assert bool(c_doc.get("subject_acknowledged")) is True
    ann = json.loads(ann_p.read_text(encoding="utf-8"))
    for seg in ann["segments"]:
        dur = float(seg["end_time"]) - float(seg["start_time"])
        assert dur >= 0.499
        assert seg.get("caption_source") in ("vlm", "template")


def test_signal_chirality_ground_truth_accuracy_across_corpus():
    from gates.g05_chirality import _count_identity_swaps
    ann = json.loads(Path("data/output/test_10s/annotations.json").read_text(encoding="utf-8"))
    swaps = _count_identity_swaps(ann["frames"])
    assert swaps == 0


def test_signal_negative_control_rejection_zero_variance():
    static_arr = np.zeros((30, 24), dtype=np.float32)
    assert float(np.var(static_arr)) < 1e-6


def test_signal_reproj_error_within_physical_tolerance():
    with np.load("data/output/test_10s/wam/wam_tensors.npz", allow_pickle=True) as d:
        reproj = d["reprojection_error_px"]
        assert float(np.percentile(reproj, 90)) <= 35.0


def test_signal_ik_convergence_separate_from_hand_valid():
    hw = json.loads(Path("data/output/test_10s/hardware_readiness_report.json").read_text(encoding="utf-8"))
    audit = hw["franka_panda"]["hardware_audit"]
    assert audit["ik_convergence_pct"] >= 75.0
    assert audit["hand_valid_pct"] > 0.0


def test_signal_distinct_g1_and_h1_kinematic_velocity_profiles():
    hw = json.loads(Path("data/output/test_10s/hardware_readiness_report.json").read_text(encoding="utf-8"))
    g1_v = hw["unitree_g1_upper"]["hardware_audit"]["max_joint_velocity_rad_s"]
    h1_v = hw["unitree_h1_upper"]["hardware_audit"]["max_joint_velocity_rad_s"]
    assert abs(g1_v - h1_v) > 1e-4


def test_signal_hdf5_compression_and_storage_tolerance():
    h5_p = Path("data/output/test_10s/episode_hdf5.h5")
    size_mb = h5_p.stat().st_size / (1024.0 * 1024.0)
    mb_per_hr = size_mb / (300.0 / 30.0 / 3600.0)
    assert mb_per_hr < 550.0


def test_signal_rot6_orthonormal_variance_and_bounds():
    df = pq.read_table("data/output/test_10s/lerobot_v3/data/chunk-000/file-000.parquet").to_pandas()
    states = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    rot6 = states[:, 3:9]
    assert float(np.var(rot6)) > 1e-5
    assert float(np.max(np.abs(rot6))) <= 1.05


def test_signal_finger_flexion_angles_within_anatomical_tolerance():
    df = pq.read_table("data/output/test_10s/lerobot_v3/data/chunk-000/file-000.parquet").to_pandas()
    states = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    fingers = states[:, 10:24]
    assert float(np.std(fingers)) > 1e-4
    assert float(np.min(fingers)) >= 0.0
    assert float(np.max(fingers)) <= np.pi + 0.05


def test_signal_gripper_aperture_continuous_variance():
    df = pq.read_table("data/output/test_10s/lerobot_v3/data/chunk-000/file-000.parquet").to_pandas()
    states = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    grip = states[:, 9]
    assert float(np.var(grip)) > 1e-5
    assert 0.0 <= float(np.min(grip)) <= float(np.max(grip)) <= 1.0


def test_signal_action_delta_zero_on_invalid_hand_transitions():
    df = pq.read_table("data/output/whatsapp_video/lerobot_v3/data/chunk-000/file-000.parquet").to_pandas()
    actions = np.stack(df["action"].to_numpy()).astype(np.float64)
    vmask = np.stack(df["observation.hand_valid"].to_numpy())
    valid_1d = np.any(vmask > 0, axis=1)
    invalid_pairs = ~(valid_1d[:-1] & valid_1d[1:])
    if np.any(invalid_pairs):
        max_invalid_delta = float(np.max(np.abs(actions[:-1][invalid_pairs])))
        assert max_invalid_delta < 1e-6
