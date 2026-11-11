"""Tests for First-Party WAM, pi0-FAST DCT Tokenization, EgoMimic, VRB, EgoDex 48-D, and Open-X 7-D."""
import json
from pathlib import Path
import h5py
import numpy as np


def test_zero_dreamzero_branding_in_source_and_configs():
    for root_dir in [Path("src"), Path("configs"), Path("scripts")]:
        for p in root_dir.rglob("*"):
            if p.is_file() and p.suffix in (".py", ".yaml", ".json"):
                txt = p.read_text(encoding="utf-8", errors="ignore").lower()
                assert "dreamzero" not in txt, f"Found forbidden branding in {p}"


def test_wam_fast_dct_vrb_egomimic_egodex_oxe_tensors():
    ep_dir = Path("data/output/client_video")
    npz_path = ep_dir / "wam" / "wam_tensors.npz"
    assert npz_path.exists(), "Missing wam/wam_tensors.npz"
    with np.load(npz_path, allow_pickle=True) as data:
        n = len(data["state_full"])
        assert data["abs_action_chunks_24"].shape == (n, 24, 8)
        assert data["rel_action_chunks_24"].shape == (n, 24, 8)
        assert data["fast_dct_coeffs"].shape == (n, 24, 8)
        assert data["fast_dct_tokens"].shape == (n, 192)
        assert data["future_world_25"].shape == (n, 25, 16)
        assert data["vrb_post_contact_trajectories_uv"].shape == (n, 12, 2)
        assert data["egomimic_cam_frame_future_12"].shape == (n, 12, 2)
        assert data["egodex_48d_actions"].shape == (n, 48)
        assert data["oxe_cartesian_delta_7d"].shape == (n, 7)
        assert np.all(data["precog_collision_clear"])


def test_wam_metadata_files_exist_and_valid():
    for root in [Path("data/output/client_video/lerobot_v3"), Path("data/output/merged_lerobot_v3")]:
        meta = root / "meta"
        assert (meta / "embodiment.json").exists()
        assert (meta / "modality.json").exists()
        assert (meta / "stats.json").exists()
        assert (meta / "relative_stats_wam.json").exists()
        assert not (meta / "relative_stats_dreamzero.json").exists()


def test_wam_rlds_hdf5_integration():
    h5_path = Path("data/output/client_video/episode_hdf5.h5")
    with h5py.File(h5_path, "r") as hf:
        ep_grp = hf[list(hf.keys())[0]]
        steps = ep_grp["steps"]
        assert "wam" in steps
        wam = steps["wam"]
        for req_k in (
            "rel_action_chunks_24",
            "fast_dct_tokens",
            "vrb_post_contact_trajectories_uv",
            "egomimic_cam_frame_future_12",
            "egodex_48d_actions",
            "oxe_cartesian_delta_7d",
        ):
            assert req_k in wam, f"Missing {req_k} in HDF5 wam group"


def test_strict_30hz_uniformity_across_all_episodes():
    info_p = Path("data/output/merged_lerobot_v3/meta/info.json")
    assert info_p.exists()
    info = json.loads(info_p.read_text())
    assert float(info["fps"]) == 30.0, f"Expected 30.0 Hz, got {info['fps']}"



def test_centific_5_layer_roboverify_certification():
    cert_p = Path("data/output/client_video/wam/roboverify_certificate.json")
    tok_p = Path("data/output/client_video/wam/pose_tokens.jsonl")
    assert cert_p.exists(), "Missing roboverify_certificate.json"
    assert tok_p.exists() and tok_p.stat().st_size > 100, "Missing or empty pose_tokens.jsonl"
    cert = json.loads(cert_p.read_text())
    assert cert["overall_qc_verdict"] in ("VERIFIED_OK", "PASS_VERIFIED", "CERTIFIED_PRODUCTION_GOLD")
    assert cert["layer_1_21dof_detection_and_4pct_yield_qc"]["reprojection_p90_px"] <= 35.0
    assert cert["layer_2_kinematic_certificate"]["solver_tracking_error_mean_deg"] <= 1.40
    assert cert["roboverify_bench_calibration"]["right_pairings_score"].startswith("10 / 10")
    assert cert["roboverify_bench_calibration"]["wrong_pairings_rejected"].startswith("10 / 10")


def test_nebius_physical_ai_workbench_contract():
    from safetensors.numpy import load_file as load_safetensors
    st_p = Path("data/output/client_video/wam/wam_tensors.safetensors")
    cond_p = Path("data/output/client_video/wam/npa_cosmos_conditioning.json")
    lancedb_p = Path("data/output/merged_lerobot_v3/meta/lancedb_trajectory_index.parquet")
    npa_yaml = Path("configs/nebius/npa_workbench_pipeline.yaml")
    run_state_p = Path("NPA_WORKBENCH_RUN_STATE.json")

    assert st_p.exists(), "Missing wam_tensors.safetensors"
    assert cond_p.exists(), "Missing npa_cosmos_conditioning.json"
    assert lancedb_p.exists(), "Missing lancedb_trajectory_index.parquet"
    assert npa_yaml.exists(), "Missing npa_workbench_pipeline.yaml"
    assert run_state_p.exists(), "Missing NPA_WORKBENCH_RUN_STATE.json"

    tensors = load_safetensors(str(st_p))
    assert "long_wam_memory_32x16" in tensors
    assert tensors["long_wam_memory_32x16"].shape[1:] == (32, 16)
