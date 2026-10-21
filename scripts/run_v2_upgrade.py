import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import cv2
import h5py
import numpy as np

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.retargeting.universal_multi_retargeter import compile_all_embodiments_for_episode
from src.rlds_exporter import export_to_rlds
from scripts.validate_delivery import validate_episode_dir

print("======================================================================")
print("  EGOANNOTATE v2.0.0: UNIVERSAL MULTI-EMBODIMENT COMPILER INITIALIZED")
print("======================================================================")

episodes = ["whatsapp_video", "test_10s", "my_video", "client_video", "new_video"]
all_reports = {}

for ep_id in episodes:
    ep_dir = Path("data/output") / ep_id
    ann_path = ep_dir / "annotations.json"
    if not ann_path.exists():
        continue
    print(f"\nCompiling universal 4-robot trajectories & quad-view proof video for {ep_id} ...")
    rep_hw = compile_all_embodiments_for_episode(ann_path, ep_dir, render_multi_video=True)
    all_reports[ep_id] = rep_hw
    export_to_rlds(ep_id, "data/output")
    val_rep = validate_episode_dir(ep_dir)
    multi_vid = ep_dir / "multi_embodiment_proof.mp4"
    print(
        f"[UNIVERSAL READY] {ep_id}: robots={list(rep_hw.keys())} | "
        f"multi_vid={multi_vid.exists()} ({multi_vid.stat().st_size // 1024 if multi_vid.exists() else 0} KB) | "
        f"delivery_passed={val_rep['passed']}"
    )

# ---------------------------------------------------------------------------
# Re-merge LeRobot v3 dataset
# ---------------------------------------------------------------------------
print("\n--- Merging LeRobot v3 multi-episode dataset ---")
if Path("data/output/merged_lerobot_v3").exists():
    shutil.rmtree("data/output/merged_lerobot_v3")
subprocess.run(".venv/bin/python scripts/run.py --merge-lerobot --config configs/default.yaml", shell=True, check=True)

# ---------------------------------------------------------------------------
# Update EGO_AUDIT_DOSSIER.json with v2.0.0 multi-embodiment compilation metrics
# ---------------------------------------------------------------------------
dossier_p = Path("EGO_AUDIT_DOSSIER.json")
if dossier_p.exists():
    dos = json.loads(dossier_p.read_text(encoding="utf-8"))
    dos["project"] = "EgoAnnotate v2.0.0 (Universal Multi-Embodiment VLA & Robot Compiler)"
    for ep_id in episodes:
        if ep_id in dos.get("episodes", {}) and ep_id in all_reports:
            dos["episodes"][ep_id]["universal_embodiments"] = {
                rname: {
                    "kind": rinfo["kind"],
                    "dof": rinfo["dof"],
                    "hardware_score": rinfo["hardware_audit"]["hardware_readiness_score"],
                    "verdict": rinfo["hardware_audit"]["verdict"],
                    "mean_ik_residual_mm": rinfo["summary"]["mean_ik_residual_mm"],
                }
                for rname, rinfo in all_reports[ep_id].items()
            }
            dos["episodes"][ep_id]["multi_embodiment_proof_video"] = f"data/output/{ep_id}/multi_embodiment_proof.mp4"
    dossier_p.write_text(json.dumps(dos, indent=2) + "\n", encoding="utf-8")
    print("[OK] Updated EGO_AUDIT_DOSSIER.json to v2.0.0 (Universal Multi-Embodiment)")

# ---------------------------------------------------------------------------
# Add test to tests/test_visual_and_arm_upgrades.py if not already present
# ---------------------------------------------------------------------------
test_p = Path("tests/test_visual_and_arm_upgrades.py")
test_txt = test_p.read_text(encoding="utf-8")
if "test_universal_multi_retargeter_exports_all_four_robots_into_rlds" not in test_txt:
    test_txt += '''

def test_universal_multi_retargeter_exports_all_four_robots_into_rlds():
    import h5py
    ep_dir = Path("data/output/whatsapp_video")
    assert (ep_dir / "multi_embodiment_proof.mp4").exists()
    for rname in ("franka_panda", "unitree_g1_upper", "unitree_h1_upper", "allegro_hand_right"):
        npz_p = ep_dir / "embodiments" / rname / "trajectory.npz"
        assert npz_p.exists(), f"Missing trajectory bundle for {rname}"
    with h5py.File(ep_dir / "episode_rlds.hdf5", "r") as hf:
        ep_grp = hf[list(hf.keys())[0]]
        steps = ep_grp["steps"]
        assert "reward" in steps
        assert "embodiments" in steps
        for rname in ("franka_panda", "unitree_g1_upper", "unitree_h1_upper", "allegro_hand_right"):
            assert rname in steps["embodiments"]
            assert "q" in steps["embodiments"][rname]
            assert "action_chunks_16" in steps["embodiments"][rname]
'''
    test_p.write_text(test_txt, encoding="utf-8")
    print("[OK] Added universal multi-embodiment HDF5 & NPZ test to tests/test_visual_and_arm_upgrades.py")

# ---------------------------------------------------------------------------
# Run full pytest suite
# ---------------------------------------------------------------------------
print("\n--- Running full pytest suite ---")
res = subprocess.run(
    [
        ".venv/bin/pytest",
        "tests/test_visual_and_arm_upgrades.py",
        "tests/test_real_robot_readiness.py",
        "tests/test_phase1_core.py",
        "tests/test_phase2_embodiments.py",
        "tests/test_phase3_ship.py",
        "tests/test_delivery_validation.py",
        "tests/test_dataset_exporter.py",
        "tests/test_rlds_exporter.py",
        "-v",
    ],
    capture_output=True,
    text=True,
)
print(res.stdout)
if res.returncode != 0:
    print(res.stderr)
    sys.exit(res.returncode)
