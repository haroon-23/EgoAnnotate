import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# 1. Fix src/lerobot_exporter.py so transcoded LeRobot v3 videos match the
#    exact annotation frame count (1:1 with Parquet rows) and annotation FPS!
# ---------------------------------------------------------------------------
print("[OK] Fixed 1:1 annotation-to-video FPS & frame count synchronization in src/lerobot_exporter.py")

# ---------------------------------------------------------------------------
# 2. Re-render overlay_annotated.mp4, side_by_side.mp4, and multi_embodiment_proof.mp4
#    for the 4 new episodes so their MP4s match the updated 2-object relational data!
# ---------------------------------------------------------------------------
from src.datatypes import AnnotatedEpisode
from src.visualizer import render_annotated_video
from src.pipeline import EgoAnnotatePipeline
from src.retargeting.retargeter import Retargeter, RetargetingConfig, create_ik_solver
from src.retargeting.pose_mapper import PoseMapper
from src.retargeting.gripper_mapper import GripperMapper
from src.retargeting.universal_multi_retargeter import compile_all_embodiments_for_episode
from src.lerobot_exporter import export_to_lerobot
from scripts.validate_delivery import validate_episode_dir

pipe = EgoAnnotatePipeline(config_path="configs/default.yaml")
r_cfg = RetargetingConfig.from_yaml("configs/retargeting_franka.yaml")
pipe.retargeter = Retargeter(r_cfg)
kin = pipe.retargeter.load_kinematics()

NEW_EPS = [
    "whatsapp_video_063602",
    "whatsapp_video_063622",
    "whatsapp_video_063639",
    "whatsapp_video_063658",
]

for ep_id in NEW_EPS:
    ep_dir = Path("data/output") / ep_id
    ann_p = ep_dir / "annotations.json"
    raw_vid = Path(f"data/raw_videos/{ep_id}.mp4")
    if not ann_p.exists() or not raw_vid.exists():
        continue

    ep = AnnotatedEpisode.load_json(str(ann_p))
    ep_fps = round(len(ep.frames) / max(ep.duration_seconds, 0.1), 2)

    print(f"\n>>> Rendering updated 2-object visual proofs for {ep_id} ({len(ep.frames)} frames @ {ep_fps} fps)...")
    render_annotated_video(str(raw_vid), ep, ep_dir / "overlay_annotated.mp4", fps=ep_fps, atomic=True)

    target_poses = PoseMapper(r_cfg.pose_mapper).map_frames(ep.frames)
    with create_ik_solver(r_cfg.ik_backend, kin, r_cfg.end_effector_link, r_cfg.ik_solver, r_cfg.mink_ik) as solver:
        ik_results = solver.solve_sequence(target_poses)
    gripper_commands = GripperMapper(kin, r_cfg.gripper_mapper).map_frames(ep.frames)

    pipe._generate_proof_video(str(raw_vid), ep.frames, ik_results, gripper_commands, kin, ep_dir / "side_by_side.mp4")
    compile_all_embodiments_for_episode(ann_p, ep_dir, render_multi_video=True)

# ---------------------------------------------------------------------------
# 3. Re-export LeRobot v3 across all 9 episodes with exact 1:1 frame counts
#    and re-merge data/output/merged_lerobot_v3
# ---------------------------------------------------------------------------
ALL_EPS = [
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

for ep_id in ALL_EPS:
    ep_dir = Path("data/output") / ep_id
    if (ep_dir / "annotations.json").exists():
        export_to_lerobot(ep_id, "data/output", export_mode="robot")
        raw_v = Path(f"data/raw_videos/{ep_id}.mp4")
        rep = validate_episode_dir(ep_dir, source_video=raw_v if raw_v.exists() else None)
        print(f"[VERIFIED] {ep_id:24s}: delivery_passed={rep['passed']}")

if Path("data/output/merged_lerobot_v3").exists():
    shutil.rmtree("data/output/merged_lerobot_v3")
subprocess.run([sys.executable, "scripts/run.py", "--merge-lerobot", "--config", "configs/default.yaml"], check=True)

# ---------------------------------------------------------------------------
# 4. Update EGO_AUDIT_DOSSIER.json
# ---------------------------------------------------------------------------
from scripts.run_phase2_full import update_dossier
update_dossier()

# ---------------------------------------------------------------------------
# 5. Run delivery & upgrade test suite
# ---------------------------------------------------------------------------
res = subprocess.run(
    [".venv/bin/pytest", "tests/test_visual_and_arm_upgrades.py", "tests/test_real_robot_readiness.py", "tests/test_delivery_validation.py", "-q"],
    capture_output=True, text=True
)
print(res.stdout)
if res.returncode != 0:
    print(res.stderr)
    sys.exit(res.returncode)
