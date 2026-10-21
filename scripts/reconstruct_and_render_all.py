import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from src.datatypes import ActionSegment, AnnotatedEpisode, ObjectAnnotation, HandLandmarks
from src.perception.dense_object_tracker import track_objects_with_keyframe_waypoints
from src.retargeting.world_locked_arm_mapper import build_world_locked_arm_trajectory
from src.flora_engine import compute_flora_rewards_and_rubric
from src.signal_segmenter import SignalSegmenter, SignalSegmenterConfig
from src.segment_labeler import build_instruction
from src.dataset_exporter import DatasetExporter, ExporterConfig
from src.rlds_exporter import export_to_rlds
from src.lerobot_exporter import export_to_lerobot
from src.mcap_exporter import export_episode_to_mcap
from src.visualizer import render_annotated_video
from src.pipeline import EgoAnnotatePipeline
from src.retargeting.retargeter import Retargeter, RetargetingConfig, create_ik_solver
from src.retargeting.pose_mapper import PoseMapper
from src.retargeting.gripper_mapper import GripperMapper
from scripts.audit_real_robot_readiness import run_hardware_audit
from scripts.validate_delivery import validate_episode_dir

EPISODE_SPECS = {
    "whatsapp_video": {
        "raw_video": "data/raw_videos/whatsapp_video.mp4",
        "task_description": "Rearranging desktop items: standing the lotion bottle upright, aligning the cigarette pack and lighter, keyfob, and moving the earbud case.",
        "waypoints": {
            "Router & Cables": [
                (0, [0.04, 0.16, 0.77, 0.40]),
                (105, [0.03, 0.11, 0.76, 0.36]),
                (300, [0.05, 0.18, 0.76, 0.42]),
                (728, [0.05, 0.24, 0.75, 0.47]),
            ],
            "Computer Mouse": [
                (0, [0.85, 0.34, 0.99, 0.48]),
                (105, [0.87, 0.30, 0.99, 0.45]),
                (450, [0.80, 0.39, 0.94, 0.52]),
                (728, [0.76, 0.46, 0.90, 0.59]),
            ],
            "Lotion Bottle": [
                (0, [0.06, 0.56, 0.34, 0.70]),
                (105, [0.05, 0.46, 0.31, 0.59]),
                (180, [0.16, 0.53, 0.38, 0.67]),
                (310, [0.27, 0.54, 0.48, 0.68]),
                (435, [0.15, 0.55, 0.32, 0.69]),
                (570, [0.14, 0.55, 0.25, 0.67]),
                (728, [0.14, 0.62, 0.25, 0.75]),
            ],
            "Cigarette Pack & Lighter": [
                (0, [0.24, 0.60, 0.49, 0.74]),
                (105, [0.22, 0.50, 0.46, 0.63]),
                (180, [0.32, 0.55, 0.54, 0.68]),
                (330, [0.35, 0.54, 0.49, 0.66]),
                (480, [0.36, 0.53, 0.47, 0.65]),
                (728, [0.36, 0.61, 0.47, 0.74]),
            ],
            "Earbud Case": [
                (0, [0.44, 0.52, 0.60, 0.60]),
                (105, [0.35, 0.43, 0.50, 0.51]),
                (250, [0.43, 0.51, 0.58, 0.59]),
                (300, [0.64, 0.49, 0.79, 0.58]),
                (450, [0.61, 0.50, 0.73, 0.59]),
                (728, [0.69, 0.57, 0.80, 0.66]),
            ],
            "Keyfob": [
                (0, [0.50, 0.64, 0.78, 0.71]),
                (105, [0.44, 0.53, 0.68, 0.60]),
                (300, [0.51, 0.58, 0.74, 0.65]),
                (495, [0.50, 0.52, 0.60, 0.65]),
                (630, [0.51, 0.59, 0.66, 0.67]),
                (728, [0.51, 0.67, 0.66, 0.75]),
            ],
        },
    },
    "test_10s": {
        "raw_video": "data/raw_videos/test_10s.mp4",
        "task_description": "Operating an industrial sewing machine to guide and stitch white fabric with bimanual coordination.",
        "waypoints": {
            "White Fabric": [
                (0, [0.32, 0.44, 0.68, 0.76]),
                (60, [0.34, 0.40, 0.66, 0.72]),
                (180, [0.33, 0.36, 0.65, 0.66]),
                (240, [0.35, 0.24, 0.64, 0.54]),
                (299, [0.35, 0.26, 0.65, 0.58]),
            ],
            "Sewing Machine Presser Foot": [
                (0, [0.42, 0.18, 0.58, 0.36]),
                (150, [0.42, 0.18, 0.58, 0.36]),
                (299, [0.42, 0.18, 0.58, 0.36]),
            ],
        },
    },
    "my_video": {
        "raw_video": "data/raw_videos/my_video.mp4",
        "task_description": "Guiding and stitching white garment fabric on an industrial sewing machine.",
        "waypoints": {
            "White Garment Fabric": [
                (0, [0.32, 0.44, 0.68, 0.76]),
                (60, [0.34, 0.40, 0.66, 0.72]),
                (180, [0.33, 0.36, 0.65, 0.66]),
                (240, [0.35, 0.24, 0.64, 0.54]),
                (299, [0.35, 0.26, 0.65, 0.58]),
            ],
            "Sewing Machine Head": [
                (0, [0.42, 0.16, 0.58, 0.35]),
                (299, [0.42, 0.16, 0.58, 0.35]),
            ],
        },
    },
    "client_video": {
        "raw_video": "data/raw_videos/client_video.mp4",
        "task_description": "Holding and feeding light blue fabric through an industrial sewing machine.",
        "waypoints": {
            "Light Blue Fabric": [
                (0, [0.48, 0.54, 0.84, 0.86]),
                (60, [0.46, 0.56, 0.82, 0.88]),
                (180, [0.38, 0.66, 0.74, 0.94]),
                (299, [0.42, 0.60, 0.78, 0.90]),
            ],
            "Sewing Machine Foot": [
                (0, [0.44, 0.28, 0.62, 0.48]),
                (299, [0.44, 0.28, 0.62, 0.48]),
            ],
        },
    },
    "new_video": {
        "raw_video": "data/raw_videos/new_video.mp4",
        "task_description": "Bimanual handling and alignment of textile workpiece on the worktable.",
        "waypoints": {
            "Textile Workpiece": [
                (0, [0.26, 0.34, 0.62, 0.64]),
                (60, [0.25, 0.38, 0.58, 0.66]),
                (120, [0.24, 0.28, 0.56, 0.54]),
                (180, [0.26, 0.34, 0.78, 0.60]),
                (240, [0.32, 0.38, 0.66, 0.64]),
                (299, [0.34, 0.30, 0.72, 0.56]),
            ],
        },
    },
}

pipe = EgoAnnotatePipeline(config_path="configs/default.yaml")
r_cfg = RetargetingConfig.from_yaml("configs/retargeting_franka.yaml")
pipe.retargeter = Retargeter(r_cfg)
kin = pipe.retargeter.load_kinematics()
exporter = DatasetExporter(ExporterConfig(output_dir="data/output", lerobot_export_mode="robot", save_viz_video=False))
sig_seg = SignalSegmenter(SignalSegmenterConfig(min_segment_duration_sec=0.25, merge_gap_sec=0.35, idle_threshold_sec=1.0))

for ep_id, spec in EPISODE_SPECS.items():
    ep_dir = Path("data/output") / ep_id
    ann_path = ep_dir / "annotations.json"
    ep = AnnotatedEpisode.load_json(str(ann_path))

    raw_vid = spec["raw_video"]
    ep.video_path = raw_vid
    ep.task_description = spec["task_description"]
    ep.target_robot = "panda"

    track_objects_with_keyframe_waypoints(ep.frames, spec["waypoints"], video_path=raw_vid)
    wl_traj = build_world_locked_arm_trajectory(ep.frames)

    l_ct = [f.left_contact for f in ep.frames]
    r_ct = [f.right_contact for f in ep.frames]
    l_gr = [f.left_grasp for f in ep.frames]
    r_gr = [f.right_grasp for f in ep.frames]
    ts = [f.timestamp for f in ep.frames]
    dwell = {}
    for f in ep.frames:
        for c in (f.left_contact, f.right_contact):
            if c and c.in_contact and c.object_name:
                dwell[c.object_name] = dwell.get(c.object_name, 0) + 1

    primary_obj = list(spec["waypoints"].keys())[0]
    cands = sig_seg.get_candidates(l_ct, r_ct, l_gr, r_gr, ts, dwell=dwell)
    new_segs = []
    for cand in cands:
        hands = cand.hands if hasattr(cand, "hands") and cand.hands else ["right"]
        hand_used = cand.hand_used if hasattr(cand, "hand_used") else ("both" if len(hands) == 2 else hands[0])
        act_name = cand.transition_type if cand.transition_type != "full_video" else "manipulate"
        obj_name = cand.object_name if (act_name != "idle" and cand.object_name and cand.object_name != "unknown") else primary_obj
        desc = build_instruction(act_name, obj_name, hands)
        new_segs.append(ActionSegment(
            name=act_name,
            start_time=cand.start_time,
            end_time=cand.end_time,
            object_name=obj_name,
            hand_used=hand_used,
            hands=hands,
            description=desc,
        ))
    ep.segments = new_segs

    for f in ep.frames:
        assigned = False
        for seg in ep.segments:
            if seg.start_time <= f.timestamp <= seg.end_time:
                f.action_segment = seg.name
                f.frame_description = seg.description
                assigned = True
                break
        if not assigned:
            f.action_segment = "idle"
            f.frame_description = "idle hand posture"

    flora_report = compute_flora_rewards_and_rubric(ep)

    target_poses = PoseMapper(r_cfg.pose_mapper).map_frames(ep.frames)
    with create_ik_solver(r_cfg.ik_backend, kin, r_cfg.end_effector_link, r_cfg.ik_solver, r_cfg.mink_ik) as solver:
        ik_results = solver.solve_sequence(target_poses)
    gripper_commands = GripperMapper(kin, r_cfg.gripper_mapper).map_frames(ep.frames)
    for idx_f, f in enumerate(ep.frames):
        f.robot_joint_angles = ik_results[idx_f].joint_angles.tolist()
        f.robot_gripper_opening_m = float(gripper_commands[idx_f].opening_m)
        f.robot_gripper_method = gripper_commands[idx_f].gripper_mapping_method
        f.robot_reachable = bool(ik_results[idx_f].reachable)

    ep.save_json(str(ann_path))
    exporter._export_json(ep, ep_dir)
    exporter._create_summary(ep, ep_dir)
    sum_p = ep_dir / "summary.json"
    sum_data = json.loads(sum_p.read_text(encoding="utf-8"))
    sum_data["flora_evaluation"] = flora_report
    sum_p.write_text(json.dumps(sum_data, indent=2) + "\n", encoding="utf-8")

    export_to_rlds(ep_id, "data/output")
    export_to_lerobot(ep_id, "data/output", export_mode="robot")
    export_episode_to_mcap(ep, ep_dir / "episode.mcap")
    run_hardware_audit(str(ann_path))

    render_annotated_video(raw_vid, ep, ep_dir / "overlay_annotated.mp4", fps=30.0, atomic=True)
    pipe._generate_proof_video(raw_vid, ep.frames, ik_results, gripper_commands, kin, ep_dir / "side_by_side.mp4")

    rep = validate_episode_dir(ep_dir)
    print(f"[SHIPPED] {ep_id}: frames={len(ep.frames)} | segs={len(ep.segments)} | flora_score={flora_report['overall_rubric_score_5']}/5.0 | passed={rep['passed']}")

if Path("data/output/merged_lerobot_v3").exists():
    shutil.rmtree("data/output/merged_lerobot_v3")
subprocess.run(".venv/bin/python scripts/run.py --merge-lerobot --config configs/default.yaml", shell=True, check=True)

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
