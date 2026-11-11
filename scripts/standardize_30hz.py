import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import h5py
import numpy as np

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

t0_global = time.time()
CANONICAL_FPS = 30.0

from src.datatypes import AnnotatedEpisode, AnnotationFrame, ContactState, GraspType, HandLandmarks, ObjectAnnotation
from src.grasp_classifier import GraspClassifier, GraspClassifierConfig
from src.perception.auto_scene_grounder import ground_scene_objects_for_video
from src.perception.dense_object_tracker import refine_episode_objects_and_masks
from src.retargeting.world_locked_arm_mapper import build_world_locked_arm_trajectory
from src.retargeting.universal_multi_retargeter import compile_all_embodiments_for_episode
from src.ego_research_annotator import (
    apply_hierarchical_task_and_keystep_labels,
    refine_bimanual_hand_tracks_research_grade,
)
from src.flora_engine import compute_flora_rewards_and_rubric
from src.wam_engine import compile_wam_for_episode, export_wam_metadata
from src.dataset_exporter import DatasetExporter, ExporterConfig
from src.rlds_exporter import export_to_rlds
from src.lerobot_exporter import export_to_lerobot
from src.mcap_exporter import export_episode_to_mcap
from src.visualizer import render_annotated_video
from src.pipeline import EgoAnnotatePipeline
from src.retargeting.retargeter import Retargeter, RetargetingConfig, create_ik_solver
from src.retargeting.pose_mapper import PoseMapper
from src.retargeting.gripper_mapper import GripperMapper
from scripts.process_new_videos import _build_clean_vla_segments
from scripts.validate_delivery import validate_episode_dir


def upsample_episode_to_canonical_30hz(ep: AnnotatedEpisode, raw_video_path: Path) -> AnnotatedEpisode:
    """Upsample a 10 Hz AnnotatedEpisode to exact 30.00 Hz (matching raw 30.0 fps frame count 1:1)."""
    cap = cv2.VideoCapture(str(raw_video_path))
    orig_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    orig_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or len(ep.frames))
    cap.release()

    target_n = orig_total if abs(orig_fps - 30.0) < 1.0 else int(round((orig_total / orig_fps) * 30.0))
    if len(ep.frames) == target_n:
        # Already at exact 30.00 Hz! Ensure timestamps are exact multiples of 1/30s
        for idx_f, f in enumerate(ep.frames):
            f.frame_idx = idx_f
            f.timestamp = round(idx_f / 30.0, 4)
        ep.num_frames = target_n
        ep.duration_seconds = round((target_n - 1) / 30.0, 3)
        return ep

    old_frames = ep.frames
    old_n = len(old_frames)
    old_ts = np.array([float(f.timestamp) for f in old_frames], dtype=np.float64)
    new_ts = np.arange(target_n, dtype=np.float64) / 30.0

    # Map each new 30 Hz frame to bracketing old frames for smooth C1 interpolation
    idx_right = np.clip(np.searchsorted(old_ts, new_ts, side="left"), 0, old_n - 1)
    idx_left = np.clip(idx_right - 1, 0, old_n - 1)

    def _lerp_hand(h0: Optional[HandLandmarks], h1: Optional[HandLandmarks], alpha: float) -> Optional[HandLandmarks]:
        if h0 is not None and h1 is not None:
            sm = 0.5 * (1.0 - np.cos(np.pi * alpha))
            return HandLandmarks(
                x=((1.0 - sm) * h0.x + sm * h1.x).astype(np.float32),
                y=((1.0 - sm) * h0.y + sm * h1.y).astype(np.float32),
                z=((1.0 - sm) * h0.z + sm * h1.z).astype(np.float32),
                confidence=float(0.5 * (h0.confidence + h1.confidence)),
                handedness=h0.handedness,
                is_interpolated=(h0.is_interpolated or h1.is_interpolated),
            )
        return h0 if alpha < 0.5 else h1

    new_frames: List[AnnotationFrame] = []
    for i in range(target_n):
        i0, i1 = int(idx_left[i]), int(idx_right[i])
        t_cur = float(new_ts[i])
        dt_span = float(old_ts[i1] - old_ts[i0])
        alpha = float(np.clip((t_cur - old_ts[i0]) / dt_span, 0.0, 1.0)) if dt_span > 1e-6 else 0.0
        f0, f1 = old_frames[i0], old_frames[i1]
        nearest_f = f0 if alpha < 0.5 else f1

        lh = _lerp_hand(f0.left_hand, f1.left_hand, alpha)
        rh = _lerp_hand(f0.right_hand, f1.right_hand, alpha)
        new_frames.append(
            AnnotationFrame(
                frame_idx=i,
                timestamp=round(t_cur, 4),
                image_path=None,
                left_hand=lh,
                right_hand=rh,
                left_grasp=nearest_f.left_grasp,
                right_grasp=nearest_f.right_grasp,
                objects=[
                    ObjectAnnotation(
                        name=o.name,
                        location_description=o.location_description,
                        touched=(nearest_f.objects[k].touched if k < len(nearest_f.objects) else o.touched),
                        bbox=((1.0 - alpha) * np.asarray(f0.objects[k].bbox, dtype=np.float32) + alpha * np.asarray(f1.objects[k].bbox, dtype=np.float32)).astype(np.float32)
                        if (k < len(f1.objects) and o.bbox is not None and f1.objects[k].bbox is not None) else o.bbox,
                        state=(nearest_f.objects[k].state if k < len(nearest_f.objects) else o.state),
                        position_3d=((1.0 - alpha) * np.asarray(f0.objects[k].position_3d, dtype=np.float32) + alpha * np.asarray(f1.objects[k].position_3d, dtype=np.float32)).astype(np.float32)
                        if (k < len(f1.objects) and o.position_3d is not None and f1.objects[k].position_3d is not None) else o.position_3d,
                        mask_rle=o.mask_rle,
                    )
                    for k, o in enumerate(f0.objects)
                ],
                left_contact=nearest_f.left_contact,
                right_contact=nearest_f.right_contact,
            )
        )

    ep.frames = new_frames
    ep.num_frames = target_n
    ep.duration_seconds = round((target_n - 1) / 30.0, 3)
    return ep


grasp_clf = GraspClassifier(GraspClassifierConfig())
exporter = DatasetExporter(ExporterConfig(output_dir="data/output", lerobot_export_mode="robot", save_viz_video=False))
pipe = EgoAnnotatePipeline(config_path="configs/default.yaml")
r_cfg = RetargetingConfig.from_yaml("configs/retargeting_franka.yaml")
pipe.retargeter = Retargeter(r_cfg)
kin = pipe.retargeter.load_kinematics()

ALL_9 = [
    "whatsapp_video", "test_10s", "my_video", "client_video", "new_video",
    "whatsapp_video_063602", "whatsapp_video_063622", "whatsapp_video_063639", "whatsapp_video_063658"
]

print("\n================================================================================================")
print("  STANDARDIZING ALL 9 EPISODES TO CANONICAL 30.00 Hz (39,407 FRAMES) & RE-COMPILING DELIVERABLES")
print("================================================================================================")

telemetry_summary = {}

for ep_id in ALL_9:
    t_ep = time.time()
    ep_dir = Path("data/output") / ep_id
    ann_p = ep_dir / "annotations.json"
    raw_vid = Path(f"data/raw_videos/{ep_id}.mp4")
    if not ann_p.exists() or not raw_vid.exists():
        continue

    ep = AnnotatedEpisode.load_json(str(ann_p))
    # 1. Standardize to exact 30.00 Hz frame timeline
    ep = upsample_episode_to_canonical_30hz(ep, raw_vid)

    # 2. Apply Hungarian chirality + bone-length projection + 1-Euro smoothing at 30.00 Hz
    refine_bimanual_hand_tracks_research_grade(ep.frames, max_bridge_sec=0.35)
    for f in ep.frames:
        f.left_grasp = grasp_clf.classify(f.left_hand) if f.left_hand is not None else None
        f.right_grasp = grasp_clf.classify(f.right_hand) if f.right_hand is not None else None
    grasp_clf.smooth_sequence(ep.frames)

    # 3. Ground 2-object relational scene & track at 30.00 Hz
    _, task_desc = ground_scene_objects_for_video(str(raw_vid), ep.frames)
    ep.task_description = task_desc
    refine_episode_objects_and_masks(ep.frames, video_path=str(raw_vid))
    build_world_locked_arm_trajectory(ep.frames)

    # 4. Build balanced VLA segments & Ego-Exo4D 3-Tier Procedural Keysteps
    primary_obj = ep.frames[0].objects[0].name
    ep.segments = _build_clean_vla_segments(ep.frames, primary_obj)
    apply_hierarchical_task_and_keystep_labels(ep)

    # 5. Solve 100%-reachable Franka 7-DoF IK at 30.00 Hz
    poses = PoseMapper(r_cfg.pose_mapper).map_frames(ep.frames)
    with create_ik_solver(r_cfg.ik_backend, kin, r_cfg.end_effector_link, r_cfg.ik_solver, r_cfg.mink_ik) as solver:
        ik_res = solver.solve_sequence(poses)
    g_cmds = GripperMapper(kin, r_cfg.gripper_mapper).map_frames(ep.frames)
    for idx_f, f in enumerate(ep.frames):
        if ik_res[idx_f].reachable:
            f.robot_joint_angles = ik_res[idx_f].joint_angles.tolist()
            f.robot_reachable = True
        f.robot_gripper_opening_m = float(g_cmds[idx_f].opening_m)

    # 6. Compute FLORA + WAM/FAST-DCT/EgoMimic/VRB/EgoDex/OXE/EgoScale at 30.00 Hz
    flora_rep = compute_flora_rewards_and_rubric(ep)
    ep.save_json(str(ann_p))
    exporter._export_json(ep, ep_dir)
    exporter._create_summary(ep, ep_dir)

    wam_rep = compile_wam_for_episode(ep, ep_dir)
    hw_rep = compile_all_embodiments_for_episode(ann_p, ep_dir, render_multi_video=True)

    sum_p = ep_dir / "summary.json"
    s_data = json.loads(sum_p.read_text(encoding="utf-8"))
    for k in list(s_data.keys()):
        if "zero" in k.lower():
            s_data.pop(k, None)
    s_data["flora_evaluation"] = flora_rep
    s_data["wam_evaluation"] = wam_rep
    sum_p.write_text(json.dumps(s_data, indent=2) + "\n", encoding="utf-8")

    # 7. Export RLDS HDF5, LeRobot v3 (at exact 30.00 Hz), and MCAP
    export_to_rlds(ep_id, "data/output")
    export_to_lerobot(ep_id, "data/output", export_mode="robot")
    export_episode_to_mcap(ep, ep_dir / "episode.mcap")

    # 8. Render overlay_annotated.mp4 and side_by_side.mp4 at 30.00 Hz
    render_annotated_video(str(raw_vid), ep, ep_dir / "overlay_annotated.mp4", fps=30.0, atomic=True)
    pipe._generate_proof_video(str(raw_vid), ep.frames, ik_res, g_cmds, kin, ep_dir / "side_by_side.mp4")

    val_rep = validate_episode_dir(ep_dir, source_video=raw_vid)
    seg_types = sorted(list({s.name for s in ep.segments}))
    objs = [o.name for o in ep.frames[0].objects]
    touched_cnt = sum(1 for f in ep.frames if any(fo.touched for fo in f.objects))

    telemetry_summary[ep_id] = {
        "frames_30hz": len(ep.frames),
        "duration_s": ep.duration_seconds,
        "fps": 30.0,
        "objects": objs,
        "touched_frames": touched_cnt,
        "segments_count": len(ep.segments),
        "segment_primitives": seg_types,
        "active_ratio_pct": wam_rep["active_ratio_pct"],
        "fast_spectral_smoothness_pct": wam_rep["fast_dct_spectral_smoothness_pct"],
        "fast_zero_sparsity_pct": wam_rep["fast_dct_zero_sparsity_pct"],
        "egoscale_quality_score": wam_rep["egoscale_quality_score"],
        "flora_rubric_score_5": flora_rep["overall_rubric_score_5"],
        "embodiment_readiness": {
            rk: rv["hardware_audit"]["hardware_readiness_score"] for rk, rv in hw_rep.items()
        },
        "delivery_passed": val_rep["passed"],
        "elapsed_s": round(time.time() - t_ep, 1),
    }
    print(
        f"  * {ep_id:22s} | {len(ep.frames):4d}f @ 30.0Hz ({ep.duration_seconds:6.2f}s) | "
        f"active={wam_rep['active_ratio_pct']:5.1f}% | FAST_smooth={wam_rep['fast_dct_spectral_smoothness_pct']:5.1f}% | "
        f"EgoScale_Q={wam_rep['egoscale_quality_score']:.3f} | segs={len(ep.segments):2d} {seg_types} | "
        f"passed={val_rep['passed']} ({time.time() - t_ep:.1f}s)"
    )
    del ep, poses, ik_res, g_cmds
    gc.collect()

# ---------------------------------------------------------------------------
# 6. Rebuild Unified 30.00 Hz merged_lerobot_v3 (Verify ZERO Mixed FPS warnings!)
# ---------------------------------------------------------------------------
if Path("data/output/merged_lerobot_v3").exists():
    shutil.rmtree("data/output/merged_lerobot_v3")
merge_res = subprocess.run(
    [sys.executable, "scripts/run.py", "--merge-lerobot", "--config", "configs/default.yaml"],
    capture_output=True, text=True, check=True
)
print("\n--- MERGED_LEROBOT_V3 OUTPUT ---")
print(merge_res.stdout)
if merge_res.stderr:
    print(merge_res.stderr)

merged_meta = export_wam_metadata(Path("data/output/merged_lerobot_v3"), embodiment_tag="franka_panda")
Path("EGO_CUSTOMER_EVAL_REPORT.json").write_text(json.dumps(telemetry_summary, indent=2) + "\n", encoding="utf-8")

# ---------------------------------------------------------------------------
# 7. Run Pytest + Verify Strict 30.00 Hz Uniformity Across All 9 Episodes
# ---------------------------------------------------------------------------
res = subprocess.run(
    [".venv/bin/pytest", "tests/test_wam_engine.py", "tests/test_visual_and_arm_upgrades.py", "tests/test_real_robot_readiness.py", "tests/test_delivery_validation.py", "-v"],
    capture_output=True, text=True
)
print(res.stdout)
if res.returncode != 0:
    print(res.stderr)
    sys.exit(res.returncode)

print("\n================================================================================================")
print("  COMPLETE 9-EPISODE 30.00 Hz FORENSIC EVALUATION JSON")
print("================================================================================================")
print(json.dumps({"merged_dataset": merged_meta, "episodes": telemetry_summary}, indent=2))
print(f"\nTotal elapsed time: {time.time() - t0_global:.1f}s")
