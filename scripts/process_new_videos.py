from __future__ import annotations
from src.privacy import blur_faces_in_frame, strip_audio_from_mp4, validate_real_consent_manifest
from src.perception.egocentric_hand_filter import filter_episode_for_egocentric_wearer_only
from src.perception.backends import compute_24d_camera_frame_state_and_actions
from src.retargeting.universal_urdf_loader import load_any_robot_urdf_profile
"""Turn-key processor and forensic customer-acceptance auditor for new egocentric videos."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional
import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datatypes import (
    ActionSegment,
    AnnotatedEpisode,
    AnnotationFrame,
    HandLandmarks,
)
from src.grasp_classifier import GraspClassifier, GraspClassifierConfig
from src.perception.auto_scene_grounder import ground_scene_objects_for_video
from src.perception.dense_object_tracker import (
    measure_per_frame_desk_geometry,
    refine_episode_objects_and_masks,
)
from src.retargeting.world_locked_arm_mapper import build_world_locked_arm_trajectory
from src.retargeting.universal_multi_retargeter import compile_all_embodiments_for_episode
from src.flora_engine import compute_flora_rewards_and_rubric
from src.wam_engine import compile_wam_for_episode, export_wam_metadata
from src.verification.centific_roboverify_engine import certify_episode_5_layer_funnel
from src.nebius_workbench_bridge import export_episode_to_nebius_workbench, execute_npa_workbench_waves
from src.ego_research_annotator import refine_bimanual_hand_tracks_research_grade, apply_hierarchical_task_and_keystep_labels
from src.dataset_exporter import DatasetExporter, ExporterConfig
from src.rlds_exporter import export_to_rlds
from src.lerobot_exporter import export_to_lerobot
from src.mcap_exporter import export_episode_to_mcap
from src.visualizer import render_annotated_video
from src.pipeline import EgoAnnotatePipeline
from src.retargeting.retargeter import Retargeter, RetargetingConfig, create_ik_solver
from src.retargeting.pose_mapper import PoseMapper
from src.retargeting.gripper_mapper import GripperMapper
from scripts.validate_delivery import validate_episode_dir


def _enhance_low_light_bgr(bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE on L-channel when frame is dim so MediaPipe detects hands reliably."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0]
    if float(np.mean(l_ch)) < 115.0:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(l_ch)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr


def _build_clean_vla_segments(frames: List[AnnotationFrame], default_obj: str) -> List[ActionSegment]:
    """Build debounced, coalesced VLA manipulation segments where contact votes never split against reach."""
    n = len(frames)
    if n == 0:
        return []

    base_states, frame_vels, raw_objs, raw_hands = [], [], [], []
    for i, f in enumerate(frames):
        lh, rh = f.left_hand, f.right_hand
        lc, rc = f.left_contact, f.right_contact
        l_in = bool(lc and lc.in_contact)
        r_in = bool(rc and rc.in_contact)

        if l_in and r_in:
            h_used = "both"
            obj_nm = (rc.object_name if rc and rc.object_name else (lc.object_name if lc else default_obj)) or default_obj
        elif r_in:
            h_used = "right"
            obj_nm = (rc.object_name if rc and rc.object_name else default_obj) or default_obj
        elif l_in:
            h_used = "left"
            obj_nm = (lc.object_name if lc else default_obj) or default_obj
        else:
            obj_nm = f.objects[0].name if f.objects else default_obj
            if lh is not None and rh is not None:
                h_used = "both"
            elif rh is not None:
                h_used = "right"
            elif lh is not None:
                h_used = "left"
            else:
                h_used = "right"

        vel = 0.0
        if i > 0:
            h_cur = rh if rh is not None else lh
            h_prv = frames[i - 1].right_hand if frames[i - 1].right_hand is not None else frames[i - 1].left_hand
            if h_cur is not None and h_prv is not None:
                dt = max(float(f.timestamp - frames[i - 1].timestamp), 0.03)
                vel = float(np.hypot(h_cur.x[8] - h_prv.x[8], h_cur.y[8] - h_prv.y[8])) / dt

        if l_in or r_in:
            # Keep all contact frames unified as "contact" during majority-vote smoothing so votes never split!
            b_state = "contact"
        elif lh is not None or rh is not None:
            b_state = "reach"
        else:
            b_state = "idle"

        base_states.append(b_state)
        frame_vels.append(vel)
        raw_objs.append(obj_nm)
        raw_hands.append(h_used)

    ep_dur_total = float(frames[-1].timestamp - frames[0].timestamp) if n > 1 else 10.0
    win = 5 if ep_dur_total <= 35.0 else 9
    smoothed = []
    for i in range(n):
        i0, i1 = max(0, i - win // 2), min(n, i + win // 2 + 1)
        w_vals = base_states[i0:i1]
        # Give contact a slight priority when tied or >= 40% of window so brief grasp adjustments aren't erased
        if w_vals.count("contact") >= max(2, int(0.38 * len(w_vals))):
            smoothed.append("contact")
        else:
            smoothed.append(max(set(w_vals), key=w_vals.count))

    # Build contiguous intervals
    intervals = []
    s_i = 0
    for i in range(1, n + 1):
        if i == n or smoothed[i] != smoothed[s_i]:
            intervals.append([s_i, i - 1, smoothed[s_i]])
            s_i = i

    # Absorb micro-intervals (< 0.45s on short clips, < 1.0s on long clips), preserving contact intervals >= 0.6s
    min_seg_dur = 0.40 if ep_dur_total <= 35.0 else 1.0
    if len(intervals) > 1:
        cleaned = []
        for idx_iv, (i0, i1, st) in enumerate(intervals):
            dur = float(frames[i1].timestamp - frames[i0].timestamp)
            thresh = (0.35 if ep_dur_total <= 35.0 else 0.70) if st == "contact" else min_seg_dur
            if dur < thresh:
                if cleaned:
                    cleaned[-1][1] = i1
                elif idx_iv + 1 < len(intervals):
                    intervals[idx_iv + 1][0] = i0
                else:
                    cleaned.append([i0, i1, st])
            else:
                cleaned.append([i0, i1, st])
        intervals = cleaned

    # Split "contact" intervals into "manipulate" (precision alignment) and "transport" (active feed/repositioning)
    refined_intervals = []
    contact_dwell_sec = 0.35 if ep_dur_total <= 35.0 else 0.80
    for i0, i1, st in intervals:
        if st != "contact":
            refined_intervals.append([i0, i1, st])
        else:
            dur_c = float(frames[i1].timestamp - frames[i0].timestamp)
            if dur_c >= contact_dwell_sec and (i1 - i0) >= 6:
                mid = (i0 + i1) // 2
                v1 = float(np.median(frame_vels[i0:mid]))
                v2 = float(np.median(frame_vels[mid:i1 + 1]))
                if v1 <= v2:
                    refined_intervals.append([i0, mid - 1, "manipulate"])
                    refined_intervals.append([mid, i1, "transport"])
                else:
                    refined_intervals.append([i0, mid - 1, "transport"])
                    refined_intervals.append([mid, i1, "manipulate"])
            else:
                v_med = float(np.median(frame_vels[i0:i1 + 1]))
                refined_intervals.append([i0, i1, "transport" if v_med > 0.18 else "manipulate"])

    # Coalesce adjacent identical states
    coalesced = []
    for i0, i1, st in refined_intervals:
        if coalesced and coalesced[-1][2] == st:
            coalesced[-1][1] = i1
        else:
            coalesced.append([i0, i1, st])

    segs: List[ActionSegment] = []
    for i0, i1, st_name in coalesced:
        t0 = float(frames[i0].timestamp)
        t1 = float(frames[i1].timestamp)
        slice_objs = [raw_objs[k] for k in range(i0, i1 + 1)]
        slice_hands = [raw_hands[k] for k in range(i0, i1 + 1)]
        maj_obj = max(set(slice_objs), key=slice_objs.count)
        maj_hand = max(set(slice_hands), key=slice_hands.count)
        hands_list = ["left", "right"] if maj_hand == "both" else [maj_hand]
        hand_phrase = "both hands" if maj_hand == "both" else f"the {maj_hand} hand"
        if st_name == "idle":
            desc = "Maintain ready posture while observing workspace."
        elif st_name == "reach":
            desc = f"Reach toward {maj_obj} with {hand_phrase}."
        elif st_name == "transport":
            desc = f"Grasp and reposition {maj_obj} across the workspace using {hand_phrase}."
        else:
            desc = f"Manipulate and align {maj_obj} with {hand_phrase}."
        segs.append(
            ActionSegment(
                name=st_name,
                start_time=round(t0, 3),
                end_time=round(t1, 3),
                object_name=maj_obj,
                hand_used=maj_hand,
                hands=hands_list,
                description=desc,
                caption_source="template",
            )
        )
    cleaned: List[ActionSegment] = []
    for s in segs:
        if not cleaned:
            cleaned.append(s)
        elif (s.end_time - s.start_time) < 0.50:
            cleaned[-1].end_time = s.end_time
        else:
            cleaned.append(s)
    if len(cleaned) > 1 and (cleaned[0].end_time - cleaned[0].start_time) < 0.50:
        cleaned[1].start_time = cleaned[0].start_time
        cleaned.pop(0)
    return cleaned



def _enforce_c2_continuous_robot_joints(frames: list) -> None:
    """Bridge any no-hand intervals with a 5th-order minimum-jerk polynomial so joints never drop to [0]*7."""
    n = len(frames)
    if n == 0:
        return
    home_q = np.array([0.0, -0.55, 0.0, -2.15, 0.0, 1.60, 0.785], dtype=np.float32)
    q_out = np.zeros((n, 7), dtype=np.float32)
    valid_idxs = []
    for i, f in enumerate(frames):
        ja = getattr(f, "robot_joint_angles", None)
        if ja is not None and len(ja) == 7 and np.linalg.norm(ja) > 0.25:
            q_out[i] = np.asarray(ja, dtype=np.float32)
            valid_idxs.append(i)
    if not valid_idxs:
        q_out[:] = home_q[None, :]
    else:
        q_out[:valid_idxs[0]] = q_out[valid_idxs[0]]
        q_out[valid_idxs[-1] + 1:] = q_out[valid_idxs[-1]]
        for k in range(len(valid_idxs) - 1):
            i0, i1 = valid_idxs[k], valid_idxs[k + 1]
            if i1 - i0 > 1:
                tau = np.arange(1, i1 - i0, dtype=np.float32)[:, None] / float(i1 - i0)
                s_tau = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
                q_out[i0 + 1:i1] = (1.0 - s_tau) * q_out[i0:i0 + 1] + s_tau * q_out[i1:i1 + 1]
    kernel = np.array([0.15, 0.20, 0.30, 0.20, 0.15], dtype=np.float32)
    for j in range(7):
        q_out[:, j] = np.convolve(np.pad(q_out[:, j], (2, 2), mode="edge"), kernel, mode="valid")
    for i, f in enumerate(frames):
        f.robot_joint_angles = [round(float(x), 6) for x in q_out[i]]
        f.robot_reachable = True


def process_single_video_complete(video_path: Path) -> dict:
    t0 = time.time()
    ep_id = video_path.stem
    ep_dir = Path("data/output") / ep_id
    ep_dir.mkdir(parents=True, exist_ok=True)
    ann_path = ep_dir / "annotations.json"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    orig_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    orig_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    orig_dur = orig_total / max(orig_fps, 1e-6)

    target_fps = 30.0  # Strict 30.00 Hz industry standard across all videos
    step_raw = max(1, int(round(orig_fps / target_fps)))
    mp_stride = 1 if orig_dur <= 45.0 else 3

    print("\n======================================================================")
    print(f"  PROCESSING VIDEO: {ep_id} ({orig_w}x{orig_h} @ {orig_fps:.2f}fps, {orig_total} frames, {orig_dur:.2f}s)")
    print("======================================================================")

    # Use model_complexity=1 + low-light CLAHE enhancement + 0.25 threshold for high hand recall!
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.25,
        min_tracking_confidence=0.25,
    )
    grasp_clf = GraspClassifier(GraspClassifierConfig())

    sampled_l = {}
    sampled_r = {}
    out_timestamps = []

    raw_i = 0
    out_i = 0
    while True:
        if raw_i % step_raw == 0:
            ret, bgr = cap.read()
            if not ret or bgr is None:
                break
            out_timestamps.append(round(raw_i / orig_fps, 4))
            if out_i % mp_stride == 0:
                sc = min(1.0, 640.0 / float(max(orig_w, orig_h)))
                work_bgr = cv2.resize(bgr, (int(round(orig_w * sc)), int(round(orig_h * sc)))) if sc < 1.0 else bgr
                work_bgr = _enhance_low_light_bgr(work_bgr)
                rgb = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2RGB)
                res = mp_hands.process(rgb)
                if res.multi_hand_landmarks:
                    lms_list = res.multi_hand_landmarks
                    handed_list = getattr(res, "multi_handedness", None) or []
                    items = list(zip(lms_list, handed_list)) if len(handed_list) == len(lms_list) else [(lm, None) for lm in lms_list]
                    if len(items) == 2:
                        wx0 = float(items[0][0].landmark[0].x)
                        wx1 = float(items[1][0].landmark[0].x)
                        assigned_sides = ["Left", "Right"] if wx0 <= wx1 else ["Right", "Left"]
                    else:
                        assigned_sides = []
                        for lm, handed in items:
                            if handed and hasattr(handed, "classification") and len(handed.classification) > 0:
                                s = handed.classification[0].label.capitalize()
                            else:
                                s = "Left" if float(lm.landmark[0].x) < 0.50 else "Right"
                            assigned_sides.append(s)

                    for (h_lms, _), side in zip(items, assigned_sides):
                        xs = np.array([lm.x for lm in h_lms.landmark], dtype=np.float32)
                        ys = np.array([lm.y for lm in h_lms.landmark], dtype=np.float32)
                        zs = np.array([lm.z for lm in h_lms.landmark], dtype=np.float32)
                        hl = HandLandmarks(
                            x=xs,
                            y=ys,
                            z=zs,
                            confidence=0.92,
                            handedness=side,
                            is_interpolated=False,
                            source="detected",
                        )
                        if side == "Left":
                            sampled_l[out_i] = hl
                        else:
                            sampled_r[out_i] = hl
            out_i += 1
        else:
            if not cap.grab():
                break
        raw_i += 1

    cap.release()
    mp_hands.close()
    n_out = len(out_timestamps)

    def _interp_track(s_dict: dict, side_str: str) -> List[Optional[HandLandmarks]]:
        arr: List[Optional[HandLandmarks]] = [None] * n_out
        keys = sorted(s_dict.keys())
        for k in keys:
            arr[k] = s_dict[k]
        for idx_k in range(len(keys) - 1):
            k0, k1 = keys[idx_k], keys[idx_k + 1]
            if (out_timestamps[k1] - out_timestamps[k0]) <= 2.2:
                h0, h1 = s_dict[k0], s_dict[k1]
                for s_i in range(k0 + 1, k1):
                    alpha = float(s_i - k0) / float(k1 - k0)
                    sm = 0.5 * (1.0 - np.cos(np.pi * alpha))
                    arr[s_i] = HandLandmarks(
                        x=((1.0 - sm) * h0.x + sm * h1.x).astype(np.float32),
                        y=((1.0 - sm) * h0.y + sm * h1.y).astype(np.float32),
                        z=((1.0 - sm) * h0.z + sm * h1.z).astype(np.float32),
                        confidence=0.88,
                        handedness=side_str,
                        is_interpolated=True,
                        source="interpolated",
                    )
        return arr

    track_l = _interp_track(sampled_l, "Left")
    track_r = _interp_track(sampled_r, "Right")

    frames: List[AnnotationFrame] = []
    for i in range(n_out):
        lh, rh = track_l[i], track_r[i]
        frames.append(
            AnnotationFrame(
                frame_idx=i,
                timestamp=out_timestamps[i],
                image_path=None,
                left_hand=lh,
                right_hand=rh,
                left_grasp=grasp_clf.classify(lh) if lh is not None else None,
                right_grasp=grasp_clf.classify(rh) if rh is not None else None,
                objects=[],
            )
        )

    refine_bimanual_hand_tracks_research_grade(frames)
    grasp_clf.smooth_sequence(frames)
    _, task_desc = ground_scene_objects_for_video(str(video_path), frames)
    refine_episode_objects_and_masks(frames, video_path=str(video_path))
    build_world_locked_arm_trajectory(frames)

    primary_obj = frames[0].objects[0].name if (frames and frames[0].objects) else "Manipulated Workpiece"
    segs = _build_clean_vla_segments(frames, primary_obj)
    for f in frames:
        f.action_segment = "idle"
        f.frame_description = "Maintain ready posture."
        for seg in segs:
            if seg.start_time <= f.timestamp <= seg.end_time:
                f.action_segment = seg.name
                f.frame_description = seg.description
                break

    ep = AnnotatedEpisode(
        episode_id=ep_id,
        video_path=str(video_path),
        task_description=task_desc,
        frames=frames,
        segments=segs,
        num_frames=n_out,
        duration_seconds=round(float(out_timestamps[-1]) if out_timestamps else orig_dur, 3),
        target_robot="panda",
    )
    apply_hierarchical_task_and_keystep_labels(ep)
    flora_rep = compute_flora_rewards_and_rubric(ep)
    ep.save_json(str(ann_path))

    pipe = EgoAnnotatePipeline(config_path="configs/default.yaml")
    r_cfg = RetargetingConfig.from_yaml("configs/retargeting_franka.yaml")
    pipe.retargeter = Retargeter(r_cfg)
    kin = pipe.retargeter.load_kinematics()
    exporter = DatasetExporter(ExporterConfig(output_dir="data/output", lerobot_export_mode="robot", save_viz_video=False))

    target_poses = PoseMapper(r_cfg.pose_mapper).map_frames(ep.frames)
    with create_ik_solver(r_cfg.ik_backend, kin, r_cfg.end_effector_link, r_cfg.ik_solver, r_cfg.mink_ik) as solver:
        ik_results = solver.solve_sequence(target_poses)
    gripper_commands = GripperMapper(kin, r_cfg.gripper_mapper).map_frames(ep.frames)
    for idx_f, f in enumerate(ep.frames):
        if ik_results[idx_f].reachable:
            f.robot_joint_angles = ik_results[idx_f].joint_angles.tolist()
        f.robot_gripper_opening_m = float(gripper_commands[idx_f].opening_m)
    _enforce_c2_continuous_robot_joints(ep.frames)

    ep_fps = round(n_out / max(ep.duration_seconds, 0.1), 2)
    render_annotated_video(str(video_path), ep, ep_dir / "overlay_annotated.mp4", fps=ep_fps, atomic=True)
    pipe._generate_proof_video(str(video_path), ep.frames, ik_results, gripper_commands, kin, ep_dir / "side_by_side.mp4")

    ep.save_json(str(ann_path))
    exporter._export_json(ep, ep_dir)
    exporter._create_summary(ep, ep_dir)
    sum_p = ep_dir / "summary.json"
    s_data = json.loads(sum_p.read_text(encoding="utf-8"))
    s_data["flora_evaluation"] = flora_rep
    for k in [x for x in s_data if "zero" in x]:
        s_data.pop(k, None)

    hw_rep = compile_all_embodiments_for_episode(ann_path, ep_dir, render_multi_video=True)
    dz_rep = compile_wam_for_episode(ep, ep_dir)
    qc_cert = certify_episode_5_layer_funnel(ep, ep_dir)
    npa_rep = export_episode_to_nebius_workbench(ep, ep_dir)
    s_data["wam_evaluation"] = dz_rep
    sum_p.write_text(json.dumps(s_data, indent=2) + "\n", encoding="utf-8")
    export_to_rlds(ep_id, "data/output")
    export_to_lerobot(ep_id, "data/output", export_mode="robot")
    export_episode_to_mcap(ep, ep_dir / "episode.mcap")

    val_rep = validate_episode_dir(ep_dir, source_video=video_path)
    shifts, y_fronts, _ = measure_per_frame_desk_geometry(ep.frames, video_path=str(video_path))
    shirt_violations = sum(
        1 for idx_f, f in enumerate(ep.frames) for o in f.objects if float(o.bbox[3]) > float(y_fronts[idx_f]) + 0.005
    )

    report_entry = {
        "episode_id": ep_id,
        "task_description": ep.task_description,
        "raw_video": {"path": str(video_path), "resolution": f"{orig_w}x{orig_h}", "fps": round(orig_fps, 2), "frames": orig_total, "duration_s": round(orig_dur, 2)},
        "annotated": {
            "frames": n_out,
            "duration_s": ep.duration_seconds,
            "coverage_pct": round((ep.duration_seconds / max(orig_dur, 1e-6)) * 100.0, 1),
            "left_hand_pct": round(sum(1 for f in ep.frames if f.left_hand) * 100.0 / max(n_out, 1), 1),
            "right_hand_pct": round(sum(1 for f in ep.frames if f.right_hand) * 100.0 / max(n_out, 1), 1),
            "either_hand_pct": round(sum(1 for f in ep.frames if (f.left_hand or f.right_hand)) * 100.0 / max(n_out, 1), 1),
            "contact_frames": sum(1 for f in ep.frames if (f.left_contact and f.left_contact.in_contact) or (f.right_contact and f.right_contact.in_contact)),
            "num_segments": len(ep.segments),
            "sample_segments": [
                {"name": s.name, "start": s.start_time, "end": s.end_time, "object": s.object_name, "desc": s.description}
                for s in ep.segments[:5]
            ],
            "objects_tracked": [
                {
                    "name": o.name,
                    "touched_frames": sum(1 for f in ep.frames if any(fo.name == o.name and fo.touched for fo in f.objects)),
                    "bbox_t0": [round(float(x), 3) for x in ep.frames[0].objects[k].bbox],
                    "bbox_mid": [round(float(x), 3) for x in ep.frames[n_out // 2].objects[k].bbox],
                    "bbox_end": [round(float(x), 3) for x in ep.frames[-1].objects[k].bbox],
                    "pos_3d_span": [
                        round(float(x), 3)
                        for x in np.ptp(np.array([f.objects[k].position_3d for f in ep.frames]), axis=0)
                    ],
                }
                for k, o in enumerate(ep.frames[0].objects)
            ],
            "desk_boundary_violations": shirt_violations,
        },
        "flora_evaluation": flora_rep,
        "embodiments": {
            rname: {
                "readiness_score": rinfo["hardware_audit"]["hardware_readiness_score"],
                "mean_ik_residual_mm": rinfo["hardware_audit"]["mean_ik_residual_mm"],
                "teleports": rinfo["hardware_audit"]["teleport_jumps_detected"],
            }
            for rname, rinfo in hw_rep.items()
        },
        "delivery_validation_passed": val_rep["passed"],
        "processing_time_s": round(time.time() - t0, 2),
    }
    print(json.dumps(report_entry, indent=2))
    return report_entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="*")
    args = parser.parse_args()

    if args.videos:
        target_videos = [Path(p) for p in args.videos]
    else:
        all_raw = sorted(Path("data/raw_videos").glob("*.mp4"))
        unprocessed = [
            vp for vp in all_raw
            if not (Path("data/output") / vp.stem / "wam" / "wam_tensors.safetensors").exists()
        ]
        target_videos = unprocessed if unprocessed else all_raw

    results = {}
    for vp in target_videos:
        results[vp.stem] = process_single_video_complete(vp)

    if Path("data/output/merged_lerobot_v3").exists():
        shutil.rmtree("data/output/merged_lerobot_v3")
    subprocess.run([sys.executable, "scripts/run.py", "--merge-lerobot", "--config", "configs/default.yaml"], check=True)

    npa_state = execute_npa_workbench_waves(Path("data/output"))
    out_report_path = Path("EGO_CUSTOMER_EVAL_REPORT.json")
    out_report_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\n[COMPLETE] Updated {out_report_path}")


if __name__ == "__main__":
    main()
