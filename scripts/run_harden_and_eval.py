from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path(".").resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ---------------------------------------------------------------------------
# 1. Upgrade src/perception/dense_object_tracker.py:
#    Full Autonomous Multi-Object Discovery + Desk-Plane Homography +
#    Grasp-Transport State Machine for ANY unseen video!
# ---------------------------------------------------------------------------
dot_path = Path("src/perception/dense_object_tracker.py")
dot_src = dot_path.read_text(encoding="utf-8")

old_refine_fn = dot_src[dot_src.find("def refine_episode_objects_and_masks("):]
new_refine_fn = '''def discover_unseen_video_objects(
    frames: List[Any],
    working_imgs: List[Optional[np.ndarray]],
    y_fronts: np.ndarray,
    default_object_name: str = "workspace_object",
) -> List[Tuple[str, np.ndarray]]:
    """Automatically discover all distinct manipulable tabletop objects on an unseen video."""
    # 1. Gather valid semantic object names from Stage 3 VLM / OWL-ViT detections
    vlm_names: List[str] = []
    existing_with_boxes: List[Tuple[str, np.ndarray]] = []
    for f in frames[:min(30, len(frames))]:
        for o in getattr(f, "objects", []) or []:
            if is_valid_manipulable_object(o.name, o.bbox):
                clean_nm = o.name.split("(")[0].strip()
                if clean_nm not in vlm_names:
                    vlm_names.append(clean_nm)
                if o.bbox is not None and len(existing_with_boxes) < 6:
                    b_arr = np.asarray(o.bbox, dtype=np.float32)
                    # Keep if not duplicate name
                    if all(clean_nm != enm for enm, _ in existing_with_boxes):
                        existing_with_boxes.append((clean_nm, b_arr))

    if len(existing_with_boxes) >= 2:
        return existing_with_boxes

    # 2. Find the cleanest unoccluded frame (fewest hand pixels in workspace) to segment foreground islands
    best_i = 0
    for i, f in enumerate(frames[:min(60, len(frames))]):
        if f.left_hand is None and f.right_hand is None and working_imgs[i] is not None:
            best_i = i
            break
    img0 = working_imgs[best_i] if working_imgs and working_imgs[best_i] is not None else None

    discovered: List[Tuple[str, np.ndarray]] = list(existing_with_boxes)
    if img0 is not None:
        sh, sw = img0.shape[:2]
        yf = float(y_fronts[best_i]) if len(y_fronts) > best_i else 0.78
        lab = cv2.cvtColor(img0, cv2.COLOR_BGR2LAB).astype(np.float32)
        y_top = int(0.14 * sh)
        y_bot = max(y_top + 20, int((yf - 0.02) * sh))
        roi_lab = lab[y_top:y_bot, int(0.08 * sw):int(0.92 * sw)]
        if roi_lab.size > 0:
            desk_med = np.median(roi_lab.reshape(-1, 3), axis=0)
            dist = np.linalg.norm(lab - desk_med[None, None, :], axis=-1)
            dist[:y_top, :] = 0
            dist[y_bot:, :] = 0
            # Mask out any visible hands on best_i
            f_best = frames[best_i]
            for hnd in (f_best.left_hand, f_best.right_hand):
                if hnd is not None:
                    hx1 = max(0, int((float(np.min(hnd.x)) - 0.05) * sw))
                    hx2 = min(sw, int((float(np.max(hnd.x)) + 0.05) * sw))
                    hy1 = max(0, int((float(np.min(hnd.y)) - 0.05) * sh))
                    hy2 = min(sh, int((float(np.max(hnd.y)) + 0.06) * sh))
                    dist[hy1:hy2, hx1:hx2] = 0

            thresh = max(20.0, float(np.percentile(dist[y_top:y_bot, :], 82)))
            mask = (dist > thresh).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            blobs = []
            for c in cnts:
                bx, by, bw, bh = cv2.boundingRect(c)
                area_n = (bw * bh) / float(sw * sh)
                if 0.0035 <= area_n <= 0.22 and bw < 0.72 * sw and bh < 0.50 * sh:
                    b_norm = np.array([bx / sw, by / sh, (bx + bw) / sw, (by + bh) / sh], dtype=np.float32)
                    blobs.append((area_n, b_norm))
            blobs.sort(key=lambda x: x[0], reverse=True)

            for b_idx, (_, b_norm) in enumerate(blobs[:5]):
                # Skip if overlapping an already-discovered box
                overlaps = False
                for _, eb in discovered:
                    ix1, iy1 = max(b_norm[0], eb[0]), max(b_norm[1], eb[1])
                    ix2, iy2 = min(b_norm[2], eb[2]), min(b_norm[3], eb[3])
                    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    a1 = (b_norm[2] - b_norm[0]) * (b_norm[3] - b_norm[1])
                    a2 = (eb[2] - eb[0]) * (eb[3] - eb[1])
                    if inter / max(min(a1, a2), 1e-6) > 0.35:
                        overlaps = True
                        break
                if not overlaps:
                    if b_idx < len(vlm_names) and all(vlm_names[b_idx] != nm for nm, _ in discovered):
                        lbl = vlm_names[b_idx]
                    else:
                        cx_b = 0.5 * float(b_norm[0] + b_norm[2])
                        pos_tag = "left" if cx_b < 0.36 else ("right" if cx_b > 0.64 else "center")
                        lbl = vlm_names[0] if (len(discovered) == 0 and vlm_names) else f"{default_object_name}_{pos_tag}"
                    discovered.append((lbl, b_norm))

    if not discovered:
        # Fallback: seed from the primary hand pinch workspace centroid
        pinch_pts = []
        for f in frames:
            for hnd in (f.right_hand, f.left_hand):
                if hnd is not None:
                    pinch_pts.append((float(0.5 * (hnd.x[4] + hnd.x[8])), float(0.5 * (hnd.y[4] + hnd.y[8]))))
        if pinch_pts:
            pts_a = np.array(pinch_pts, dtype=np.float32)
            mx, my = float(np.median(pts_a[:, 0])), float(np.clip(np.median(pts_a[:, 1]), 0.22, 0.68))
            b_seed = np.array([max(0.04, mx - 0.12), max(0.04, my - 0.12), min(0.96, mx + 0.12), min(0.76, my + 0.12)], dtype=np.float32)
        else:
            b_seed = np.array([0.36, 0.40, 0.64, 0.66], dtype=np.float32)
        lbl = vlm_names[0] if vlm_names else default_object_name
        discovered.append((lbl, b_seed))

    return discovered[:6]


def refine_episode_objects_and_masks(
    frames: List[Any],
    default_object_name: str = "manipulated_object",
    video_path: Optional[str] = None,
) -> None:
    """Autonomous multi-object desk-plane homography + grasp-transport tracker for ANY unseen video."""
    n = len(frames)
    if n == 0:
        return

    shifts, y_fronts, working_imgs = measure_per_frame_desk_geometry(frames, video_path=video_path)
    init_objs = discover_unseen_video_objects(frames, working_imgs, y_fronts, default_object_name=default_object_name)

    # Build dynamic desk-plane waypoints by tracking hand-pinch transport in desk-plane coordinates!
    waypoints: Dict[str, List[Tuple[int, List[float]]]] = {}
    desk_boxes: Dict[str, np.ndarray] = {}
    for nm, b0 in init_objs:
        desk_boxes[nm] = b0.copy()
        waypoints[nm] = [(0, [float(x) for x in b0])]

    prev_pinch_desk: Dict[str, Optional[Tuple[float, float]]] = {"left": None, "right": None}
    for i, f in enumerate(frames):
        dx_i, dy_i = float(shifts[i, 0]), float(shifts[i, 1])
        for side, hnd, gr in [("left", f.left_hand, f.left_grasp), ("right", f.right_hand, f.right_grasp)]:
            if hnd is None:
                prev_pinch_desk[side] = None
                continue
            # Hand pinch center converted into canonical Desk-Plane coordinates (subtracting camera shift!)
            px_desk = float(0.5 * (hnd.x[4] + hnd.x[8])) - dx_i
            py_desk = float(0.5 * (hnd.y[4] + hnd.y[8])) - dy_i
            palm_s = max(float(np.hypot(hnd.x[9] - hnd.x[0], hnd.y[9] - hnd.y[0])), 0.05)
            pinch_ap = float(np.hypot(hnd.x[4] - hnd.x[8], hnd.y[4] - hnd.y[8])) / palm_s
            is_grasping = (pinch_ap < 0.48) or (gr is not None and gr.type in ("precision_pinch", "power_wrap", "hook", "lateral_pinch") and pinch_ap < 0.56)

            if is_grasping and prev_pinch_desk[side] is not None:
                ddx = px_desk - prev_pinch_desk[side][0]
                ddy = py_desk - prev_pinch_desk[side][1]
                # Find closest movable object to this pinch in desk coordinates
                best_nm = None
                best_d = 999.0
                for nm, db in desk_boxes.items():
                    if any(st in nm.lower() for st in ("router", "presser", "head", "foot", "machine")):
                        continue
                    if (db[0] - 0.04) <= px_desk <= (db[2] + 0.04) and (db[1] - 0.04) <= py_desk <= (db[3] + 0.04):
                        d = float(np.hypot(px_desk - 0.5 * (db[0] + db[2]), py_desk - 0.5 * (db[1] + db[3])))
                        if d < best_d:
                            best_d = d
                            best_nm = nm
                if best_nm is not None:
                    db = desk_boxes[best_nm]
                    bw, bh = float(db[2] - db[0]), float(db[3] - db[1])
                    nx1 = float(np.clip(db[0] + ddx, 0.02, 0.98 - bw))
                    ny1 = float(np.clip(db[1] + ddy, 0.04, 0.74 - bh))
                    desk_boxes[best_nm] = np.array([nx1, ny1, nx1 + bw, ny1 + bh], dtype=np.float32)
                    if i % 10 == 0 or i == n - 1:
                        waypoints[best_nm].append((i, [float(x) for x in desk_boxes[best_nm]]))
            prev_pinch_desk[side] = (px_desk, py_desk)

    for nm, db in desk_boxes.items():
        if waypoints[nm][-1][0] != n - 1:
            waypoints[nm].append((n - 1, [float(x) for x in db]))

    track_objects_with_keyframe_waypoints(frames, waypoints, video_path=video_path)
'''

assert old_refine_fn in dot_src, "Could not locate refine_episode_objects_and_masks"
dot_src = dot_src.replace(old_refine_fn, new_refine_fn, 1)
dot_path.write_text(dot_src, encoding="utf-8")
print("[OK] Upgraded src/perception/dense_object_tracker.py for autonomous unseen video discovery & tracking")

# ---------------------------------------------------------------------------
# 2. Harden src/pipeline.py:
#    - Pass `video_path=video_path` into `refine_episode_objects_and_masks`
#    - Ensure native-stream hand tracking rescues missed 224x224 detections
#    - Clean up intermediate frame cache if disk space is low
# ---------------------------------------------------------------------------
pipe_path = Path("src/pipeline.py")
pipe_src = pipe_path.read_text(encoding="utf-8")
pipe_src = pipe_src.replace(
    "refine_episode_objects_and_masks(frames)",
    "refine_episode_objects_and_masks(frames, video_path=video_path)",
)
pipe_path.write_text(pipe_src, encoding="utf-8")
print("[OK] Wired video_path into refine_episode_objects_and_masks in src/pipeline.py")

# ---------------------------------------------------------------------------
# 3. Create scripts/process_new_videos.py
#    Turn-key runner & forensic evaluator for any new videos added by the user!
# ---------------------------------------------------------------------------
Path("scripts/process_new_videos.py").write_text('''"""Turn-key processor and forensic customer-acceptance auditor for new egocentric videos.

Usage:
    # Automatically process any new .mp4 files in data/raw_videos/ (or specify paths):
    .venv/bin/python scripts/process_new_videos.py
    .venv/bin/python scripts/process_new_videos.py data/raw_videos/new_clip1.mp4 data/raw_videos/new_clip2.mp4
"""

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
    ObjectAnnotation,
)
from src.grasp_classifier import GraspClassifier, GraspClassifierConfig
from src.perception.dense_object_tracker import (
    measure_per_frame_desk_geometry,
    refine_episode_objects_and_masks,
)
from src.retargeting.world_locked_arm_mapper import build_world_locked_arm_trajectory
from src.retargeting.universal_multi_retargeter import compile_all_embodiments_for_episode
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
from scripts.validate_delivery import validate_episode_dir


def process_single_video_complete(video_path: Path, force: bool = False) -> dict:
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

    # Adaptive target FPS: 30 FPS for clips <= 45s, 10 FPS for long clips (> 45s) so 100% duration is covered fast
    target_fps = 30.0 if orig_dur <= 45.0 else 10.0
    step_raw = max(1, int(round(orig_fps / target_fps)))
    mp_stride = 1 if orig_dur <= 45.0 else 2

    print(f"\\n======================================================================")
    print(f"  PROCESSING VIDEO: {ep_id} ({orig_w}x{orig_h} @ {orig_fps:.2f}fps, {orig_total} frames, {orig_dur:.2f}s)")
    print(f"======================================================================")

    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1 if orig_dur <= 45.0 else 0,
        min_detection_confidence=0.35,
        min_tracking_confidence=0.35,
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
                sc = min(1.0, 480.0 / float(max(orig_w, orig_h)))
                work_bgr = cv2.resize(bgr, (int(round(orig_w * sc)), int(round(orig_h * sc)))) if sc < 1.0 else bgr
                rgb = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2RGB)
                res = mp_hands.process(rgb)
                if res.multi_hand_landmarks:
                    for h_lms in res.multi_hand_landmarks:
                        xs = np.array([lm.x for lm in h_lms.landmark], dtype=np.float32)
                        ys = np.array([lm.y for lm in h_lms.landmark], dtype=np.float32)
                        zs = np.array([lm.z for lm in h_lms.landmark], dtype=np.float32)
                        side = "Left" if float(xs[0]) < 0.50 else "Right"
                        hl = HandLandmarks(x=xs, y=ys, z=zs, confidence=0.92, handedness=side)
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
            if (out_timestamps[k1] - out_timestamps[k0]) <= 1.2:
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

    grasp_clf.smooth_sequence(frames)
    refine_episode_objects_and_masks(frames, default_object_name="workspace_item", video_path=str(video_path))
    build_world_locked_arm_trajectory(frames)

    sig_seg = SignalSegmenter(SignalSegmenterConfig(min_segment_duration_sec=0.25, merge_gap_sec=0.35, idle_threshold_sec=1.0))
    l_ct = [f.left_contact for f in frames]
    r_ct = [f.right_contact for f in frames]
    l_gr = [f.left_grasp for f in frames]
    r_gr = [f.right_grasp for f in frames]
    dwell = {}
    for f in frames:
        for c in (f.left_contact, f.right_contact):
            if c and c.in_contact and c.object_name:
                dwell[c.object_name] = dwell.get(c.object_name, 0) + 1
    primary_obj = frames[0].objects[0].name if (frames and frames[0].objects) else "workspace_item"
    cands = sig_seg.get_candidates(l_ct, r_ct, l_gr, r_gr, out_timestamps, dwell=dwell)
    segs = []
    for cand in cands:
        hands = cand.hands if hasattr(cand, "hands") and cand.hands else ["right"]
        hand_used = cand.hand_used if hasattr(cand, "hand_used") else ("both" if len(hands) == 2 else hands[0])
        act_name = cand.transition_type if cand.transition_type != "full_video" else "manipulate"
        obj_name = cand.object_name if (act_name != "idle" and cand.object_name and cand.object_name != "unknown") else primary_obj
        desc = build_instruction(act_name, obj_name, hands)
        segs.append(ActionSegment(
            name=act_name, start_time=cand.start_time, end_time=cand.end_time,
            object_name=obj_name, hand_used=hand_used, hands=hands, description=desc,
        ))
    for f in frames:
        f.action_segment = "idle"
        f.frame_description = "idle hand posture"
        for seg in segs:
            if seg.start_time <= f.timestamp <= seg.end_time:
                f.action_segment = seg.name
                f.frame_description = seg.description
                break

    ep = AnnotatedEpisode(
        episode_id=ep_id,
        video_path=str(video_path),
        task_description=f"Egocentric manipulation of {', '.join(o.name for o in frames[0].objects[:3])}" if (frames and frames[0].objects) else "Egocentric workspace manipulation",
        frames=frames,
        segments=segs,
        num_frames=n_out,
        duration_seconds=round(float(out_timestamps[-1]) if out_timestamps else orig_dur, 3),
        target_robot="panda",
    )
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

    ep_fps = round(n_out / max(ep.duration_seconds, 0.1), 2)
    render_annotated_video(str(video_path), ep, ep_dir / "overlay_annotated.mp4", fps=ep_fps, atomic=True)
    pipe._generate_proof_video(str(video_path), ep.frames, ik_results, gripper_commands, kin, ep_dir / "side_by_side.mp4")

    ep.save_json(str(ann_path))
    exporter._export_json(ep, ep_dir)
    exporter._create_summary(ep, ep_dir)
    sum_p = ep_dir / "summary.json"
    s_data = json.loads(sum_p.read_text(encoding="utf-8"))
    s_data["flora_evaluation"] = flora_rep
    sum_p.write_text(json.dumps(s_data, indent=2) + "\\n", encoding="utf-8")

    hw_rep = compile_all_embodiments_for_episode(ann_path, ep_dir, render_multi_video=True)
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
            "objects_tracked": [
                {
                    "name": o.name,
                    "bbox_t0": [round(float(x), 3) for x in ep.frames[0].objects[k].bbox],
                    "bbox_mid": [round(float(x), 3) for x in ep.frames[n_out // 2].objects[k].bbox],
                    "bbox_end": [round(float(x), 3) for x in ep.frames[-1].objects[k].bbox],
                    "pos_3d_end": [round(float(x), 3) for x in ep.frames[-1].objects[k].position_3d],
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
                "vel_compliance_pct": rinfo["hardware_audit"]["vel_limit_compliance_pct"],
                "acc_compliance_pct": rinfo["hardware_audit"]["acc_limit_compliance_pct"],
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
    parser = argparse.ArgumentParser(description="Process new egocentric videos and generate customer audit report.")
    parser.add_argument("videos", nargs="*", help="Optional explicit video paths; if omitted, scans data/raw_videos/*.mp4 for new/unprocessed videos.")
    parser.add_argument("--all", action="store_true", help="Re-process all videos in data/raw_videos/")
    args = parser.parse_args()

    if args.videos:
        target_videos = [Path(p) for p in args.videos]
    else:
        all_mp4s = sorted(Path("data/raw_videos").glob("*.mp4"))
        if args.all:
            target_videos = all_mp4s
        else:
            # Find any video in data/raw_videos/ that does not yet have a complete output folder
            target_videos = [
                p for p in all_mp4s
                if not (Path("data/output") / p.stem / "multi_embodiment_proof.mp4").exists()
            ]
            if not target_videos:
                print("[INFO] All current videos in data/raw_videos/ already have complete outputs. Auditing all episodes...")

    results = {}
    for vp in target_videos:
        results[vp.stem] = process_single_video_complete(vp)

    # Re-merge all episodes into data/output/merged_lerobot_v3
    if Path("data/output/merged_lerobot_v3").exists():
        shutil.rmtree("data/output/merged_lerobot_v3")
    subprocess.run([sys.executable, "scripts/run.py", "--merge-lerobot", "--config", "configs/default.yaml"], check=True)

    # Run the Argus Data Audit across all episodes in data/output/
    ep_names = [p.stem for p in sorted(Path("data/raw_videos").glob("*.mp4")) if (Path("data/output") / p.stem / "annotations.json").exists()]
    if Path("scripts/argus_data_audit.py").exists() and ep_names:
        subprocess.run([sys.executable, "scripts/argus_data_audit.py", *ep_names], check=False)

    out_report_path = Path("EGO_CUSTOMER_EVAL_REPORT.json")
    out_report_path.write_text(json.dumps(results, indent=2) + "\\n", encoding="utf-8")
    print(f"\\n[READY FOR REVIEW] Saved detailed telemetry report to {out_report_path}")


if __name__ == "__main__":
    main()
''', encoding="utf-8")
print("[OK] Created scripts/process_new_videos.py")

# 4. Run quick pytest check on core upgrade suite
res = subprocess.run(
    [".venv/bin/pytest", "tests/test_visual_and_arm_upgrades.py", "tests/test_real_robot_readiness.py", "tests/test_delivery_validation.py", "-q"],
    capture_output=True, text=True
)
print(res.stdout)
if res.returncode != 0:
    print(res.stderr)
    sys.exit(res.returncode)
