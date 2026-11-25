"""Behavior understanding: atomic action segmentation, Feix grasp taxonomy, HOI, bimanual & scene graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.spatial.transform import Rotation

from src.egoannotate.config import BehaviorConfig
from src.egoannotate.schemas import (
    AtomicActionSegment,
    FrameAnnotation,
    GraspEvent,
    HandState,
    SceneGraphSummary,
)

ATOMIC_ACTION_TAXONOMY = (
    "reach",
    "grasp",
    "pick",
    "transport",
    "manipulate",
    "pour_tilt",
    "place",
    "release",
    "push_pull",
    "handover",
    "idle",
)

HandSide = Literal["left", "right"]
HandRole = Literal["left", "right", "bimanual", "none"]
ManipPhase = Literal["approach", "contact_grasp", "transport_manipulate", "place_release", "idle"]
CoarseGrasp = Literal["power", "precision", "pinch", "lateral", "non_prehensile", "unknown"]
BimanualLabel = Literal[
    "unimanual_left",
    "unimanual_right",
    "symmetric_cooperative",
    "asymmetric_support",
    "handover",
    "independent_parallel",
    "idle",
]


@dataclass
class _RawSpan:
    start: int
    end: int
    verb: str
    hand: HandRole
    phase: ManipPhase


def compute_head_gaze_world_dir(quat_xyzw_world: list[float]) -> list[float]:
    """Return unit camera optical axis (+Z_c) expressed in gravity-aligned world frame as attention proxy."""
    rot_wc = Rotation.from_quat(quat_xyzw_world).as_matrix()
    gaze = rot_wc @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n = float(np.linalg.norm(gaze))
    if n < 1e-6:
        return [1.0, 0.0, 0.0]
    gaze = gaze / n
    return [round(float(c), 5) for c in gaze]


def classify_feix_grasp(hand: HandState, cfg: BehaviorConfig) -> tuple[str, CoarseGrasp]:
    """Classify articulated 21-joint hand pose into the Feix (2016) GRASP taxonomy."""
    if not hand.detected or len(hand.finger_curls_5d) < 5:
        return "none", "unknown"

    thumb, index, middle, ring, pinky = hand.finger_curls_5d
    aperture = hand.gripper_aperture_m
    all_four = 0.25 * (index + middle + ring + pinky)
    outer_three = (middle + ring + pinky) / 3.0

    if aperture <= cfg.grasp_pinch_aperture_m and index >= 0.30 and outer_three < 0.45:
        return "tip_pinch", "pinch"

    if aperture <= cfg.grasp_pinch_aperture_m * 1.25 and thumb >= 0.45 and index >= 0.45 and outer_three >= 0.50:
        return "lateral_pinch", "lateral"

    if aperture <= cfg.grasp_power_aperture_m * 0.80 and index >= 0.28 and middle >= 0.28 and pinky < 0.40:
        return "tripod_precision", "precision"

    if 0.22 <= all_four <= 0.55 and cfg.grasp_pinch_aperture_m < aperture <= cfg.grasp_power_aperture_m * 1.15:
        return "precision_disk", "precision"

    if all_four >= 0.52:
        if aperture <= cfg.grasp_pinch_aperture_m * 1.3:
            return "small_diameter_wrap", "power"
        return "medium_wrap", "power"

    if all_four < 0.22 and aperture > cfg.grasp_power_aperture_m * 0.85:
        return "flat_palm_support", "non_prehensile"

    return "light_tool_grasp", "precision"


def _hand_has_contact(
    frame: FrameAnnotation,
    side: HandSide,
    cfg: BehaviorConfig,
) -> tuple[bool, str, list[list[float]]]:
    hand = frame.left_hand if side == "left" else frame.right_hand
    if not hand.detected:
        return False, "none", []

    tips = np.asarray(hand.fingertips_world_m, dtype=np.float64) if hand.fingertips_world_m else np.zeros((0, 3))
    wrist = np.asarray(hand.wrist_pos_world_m, dtype=np.float64)

    for obj in frame.objects:
        if obj.in_hand in (side, "both"):
            pts = (
                [hand.fingertips_world_m[0], hand.fingertips_world_m[1]]
                if len(hand.fingertips_world_m) >= 2
                else [hand.wrist_pos_world_m]
            )
            return True, obj.label, pts
        obj_p = np.asarray(obj.pos_world_m, dtype=np.float64)
        min_d = float(np.linalg.norm(wrist - obj_p))
        if len(tips) > 0:
            min_d = min(min_d, float(np.min(np.linalg.norm(tips - obj_p[None, :], axis=1))))
        if min_d <= cfg.contact_proximity_m and (
            hand.gripper_aperture_m <= cfg.grasp_power_aperture_m or float(np.mean(hand.finger_curls_5d)) >= 0.30
        ):
            pts = (
                [hand.fingertips_world_m[0], hand.fingertips_world_m[1]]
                if len(hand.fingertips_world_m) >= 2
                else [hand.wrist_pos_world_m]
            )
            return True, obj.label, pts

    if float(np.mean(hand.finger_curls_5d)) >= 0.55 and hand.gripper_aperture_m <= cfg.grasp_pinch_aperture_m * 1.2:
        pts = (
            [hand.fingertips_world_m[0], hand.fingertips_world_m[1]]
            if len(hand.fingertips_world_m) >= 2
            else [hand.wrist_pos_world_m]
        )
        return True, "manipulated_object", pts

    return False, "none", []


def extract_grasp_events(
    frames: list[FrameAnnotation],
    cfg: BehaviorConfig,
) -> list[GraspEvent]:
    """Detect temporally coherent grasp onset -> hold -> release windows per hand across full video."""
    events: list[GraspEvent] = []
    if not frames:
        return events

    min_frames = max(3, round(cfg.min_segment_duration_s * 15.0))
    sides: tuple[HandSide, ...] = ("left", "right")

    for side in sides:
        active_start: int | None = None
        gap_count = 0
        window_labels: list[str] = []
        window_objs: list[str] = []
        window_contacts: list[list[float]] = []
        window_apertures: list[float] = []
        window_confs: list[float] = []

        def _flush(end_idx: int, cur_side: HandSide) -> None:
            nonlocal active_start, window_labels, window_objs, window_contacts, window_apertures, window_confs
            if active_start is None:
                return
            if (end_idx - active_start + 1) >= min_frames:
                feix = max(set(window_labels), key=window_labels.count) if window_labels else "unknown"
                mid_frame = frames[(active_start + end_idx) // 2]
                mid_hand = mid_frame.left_hand if cur_side == "left" else mid_frame.right_hand
                _, coarse = classify_feix_grasp(mid_hand, cfg)
                target_obj = max(set(window_objs), key=window_objs.count) if window_objs else "manipulated_object"
                ap_std = float(np.std(window_apertures)) if len(window_apertures) > 1 else 0.0
                stability = float(np.clip(1.0 - ap_std / 0.04, 0.15, 0.98))
                conf = float(np.clip(np.mean(window_confs), 0.1, 0.99)) if window_confs else 0.5
                events.append(
                    GraspEvent(
                        hand=cur_side,
                        onset_frame=frames[active_start].frame_idx,
                        release_frame=frames[end_idx].frame_idx,
                        onset_s=round(frames[active_start].timestamp_s, 4),
                        release_s=round(frames[end_idx].timestamp_s, 4),
                        feix_taxonomy=feix,
                        coarse_grasp=coarse,
                        target_object=target_obj,
                        contact_points_world_m=window_contacts[:4],
                        stability_proxy=round(stability, 4),
                        confidence=round(conf, 4),
                    )
                )
            active_start = None
            window_labels = []
            window_objs = []
            window_contacts = []
            window_apertures = []
            window_confs = []

        for idx, fr in enumerate(frames):
            hand = fr.left_hand if side == "left" else fr.right_hand
            in_contact, obj_lbl, c_pts = _hand_has_contact(fr, side, cfg)
            if in_contact:
                if active_start is None:
                    active_start = idx
                gap_count = 0
                feix_lbl, _ = classify_feix_grasp(hand, cfg)
                window_labels.append(feix_lbl)
                if obj_lbl != "none":
                    window_objs.append(obj_lbl)
                if c_pts and len(window_contacts) < 4:
                    window_contacts.append(c_pts[0])
                window_apertures.append(hand.gripper_aperture_m)
                window_confs.append(hand.confidence)
            elif active_start is not None:
                gap_count += 1
                if gap_count > 4:
                    _flush(idx - gap_count, side)
                    gap_count = 0

        if active_start is not None:
            _flush(len(frames) - 1, side)

    events.sort(key=lambda e: (e.onset_frame, e.hand))
    return events


def _frame_motion_state(fr: FrameAnnotation, cfg: BehaviorConfig) -> tuple[str, HandRole, ManipPhase]:
    l_contact, _, _ = _hand_has_contact(fr, "left", cfg)
    r_contact, _, _ = _hand_has_contact(fr, "right", cfg)
    l_spd = float(np.linalg.norm(fr.left_hand.wrist_vel_world_mps)) if fr.left_hand.detected else 0.0
    r_spd = float(np.linalg.norm(fr.right_hand.wrist_vel_world_mps)) if fr.right_hand.detected else 0.0
    l_vz = float(fr.left_hand.wrist_vel_world_mps[2]) if fr.left_hand.detected else 0.0
    r_vz = float(fr.right_hand.wrist_vel_world_mps[2]) if fr.right_hand.detected else 0.0

    hand_role: HandRole
    if l_contact and r_contact:
        hand_role = "bimanual"
    elif l_contact:
        hand_role = "left"
    elif r_contact:
        hand_role = "right"
    elif l_spd >= 0.08 and r_spd >= 0.08:
        hand_role = "bimanual"
    elif l_spd >= 0.08:
        hand_role = "left"
    elif r_spd >= 0.08:
        hand_role = "right"
    else:
        hand_role = "none"

    any_contact = l_contact or r_contact
    peak_spd = max(l_spd, r_spd)
    dom_vz = l_vz if l_spd >= r_spd else r_vz

    if any_contact:
        if peak_spd < 0.06:
            return "grasp", hand_role, "contact_grasp"
        if dom_vz > 0.14:
            return "pick", hand_role, "transport_manipulate"
        if dom_vz < -0.14:
            return "place", hand_role, "place_release"
        if peak_spd >= 0.18:
            return "transport", hand_role, "transport_manipulate"
        return "manipulate", hand_role, "transport_manipulate"

    if peak_spd >= 0.08:
        return "reach", hand_role, "approach"
    return "idle", hand_role, "idle"


def segment_atomic_actions(
    frames: list[FrameAnnotation],
    grasp_events: list[GraspEvent],
    cfg: BehaviorConfig,
) -> list[AtomicActionSegment]:
    """Segment continuous egocentric frames (0..N-1) into temporally smoothed atomic manipulation segments."""
    if not frames:
        return []

    fps = 30.0
    if len(frames) >= 2 and frames[-1].timestamp_s > frames[0].timestamp_s:
        fps = (len(frames) - 1) / (frames[-1].timestamp_s - frames[0].timestamp_s)

    raw_labels = [_frame_motion_state(fr, cfg) for fr in frames]

    spans: list[_RawSpan] = []
    cur_start = 0
    for i in range(1, len(raw_labels) + 1):
        if i == len(raw_labels) or raw_labels[i][0] != raw_labels[cur_start][0]:
            spans.append(
                _RawSpan(
                    start=cur_start,
                    end=i - 1,
                    verb=raw_labels[cur_start][0],
                    hand=raw_labels[cur_start][1],
                    phase=raw_labels[cur_start][2],
                )
            )
            cur_start = i

    min_frames = max(3, round(cfg.min_segment_duration_s * fps))
    max_span_frames = round(10.0 * fps)
    merged: list[_RawSpan] = []
    for sp in spans:
        length = sp.end - sp.start + 1
        if merged and (
            (sp.verb == merged[-1].verb and (sp.end - merged[-1].start) < max_span_frames)
            or (length < min_frames and (sp.end - merged[-1].start) < max_span_frames)
        ):
            merged[-1].end = sp.end
            if sp.verb != "idle" and merged[-1].verb == "idle":
                merged[-1].verb = sp.verb
                merged[-1].hand = sp.hand
                merged[-1].phase = sp.phase
        else:
            merged.append(sp)

    segments: list[AtomicActionSegment] = []
    for seg_id, sp in enumerate(merged):
        sub_frames = frames[sp.start : sp.end + 1]
        if not sub_frames:
            continue

        l_active = sum(
            1
            for f in sub_frames
            if f.left_hand.detected
            and (np.linalg.norm(f.left_hand.wrist_vel_world_mps) > 0.06 or _hand_has_contact(f, "left", cfg)[0])
        )
        r_active = sum(
            1
            for f in sub_frames
            if f.right_hand.detected
            and (np.linalg.norm(f.right_hand.wrist_vel_world_mps) > 0.06 or _hand_has_contact(f, "right", cfg)[0])
        )
        n_sub = max(len(sub_frames), 1)
        l_frac = l_active / n_sub
        r_frac = r_active / n_sub

        hand_role: HandRole
        bimanual_lbl: BimanualLabel

        if l_frac >= 0.30 and r_frac >= 0.30:
            hand_role = "bimanual"
            dists = [
                float(np.linalg.norm(np.array(f.left_hand.wrist_pos_world_m) - np.array(f.right_hand.wrist_pos_world_m)))
                for f in sub_frames
                if f.left_hand.detected and f.right_hand.detected
            ]
            l_vel_mean = float(np.mean([np.linalg.norm(f.left_hand.wrist_vel_world_mps) for f in sub_frames]))
            r_vel_mean = float(np.mean([np.linalg.norm(f.right_hand.wrist_vel_world_mps) for f in sub_frames]))
            if dists and min(dists) < 0.14 and abs(l_vel_mean - r_vel_mean) < 0.20:
                bimanual_lbl = "symmetric_cooperative"
            elif max(l_vel_mean, r_vel_mean) > 2.0 * max(min(l_vel_mean, r_vel_mean), 0.02):
                bimanual_lbl = "asymmetric_support"
            else:
                bimanual_lbl = "independent_parallel"
        elif l_frac >= 0.20:
            hand_role = "left"
            bimanual_lbl = "unimanual_left"
        elif r_frac >= 0.20:
            hand_role = "right"
            bimanual_lbl = "unimanual_right"
        else:
            hand_role = "none"
            bimanual_lbl = "idle"

        f_start = sub_frames[0].frame_idx
        f_end = sub_frames[-1].frame_idx
        rel_hands = {g.hand for g in grasp_events if f_start <= g.release_frame <= f_end}
        ons_hands = {g.hand for g in grasp_events if f_start <= g.onset_frame <= f_end}
        verb = sp.verb
        phase: ManipPhase = sp.phase
        if len(rel_hands) == 1 and len(ons_hands) == 1 and rel_hands != ons_hands:
            verb = "handover"
            bimanual_lbl = "handover"
            hand_role = "bimanual"
            phase = "transport_manipulate"
        elif verb == "idle" and rel_hands:
            verb = "release"
            phase = "place_release"

        obj_names: list[str] = []
        for f in sub_frames:
            for obj in f.objects:
                if obj.label not in obj_names:
                    obj_names.append(obj.label)
        if not obj_names and verb != "idle":
            obj_names = ["manipulated_object"]

        overlapping_grasps = [
            g for g in grasp_events if not (g.release_frame < f_start or g.onset_frame > f_end)
        ]
        grasp_type = overlapping_grasps[0].feix_taxonomy if overlapping_grasps else "none"

        dom_side: HandSide = "left" if hand_role == "left" else "right"
        positions = [
            np.asarray((f.left_hand if dom_side == "left" else f.right_hand).wrist_pos_world_m, dtype=np.float64)
            for f in sub_frames
            if (f.left_hand if dom_side == "left" else f.right_hand).detected
        ]
        if len(positions) >= 2:
            delta = positions[-1] - positions[0]
            dn = float(np.linalg.norm(delta))
            approach_dir = (delta / dn).tolist() if dn > 1e-4 else [0.0, 0.0, 0.0]
        else:
            approach_dir = [0.0, 0.0, 0.0]

        speeds = [
            max(
                float(np.linalg.norm(f.left_hand.wrist_vel_world_mps)) if f.left_hand.detected else 0.0,
                float(np.linalg.norm(f.right_hand.wrist_vel_world_mps)) if f.right_hand.detected else 0.0,
            )
            for f in sub_frames
        ]
        peak_speed = float(max(speeds)) if speeds else 0.0

        gaze_scores: list[float] = []
        for f in sub_frames:
            h = f.left_hand if dom_side == "left" else f.right_hand
            if not h.detected:
                continue
            cam_p = np.asarray(f.camera_pose.pos_world_m, dtype=np.float64)
            hand_p = np.asarray(h.wrist_pos_world_m, dtype=np.float64)
            to_hand = hand_p - cam_p
            tn = float(np.linalg.norm(to_hand))
            if tn > 1e-4:
                cos_sim = float(np.dot(np.asarray(f.gaze_world_dir, dtype=np.float64), to_hand / tn))
                gaze_scores.append(float(np.clip(0.5 * (cos_sim + 1.0), 0.0, 1.0)))
        gaze_align = float(np.mean(gaze_scores)) if gaze_scores else 0.5

        state_map = {
            "reach": ("stationary", "approached"),
            "grasp": ("approached", "held"),
            "pick": ("resting_on_surface", "lifted_held"),
            "transport": ("lifted_held", "relocated_held"),
            "manipulate": ("held", "manipulated"),
            "handover": ("held_by_initial_hand", "held_by_receiving_hand"),
            "place": ("lifted_held", "resting_on_surface"),
            "release": ("held", "detached_stationary"),
            "idle": ("stationary", "stationary"),
        }
        st_before, st_after = state_map.get(verb, ("unknown", "unknown"))

        confs = [
            max(
                f.left_hand.confidence if f.left_hand.detected else 0.0,
                f.right_hand.confidence if f.right_hand.detected else 0.0,
            )
            for f in sub_frames
        ]
        seg_conf = float(np.mean(confs)) if confs else 0.3

        segments.append(
            AtomicActionSegment(
                segment_id=seg_id,
                start_frame=f_start,
                end_frame=f_end,
                start_s=round(sub_frames[0].timestamp_s, 4),
                end_s=round(sub_frames[-1].timestamp_s, 4),
                verb=verb,
                acting_hand=hand_role,
                target_objects=obj_names[:3],
                bimanual_coordination=bimanual_lbl,
                grasp_type=grasp_type,
                manipulation_phase=phase,
                object_state_before=st_before,
                object_state_after=st_after,
                approach_direction_world=[round(float(c), 4) for c in approach_dir],
                peak_hand_speed_mps=round(peak_speed, 4),
                gaze_alignment_score=round(gaze_align, 4),
                confidence=round(float(np.clip(seg_conf, 0.0, 1.0)), 4),
            )
        )

    return segments


def build_scene_graph_and_affordances(
    frames: list[FrameAnnotation],
    grasp_events: list[GraspEvent],
    sparse_world_points: list[list[float]],
) -> SceneGraphSummary:
    """Build world-frame scene graph, workspace reachability bounds, and contact affordance hotspots."""
    nodes: list[dict[str, Any]] = [
        {"id": "wearer_head", "type": "camera_agent"},
        {"id": "left_hand", "type": "manipulator"},
        {"id": "right_hand", "type": "manipulator"},
    ]
    seen_objs: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    hand_pts: list[list[float]] = []

    for fr in frames:
        if fr.left_hand.detected:
            hand_pts.append(fr.left_hand.wrist_pos_world_m)
        if fr.right_hand.detected:
            hand_pts.append(fr.right_hand.wrist_pos_world_m)
        for obj in fr.objects:
            if obj.label not in seen_objs:
                seen_objs[obj.label] = {
                    "id": obj.label,
                    "type": "manipulated_object",
                    "track_id": obj.track_id,
                    "last_pos_world_m": obj.pos_world_m,
                }
            else:
                seen_objs[obj.label]["last_pos_world_m"] = obj.pos_world_m

    nodes.extend(seen_objs.values())

    affordances: list[dict[str, Any]] = []
    for idx, g in enumerate(grasp_events):
        edges.append(
            {
                "source": f"{g.hand}_hand",
                "relation": f"grasps:{g.feix_taxonomy}",
                "target": g.target_object,
                "start_s": g.onset_s,
                "end_s": g.release_s,
                "confidence": g.confidence,
            }
        )
        if g.contact_points_world_m:
            affordances.append(
                {
                    "hotspot_id": idx,
                    "object": g.target_object,
                    "grasp_type": g.feix_taxonomy,
                    "pos_world_m": g.contact_points_world_m[0],
                    "confidence": g.confidence,
                }
            )

    workspace: dict[str, list[float]] = {}
    if hand_pts:
        arr = np.asarray(hand_pts, dtype=np.float64)
        workspace = {
            "min_xyz_m": [round(float(v), 4) for v in np.min(arr, axis=0)],
            "max_xyz_m": [round(float(v), 4) for v in np.max(arr, axis=0)],
            "center_xyz_m": [round(float(v), 4) for v in np.mean(arr, axis=0)],
        }

    return SceneGraphSummary(
        nodes=nodes,
        edges=edges,
        workspace_bounds_world_m=workspace,
        affordance_hotspots_world_m=affordances,
        point_cloud_sample_world_m=sparse_world_points[:500],
    )


def enrich_frames_with_gaze(frames: list[FrameAnnotation]) -> None:
    """Populate world-frame head-gaze direction proxy on every frame."""
    for fr in frames:
        fr.gaze_world_dir = compute_head_gaze_world_dir(fr.camera_pose.quat_xyzw_world)
