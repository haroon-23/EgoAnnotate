"""Canonical robot-ready action contracts, gap-free 3-level hierarchy, and contact state machine."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy.spatial.transform import Rotation

from src.egoannotate.schemas import (
    AtomicActionSegment,
    FrameAnnotation,
    HierarchicalLanguage,
)

CANONICAL_DELTA_ACTION_NAMES = [
    "left_dx_m",
    "left_dy_m",
    "left_dz_m",
    "left_droll_rad",
    "left_dpitch_rad",
    "left_dyaw_rad",
    "left_daperture_m",
    "left_contact_state",
    "right_dx_m",
    "right_dy_m",
    "right_dz_m",
    "right_droll_rad",
    "right_dpitch_rad",
    "right_dyaw_rad",
    "right_daperture_m",
    "right_contact_state",
]


def _delta_rot_rpy(q_prev: list[float], q_next: list[float]) -> np.ndarray:
    qa = np.asarray(q_prev, dtype=np.float64)
    qb = np.asarray(q_next, dtype=np.float64)
    if qa.size != 4 or qb.size != 4:
        return np.zeros(3, dtype=np.float32)
    na = float(np.linalg.norm(qa))
    nb = float(np.linalg.norm(qb))
    if na < 1e-6 or nb < 1e-6:
        return np.zeros(3, dtype=np.float32)
    ra = Rotation.from_quat(qa / na)
    rb = Rotation.from_quat(qb / nb)
    rel = rb * ra.inv()
    return rel.as_euler("xyz", degrees=False).astype(np.float32)


def populate_canonical_delta_actions(frames: list[FrameAnnotation]) -> dict[str, np.ndarray]:
    """Populate 16D embodiment-agnostic bimanual delta actions & proprio_valid masks on all frames."""
    n = len(frames)
    actions = np.zeros((n, 16), dtype=np.float32)
    proprio_valid = np.zeros((n,), dtype=np.bool_)

    for i in range(n):
        fr = frames[i]
        nxt = frames[i + 1] if (i + 1 < n) else fr

        valid = bool(fr.camera_pose.tracking_ok and (fr.left_hand.detected or fr.right_hand.detected))
        fr.proprio_valid = valid
        proprio_valid[i] = valid

        row = np.zeros(16, dtype=np.float32)
        for offset, side in ((0, "left"), (8, "right")):
            h_cur = fr.left_hand if side == "left" else fr.right_hand
            h_nxt = nxt.left_hand if side == "left" else nxt.right_hand
            tac = fr.tactile_left if side == "left" else fr.tactile_right
            if h_cur.detected and h_nxt.detected:
                dp = np.asarray(h_nxt.wrist_pos_world_m, dtype=np.float32) - np.asarray(
                    h_cur.wrist_pos_world_m, dtype=np.float32
                )
                drpy = _delta_rot_rpy(h_cur.wrist_quat_xyzw_world, h_nxt.wrist_quat_xyzw_world)
                dap = float(h_nxt.gripper_aperture_m - h_cur.gripper_aperture_m)
                row[offset : offset + 3] = dp
                row[offset + 3 : offset + 6] = drpy
                row[offset + 6] = dap
            row[offset + 7] = 1.0 if float(tac.contact_prob) >= 0.5 else 0.0

        actions[i] = row
        fr.canonical_delta_action = [round(float(v), 6) for v in row.tolist()]

    return {
        "canonical_delta_action": actions,
        "proprio_valid": proprio_valid,
    }


def annotate_contact_and_tactile_events(frames: list[FrameAnnotation]) -> dict[str, int]:
    """Derive frame-accurate grasp_onset, sustained_contact, tool_workpiece_contact, and release tags."""
    counts = {
        "grasp_onset": 0,
        "sustained_contact": 0,
        "tool_workpiece_contact": 0,
        "release": 0,
    }
    prev_contact = {"left": False, "right": False}

    for fr in frames:
        tags: list[str] = []
        max_iou = max((float(o.mask_iou_with_hand) for o in fr.objects), default=0.0)
        has_tool = any(o.role == "tool" or "tool" in o.label.lower() for o in fr.objects)

        for side in ("left", "right"):
            tac = fr.tactile_left if side == "left" else fr.tactile_right
            hand = fr.left_hand if side == "left" else fr.right_hand
            vel_mag = float(np.linalg.norm(np.asarray(hand.wrist_vel_world_mps, dtype=np.float32)))
            slow_contact = vel_mag < 0.28
            in_contact = bool(
                hand.detected
                and (float(tac.contact_prob) >= 0.52 or (max_iou >= 0.08 and slow_contact))
            )
            if in_contact and not prev_contact[side]:
                tags.append(f"{side}:grasp_onset")
                counts["grasp_onset"] += 1
            elif in_contact and prev_contact[side]:
                tags.append(f"{side}:sustained_contact")
                counts["sustained_contact"] += 1
            elif not in_contact and prev_contact[side]:
                tags.append(f"{side}:release")
                counts["release"] += 1
            prev_contact[side] = in_contact

        if has_tool and (prev_contact["left"] or prev_contact["right"]):
            tags.append("tool_workpiece_contact")
            counts["tool_workpiece_contact"] += 1

        fr.contact_event_tags = tags
    return counts


def enforce_gap_free_hierarchical_segments(
    frames: list[FrameAnnotation],
    segments: list[AtomicActionSegment],
    language: HierarchicalLanguage,
) -> list[AtomicActionSegment]:
    """Guarantee 100% gap-free 0..N-1 timeline coverage, outcome tags, and 5-30s Level 1 subtasks."""
    if not frames:
        return segments

    frames_by_idx = {f.frame_idx: f for f in frames}
    first_fi = frames[0].frame_idx
    last_fi = frames[-1].frame_idx

    ordered = sorted(segments, key=lambda s: (s.start_frame, s.end_frame))
    normalized: list[AtomicActionSegment] = []
    cursor = first_fi

    def _make_idle(sf: int, ef: int, seg_id: int) -> AtomicActionSegment:
        s_t = frames_by_idx[sf].timestamp_s if sf in frames_by_idx else float(sf) / 30.0
        e_t = frames_by_idx[ef].timestamp_s if ef in frames_by_idx else float(ef) / 30.0
        return AtomicActionSegment(
            segment_id=seg_id,
            start_frame=sf,
            end_frame=ef,
            start_s=round(s_t, 4),
            end_s=round(e_t, 4),
            verb="idle",
            acting_hand="none",
            bimanual_coordination="idle",
            manipulation_phase="idle",
            semantic_description="Hold position stationary over workspace.",
            outcome_tag="idle",
            confidence=0.9,
        )

    for seg in ordered:
        sf = max(cursor, int(seg.start_frame))
        ef = min(last_fi, int(seg.end_frame))
        if ef < sf:
            continue
        if sf > cursor:
            normalized.append(_make_idle(cursor, sf - 1, len(normalized)))
        seg_copy = seg.model_copy(
            update={
                "segment_id": len(normalized),
                "start_frame": sf,
                "end_frame": ef,
                "start_s": round(
                    frames_by_idx[sf].timestamp_s if sf in frames_by_idx else seg.start_s, 4
                ),
                "end_s": round(
                    frames_by_idx[ef].timestamp_s if ef in frames_by_idx else seg.end_s, 4
                ),
            }
        )
        normalized.append(seg_copy)
        cursor = ef + 1

    if cursor <= last_fi:
        normalized.append(_make_idle(cursor, last_fi, len(normalized)))

    for idx, seg in enumerate(normalized):
        if seg.verb == "idle":
            seg.outcome_tag = "idle"
            continue
        sub = [frames_by_idx[fi] for fi in range(seg.start_frame, seg.end_frame + 1) if fi in frames_by_idx]
        max_slip = max(
            (max(float(f.tactile_left.slip_proxy_01), float(f.tactile_right.slip_proxy_01)) for f in sub),
            default=0.0,
        )
        prev_seg = normalized[idx - 1] if idx > 0 else None
        outcome: Literal["success", "retry", "slip_recovery", "idle"] = "success"
        if max_slip >= 0.65:
            outcome = "slip_recovery"
        elif (
            prev_seg is not None
            and prev_seg.verb == seg.verb
            and prev_seg.verb in ("grasp", "reach", "pick")
            and (seg.start_s - prev_seg.end_s) <= 2.0
        ):
            outcome = "retry"
        seg.outcome_tag = outcome

    subtasks: list[dict[str, Any]] = []
    group: list[AtomicActionSegment] = []

    def _commit_group() -> None:
        nonlocal group
        if not group:
            return
        st_id = len(subtasks)
        for s in group:
            s.subtask_id = st_id
        non_idle = [s for s in group if s.verb != "idle"]
        objs = sorted({o for s in non_idle for o in s.target_objects})
        obj_txt = objs[0].replace("_", " ") if objs else "workspace object"
        verbs = [s.verb for s in non_idle]
        if not verbs:
            goal = "Observe and stabilize workspace posture."
        elif "pick" in verbs and "place" in verbs:
            goal = f"Pick and place {obj_txt}."
        elif "manipulate" in verbs:
            goal = f"Manipulate {obj_txt}."
        else:
            goal = f"{verbs[0].replace('_', ' ').capitalize()} {obj_txt}."
        subtasks.append(
            {
                "subtask_id": st_id,
                "start_frame": group[0].start_frame,
                "end_frame": group[-1].end_frame,
                "start_s": group[0].start_s,
                "end_s": group[-1].end_s,
                "segment_ids": [s.segment_id for s in group],
                "goal": goal,
            }
        )
        group = []

    for seg in normalized:
        group.append(seg)
        span_s = group[-1].end_s - group[0].start_s
        if (span_s >= 5.0 and seg.verb in ("place", "release", "idle")) or span_s >= 28.0:
            _commit_group()
    _commit_group()

    language.subtasks = subtasks
    return normalized
