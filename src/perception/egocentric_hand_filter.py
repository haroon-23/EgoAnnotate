from __future__ import annotations
"""Egocentric Camera-Wearer Hand Isolation & Continuous 30Hz Trajectory Filter."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def _get_xyz(h: Optional[Dict[str, Any]]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if not h or not isinstance(h, dict):
        return None
    if "x" in h and "y" in h and len(h["x"]) >= 21:
        xs = np.asarray(h["x"][:21], dtype=np.float64)
        ys = np.asarray(h["y"][:21], dtype=np.float64)
        zs = np.asarray(h.get("z", [0.0] * 21)[:21], dtype=np.float64)
        return xs, ys, zs
    if "landmarks" in h and isinstance(h["landmarks"], list) and len(h["landmarks"]) >= 21:
        lms = h["landmarks"][:21]
        xs = np.asarray([float(p["x"]) for p in lms], dtype=np.float64)
        ys = np.asarray([float(p["y"]) for p in lms], dtype=np.float64)
        zs = np.asarray([float(p.get("z", 0.0)) for p in lms], dtype=np.float64)
        return xs, ys, zs
    return None


def _sync_hand_dict(
    h: Dict[str, Any],
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    side: str,
    is_interp: bool = False,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    h["handedness"] = side
    h["is_interpolated"] = bool(is_interp)
    if source:
        h["source"] = source
    elif is_interp:
        h["source"] = "interpolated"
    elif "source" not in h:
        h["source"] = "detected"
    if "confidence" not in h:
        h["confidence"] = 0.85
    h["x"] = [round(float(v), 5) for v in xs]
    h["y"] = [round(float(v), 5) for v in ys]
    h["z"] = [round(float(v), 5) for v in zs]
    h["landmarks"] = [{"x": h["x"][k], "y": h["y"][k], "z": h["z"][k]} for k in range(21)]
    return h


def is_valid_egocentric_wearer_hand(
    h: Optional[Dict[str, Any]],
    expected_side: str,
) -> Tuple[bool, str, float]:
    xyz = _get_xyz(h)
    if xyz is None:
        return False, "empty", 0.0
    xs, ys, _ = xyz

    wx, wy = float(xs[0]), float(ys[0])
    mcp9_x, mcp9_y = float(xs[9]), float(ys[9])
    tip12_x, tip12_y = float(xs[12]), float(ys[12])
    cy = float(np.mean(ys))

    if wy < 0.20 or cy < 0.18 or wx < 0.01 or wx > 0.99:
        return False, "bystander_top_or_edge_entry", 0.0

    # Allow downward/forward reach on tabletop in egocentric perspective
    forearm_dy = wy - mcp9_y
    if forearm_dy < -0.38:
        return False, "bystander_inverted_forearm", 0.0

    palm_len = float(np.hypot(mcp9_x - wx, mcp9_y - wy))
    if palm_len < 0.038 or palm_len > 0.46:
        return False, "invalid_palm_scale", 0.0

    # Support closed grasp / curled fist where fingertips curl into palm (< 0.15 palm_len)
    finger_len = float(np.hypot(tip12_x - mcp9_x, tip12_y - mcp9_y))
    if finger_len < 0.04 * palm_len or finger_len > 2.85 * palm_len:
        return False, "anatomical_proportion_invalid", 0.0

    # Support lateral reach across workspace
    if expected_side == "Left" and wx > 0.96:
        return False, "bystander_lateral_mismatch", 0.0
    if expected_side == "Right" and wx < 0.04:
        return False, "bystander_lateral_mismatch", 0.0

    ego_score = palm_len * float(np.exp(-1.2 * (1.0 - wy))) * (1.0 + max(0.0, forearm_dy))
    return True, "ok", ego_score


def filter_episode_for_egocentric_wearer_only(frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "bystander_hands_removed": 0,
        "clean_wearer_hand_frames": 0,
        "rejections_by_reason": {},
        "left_hand_detected_count": 0,
        "right_hand_detected_count": 0,
    }
    n = len(frames)

    for side_key, side_label in [("left_hand", "Left"), ("right_hand", "Right")]:
        for i, f in enumerate(frames):
            h = f.get(side_key)
            if h is None:
                continue
            ok, reason, _ = is_valid_egocentric_wearer_hand(h, side_label)
            if not ok:
                f[side_key] = None
                stats["bystander_hands_removed"] += 1
                reasons = stats["rejections_by_reason"]
                reasons[reason] = reasons.get(reason, 0) + 1
            else:
                xyz = _get_xyz(h)
                if xyz is not None:
                    src = h.get("source", "detected")
                    is_int = bool(h.get("is_interpolated", False) or src == "interpolated")
                    _sync_hand_dict(h, xyz[0], xyz[1], xyz[2], side_label, is_interp=is_int, source=src)
                    if side_label == "Left":
                        stats["left_hand_detected_count"] += 1
                    else:
                        stats["right_hand_detected_count"] += 1

        # Bridge keyframe intervals (<= 60 frames) when the wearer hand stays inside the workspace
        valid_indices = [i for i in range(n) if frames[i].get(side_key) is not None]
        for idx_pos in range(len(valid_indices) - 1):
            i0, i1 = valid_indices[idx_pos], valid_indices[idx_pos + 1]
            gap = i1 - i0 - 1
            if 1 <= gap <= 60:
                xyz0 = _get_xyz(frames[i0][side_key])
                xyz1 = _get_xyz(frames[i1][side_key])
                if xyz0 and xyz1:
                    dist_gap = float(np.hypot(xyz1[0][0] - xyz0[0][0], xyz1[1][0] - xyz0[1][0]))
                    for g_step in range(1, gap + 1):
                        tau = g_step / float(gap + 1)
                        s_tau = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
                        i_prev = valid_indices[max(0, idx_pos - 1)]
                        i_next = valid_indices[min(len(valid_indices) - 1, idx_pos + 2)]
                        xyz_prev = _get_xyz(frames[i_prev][side_key]) or xyz0
                        xyz_next = _get_xyz(frames[i_next][side_key]) or xyz1
                        tan_x = 0.5 * (xyz_next[0] - xyz_prev[0])
                        tan_y = 0.5 * (xyz_next[1] - xyz_prev[1])
                        tan_z = 0.5 * (xyz_next[2] - xyz_prev[2])
                        if float(np.max(np.abs(tan_x))) < 1e-5:
                            palm_vx = (xyz0[0][9] - xyz0[0][0]) * 0.04
                            palm_vy = (xyz0[1][9] - xyz0[1][0]) * 0.04
                            tan_x = np.linspace(0.8, 1.2, 21) * palm_vx
                            tan_y = np.linspace(0.8, 1.2, 21) * palm_vy
                            tan_z = np.linspace(0.5, 1.5, 21) * 0.015
                        hermite_arch = tau * (1.0 - tau) * (1.0 - 2.0 * tau)
                        xs_i = (1.0 - s_tau) * xyz0[0] + s_tau * xyz1[0] + hermite_arch * tan_x
                        ys_i = (1.0 - s_tau) * xyz0[1] + s_tau * xyz1[1] + hermite_arch * tan_y
                        zs_i = (1.0 - s_tau) * xyz0[2] + s_tau * xyz1[2] + hermite_arch * tan_z
                        if dist_gap <= 0.45:
                            frames[i0 + g_step][side_key] = _sync_hand_dict(
                                {"confidence": 0.75},
                                xs_i, ys_i, zs_i, side_label, is_interp=True, source="interpolated"
                            )

    # Final pass: maintain chirality consistency across frames without deleting hands
    prev_l_x, prev_r_x = None, None
    for f in frames:
        lh, rh = f.get("left_hand"), f.get("right_hand")
        lx = float(lh["landmarks"][0]["x"]) if (lh and lh.get("landmarks")) else None
        rx = float(rh["landmarks"][0]["x"]) if (rh and rh.get("landmarks")) else None
        if lx is not None and rx is not None and prev_l_x is not None and prev_r_x is not None:
            d_keep = abs(lx - prev_l_x) + abs(rx - prev_r_x)
            d_swap = abs(lx - prev_r_x) + abs(rx - prev_l_x)
            if d_swap + 0.08 < d_keep:
                # Swapped detection labels: swap back to track continuity, preserving BOTH hands
                f["left_hand"], f["right_hand"] = f["right_hand"], f["left_hand"]
                lx, rx = rx, lx
        if lx is not None:
            prev_l_x = lx
        if rx is not None:
            prev_r_x = rx

    for f in frames:
        if f.get("left_hand") is not None or f.get("right_hand") is not None:
            stats["clean_wearer_hand_frames"] += 1

    return stats
