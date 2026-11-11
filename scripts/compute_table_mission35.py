#!/usr/bin/env python3
"""Compute exact Mission 3.5 Table across all 9 episodes."""
import json
from pathlib import Path
import numpy as np


def is_touching_border(hand_dict, border_margin=0.01):
    """Check if any 2D landmark of the hand touches the frame border [0.01, 0.99]."""
    if not hand_dict:
        return False
    xs = hand_dict.get("x")
    ys = hand_dict.get("y")
    if not xs or not ys:
        lms = hand_dict.get("landmarks", [])
        if lms:
            xs = [p["x"] for p in lms]
            ys = [p["y"] for p in lms]
    if not xs or not ys:
        return False
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    # Touching border if any point <= margin or >= 1 - margin
    return (min_x <= border_margin or max_x >= (1.0 - border_margin) or
            min_y <= border_margin or max_y >= (1.0 - border_margin))


def compute_episode_row(ep_id: str, device_profile: str = "wide_fisheye_actioncam") -> dict:
    ep_dir = Path("data/output") / ep_id
    ann_p = ep_dir / "annotations.json"
    if not ann_p.exists():
        return {"id": ep_id, "error": "missing annotations.json"}

    ann = json.loads(ann_p.read_text(encoding="utf-8"))
    frames = ann.get("frames", [])
    n = len(frames)
    if n == 0:
        return {"id": ep_id, "error": "0 frames"}

    detected_cnt = 0
    rescued_cnt = 0
    interpolated_cnt = 0
    left_cnt = 0
    right_cnt = 0
    untruncated_frames_cnt = 0

    for f in frames:
        lh = f.get("left_hand")
        rh = f.get("right_hand")

        def get_stat(h):
            if not h:
                return "none"
            is_int = h.get("is_interpolated", False)
            src = str(h.get("source", "interpolated" if is_int else "detected")).lower()
            if is_int or src in ("interpolated", "held"):
                return "interpolated"
            if src == "rescued":
                return "rescued"
            return "detected"

        l_stat = get_stat(lh)
        r_stat = get_stat(rh)

        if l_stat != "none":
            left_cnt += 1
        if r_stat != "none":
            right_cnt += 1

        # Hand presence classification for the frame
        if l_stat == "detected" or r_stat == "detected":
            detected_cnt += 1
        elif l_stat == "rescued" or r_stat == "rescued":
            rescued_cnt += 1
        elif l_stat == "interpolated" or r_stat == "interpolated":
            interpolated_cnt += 1

        # Check border truncation
        has_hand = (lh is not None or rh is not None)
        l_touch = is_touching_border(lh) if lh else False
        r_touch = is_touching_border(rh) if rh else False
        if has_hand and not (l_touch or r_touch):
            untruncated_frames_cnt += 1

    det_ratio = round(detected_cnt / n, 4)
    rescued_ratio = round(rescued_cnt / n, 4)
    interp_ratio = round(interpolated_cnt / n, 4)
    left_ratio = round(left_cnt / n, 4)
    right_ratio = round(right_cnt / n, 4)
    untruncated_ratio = round(untruncated_frames_cnt / max(1, (detected_cnt + rescued_cnt + interpolated_cnt)), 4)

    # Segment durations
    segments = ann.get("segments", [])
    seg_durs = [float(s.get("end_time", 0)) - float(s.get("start_time", 0)) for s in segments] if segments else [0.0]
    max_seg = max(seg_durs) if seg_durs else 0.0
    min_seg = min(seg_durs) if seg_durs else 0.0

    # Decision logic
    reasons = []
    if det_ratio >= 0.60:
        tier = "ACCEPT"
    elif det_ratio >= 0.25:
        tier = "REVIEW"
        reasons.append(f"detected_ratio in [0.25, 0.60) ({det_ratio:.3f})")
    else:
        tier = "REJECT"
        reasons.append(f"low detected_hand_ratio < 0.25 ({det_ratio:.3f})")

    if untruncated_ratio < 0.50:
        reasons.append(f"untruncated_ratio < 0.50 ({untruncated_ratio:.3f})")
    if max_seg > 10.05:
        reasons.append(f"max_segment > 10.0s ({max_seg:.2f}s)")
    if min_seg < 0.49:
        reasons.append(f"min_segment < 0.5s ({min_seg:.2f}s)")

    if tier == "ACCEPT" and not reasons:
        verdict = "ACCEPT"
        reasons_str = "High quality (det >= 0.60, segments [0.5, 10]s, untruncated >= 0.50)"
    elif tier == "REVIEW":
        verdict = "REVIEW"
        reasons_str = "; ".join(reasons)
    else:
        verdict = "REJECT" if tier == "REJECT" else f"REVIEW ({tier})"
        reasons_str = "; ".join(reasons)

    return {
        "id": ep_id,
        "device_profile": device_profile,
        "detected_hand_ratio": det_ratio,
        "rescued_ratio": rescued_ratio,
        "interpolated_ratio": interp_ratio,
        "left_ratio": left_ratio,
        "right_ratio": right_ratio,
        "untruncated_ratio": untruncated_ratio,
        "tier": tier,
        "verdict": verdict,
        "reasons": reasons_str,
    }


def main():
    episodes = [
        "client_video",
        "my_video",
        "new_video",
        "test_10s",
        "whatsapp_video",
        "whatsapp_video_063602",
        "whatsapp_video_063622",
        "whatsapp_video_063639",
        "whatsapp_video_063658",
    ]

    rows = [compute_episode_row(ep) for ep in episodes]

    # Print markdown table
    headers = [
        "Episode ID", "Device Profile", "Detected Ratio", "Rescued Ratio",
        "Interp Ratio", "Left Ratio", "Right Ratio", "Untruncated Ratio",
        "Verdict", "Reasons"
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        print(f"| `{r['id']}` | `{r['device_profile']}` | {r['detected_hand_ratio']:.4f} | {r['rescued_ratio']:.4f} | {r['interpolated_ratio']:.4f} | {r['left_ratio']:.4f} | {r['right_ratio']:.4f} | {r['untruncated_ratio']:.4f} | **{r['verdict']}** | {r['reasons']} |")


if __name__ == "__main__":
    main()
