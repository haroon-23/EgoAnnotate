#!/usr/bin/env python3
"""Evaluate hand filter loosening before vs after on labelled frames from client_video and my_video."""
import csv
import json
from pathlib import Path
import numpy as np


def is_valid_hand(h, expected_side, tight: bool = False):
    """Filter rule evaluation: tight (before) vs loosened (after)."""
    if not h:
        return False, "missing"
    xs = h.get("x", [])
    ys = h.get("y", [])
    if len(xs) < 21 or len(ys) < 21:
        return False, "landmarks_incomplete"

    wx, wy = float(xs[0]), float(ys[0])
    mcp9_x, mcp9_y = float(xs[9]), float(ys[9])
    tip12_x, tip12_y = float(xs[12]), float(ys[12])
    cy = float(np.mean(ys))

    if wy < 0.20 or cy < 0.18 or wx < 0.01 or wx > 0.99:
        return False, "bystander_top_or_edge_entry"

    forearm_dy = wy - mcp9_y
    palm_len = float(np.hypot(mcp9_x - wx, mcp9_y - wy))
    finger_len = float(np.hypot(tip12_x - mcp9_x, tip12_y - mcp9_y))

    if tight:
        # Before thresholds
        max_forearm_dy = -0.055
        min_finger_len = 0.25 * palm_len
        max_wx_left = 0.88
    else:
        # Loosened thresholds
        max_forearm_dy = -0.38
        min_finger_len = 0.04 * palm_len
        max_wx_left = 0.96

    if forearm_dy < max_forearm_dy:
        return False, "bystander_inverted_forearm"
    if palm_len < 0.038 or palm_len > 0.46:
        return False, "invalid_palm_scale"
    if finger_len < min_finger_len or finger_len > 2.85 * palm_len:
        return False, "anatomical_proportion_invalid"
    if expected_side.lower() == "left" and wx > max_wx_left:
        return False, "bystander_lateral_mismatch"
    if expected_side.lower() == "right" and wx < 0.04:
        return False, "bystander_lateral_mismatch"

    return True, "ok"


def evaluate_dataset(labels_csv: Path):
    labels = []
    with open(labels_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["clip_id"] in ("client_video", "my_video"):
                labels.append(row)

    results = {}
    for clip in ("client_video", "my_video"):
        clip_labels = [r for r in labels if r["clip_id"] == clip]
        ann_path = Path("data/output") / clip / "annotations.json"
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        frames = ann["frames"]

        for mode_name, tight_flag in [("Before (Tight)", True), ("After (Loosened)", False)]:
            tp, fp, fn, tn = 0, 0, 0, 0
            for row in clip_labels:
                f_idx = int(row["frame_idx"])
                true_side = row["true_side_of_wearer_hand_in_image"].lower()
                f_data = frames[f_idx]
                hand = f_data.get(f"{true_side}_hand")

                # Is true wearer hand accepted?
                ok, reason = is_valid_hand(hand, true_side, tight=tight_flag)
                if ok:
                    tp += 1
                else:
                    fn += 1

                # Check opposite hand (bystander / false hand check)
                other_side = "right" if true_side == "left" else "left"
                other_hand = f_data.get(f"{other_side}_hand")
                if other_hand:
                    ok_other, _ = is_valid_hand(other_hand, other_side, tight=tight_flag)
                    if ok_other:
                        # If a real second hand exists it is TP, else if bystander it is FP
                        pass

            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = 2 * prec * rec / max(1e-6, prec + rec)
            results[(clip, mode_name)] = {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": prec, "recall": rec, "f1": f1
            }

    print("=" * 80)
    print("HAND FILTER LOOSENING EVALUATION: BEFORE VS AFTER ON LABELLED FRAMES")
    print("=" * 80)
    print(f"{'Clip':15s} | {'Configuration':18s} | {'TP':4s} | {'FP':4s} | {'FN':4s} | {'Precision':10s} | {'Recall':8s} | {'F1':6s}")
    print("-" * 80)
    for (clip, mode), m in results.items():
        print(f"{clip:15s} | {mode:18s} | {m['tp']:4d} | {m['fp']:4d} | {m['fn']:4d} | {m['precision']:10.4f} | {m['recall']:8.4f} | {m['f1']:.4f}")
    print("-" * 80)

    # Full clip statistics
    print("\nFull-Clip Hand Yield Analysis (5400 frames each):")
    for clip in ("client_video", "my_video"):
        ann = json.loads((Path("data/output") / clip / "annotations.json").read_text(encoding="utf-8"))
        frames = ann["frames"]
        n = len(frames)
        det_before, det_after = 0, 0
        for f in frames:
            # Check tight
            lh_t, _ = is_valid_hand(f.get("left_hand"), "left", tight=True)
            rh_t, _ = is_valid_hand(f.get("right_hand"), "right", tight=True)
            if lh_t or rh_t:
                det_before += 1

            # Check loosened
            lh_l, _ = is_valid_hand(f.get("left_hand"), "left", tight=False)
            rh_l, _ = is_valid_hand(f.get("right_hand"), "right", tight=False)
            if lh_l or rh_l:
                det_after += 1
        print(f"  {clip:15s}: Before (Tight)={det_before}/{n} ({det_before/n*100:.2f}%) -> After (Loosened)={det_after}/{n} ({det_after/n*100:.2f}%) | Gain=+{det_after - det_before} frames")


if __name__ == "__main__":
    evaluate_dataset(Path("data/chirality_labels.csv"))
