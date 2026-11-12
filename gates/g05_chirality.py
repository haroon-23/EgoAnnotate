#!/usr/bin/env python3
"""Gate 05: Chirality accuracy against human-labelled ground-truth CSV & mid-clip identity stability."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np


def _match_hand_side_in_frame(frame_dict: Dict[str, Any], target_wx: float) -> str:
    best_side = "MISSING"
    best_dist = 999.0
    for key, side_name in [("left_hand", "Left"), ("right_hand", "Right")]:
        h = frame_dict.get(key)
        if h and h.get("landmarks"):
            wx = float(h["landmarks"][0]["x"])
            d = abs(wx - target_wx)
            if d < best_dist:
                best_dist = d
                best_side = h.get("handedness", side_name)
    return best_side if best_dist <= 0.12 else "UNMATCHED"


def _count_identity_swaps(frames: List[Dict[str, Any]]) -> int:
    swaps = 0
    prev_l_x, prev_r_x = None, None
    for f in frames:
        lh, rh = f.get("left_hand"), f.get("right_hand")
        lx = float(lh["landmarks"][0]["x"]) if (lh and lh.get("landmarks")) else None
        rx = float(rh["landmarks"][0]["x"]) if (rh and rh.get("landmarks")) else None
        if lx is not None and rx is not None and prev_l_x is not None and prev_r_x is not None:
            # If Left and Right jumped directly to each other's previous coordinates in 1 frame
            d_keep = abs(lx - prev_l_x) + abs(rx - prev_r_x)
            d_swap = abs(lx - prev_r_x) + abs(rx - prev_l_x)
            if d_swap + 0.08 < d_keep:
                swaps += 1
        if lx is not None:
            prev_l_x = lx
        if rx is not None:
            prev_r_x = rx
    return swaps


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate 05: Chirality & Persistent Identity Verification")
    parser.add_argument("episode_dir", type=str, help="Path to episode output dir")
    parser.add_argument("--labels-csv", type=str, default="data/chirality_labels.csv")
    parser.add_argument("--min-labels", type=int, default=60)
    parser.add_argument("--min-accuracy", type=float, default=0.95)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    print("=== G05: CHIRALITY ACCURACY & PERSISTENT IDENTITY GATE ===")
    ep_dir = Path(args.episode_dir)
    ann_path = ep_dir / "annotations.json"
    if not ann_path.exists():
        print(f"  [FAIL] Missing {ann_path}")
        return 1

    ann = json.loads(ann_path.read_text(encoding="utf-8"))
    frames = ann.get("frames", [])
    frame_map = {int(f["frame_idx"]): f for f in frames}
    ep_id = ann.get("metadata", {}).get("episode_id", ep_dir.name)

    swaps = _count_identity_swaps(frames)
    print(f"  Mid-clip instantaneous Left/Right coordinate swaps in `{ep_id}`: {swaps}")

    csv_path = Path(args.labels_csv)
    failures: List[str] = []
    if swaps > 0:
        failures.append(f"Detected {swaps} mid-clip Left/Right hand identity swaps")

    left_correct, left_total = 0, 0
    right_correct, right_total = 0, 0

    if not csv_path.exists():
        failures.append(
            f"Human-labelled chirality CSV not found at {csv_path}. "
            f"Run `tools/label_frames.py` on >= {args.min_labels} frames first (STATUS: NOT VERIFIED)."
        )
        total_rows = 0
    else:
        with open(csv_path, "r", encoding="utf-8") as f_in:
            rows = list(csv.DictReader(f_in))
        total_rows = len(rows)
        print(f"  Loaded {total_rows} human-labelled hand instances from {csv_path} (required >= {args.min_labels})")
        if total_rows < args.min_labels:
            failures.append(f"Insufficient human-labelled frames: {total_rows} < {args.min_labels}")

        # Load any referenced clip annotations
        clip_cache: Dict[str, Dict[int, Any]] = {ep_id: frame_map}
        for r in rows:
            cid = r["clip_id"]
            if cid not in clip_cache:
                cand_ann = ep_dir.parent / cid / "annotations.json"
                if cand_ann.exists():
                    c_data = json.loads(cand_ann.read_text(encoding="utf-8"))
                    clip_cache[cid] = {int(fr["frame_idx"]): fr for fr in c_data.get("frames", [])}
                else:
                    clip_cache[cid] = {}

            f_dict = clip_cache.get(cid, {}).get(int(r["frame_idx"]))
            true_side = r["true_side_of_wearer_hand_in_image"].strip().capitalize()
            wx = float(r["wrist_x_norm"])
            pred_side = _match_hand_side_in_frame(f_dict, wx) if f_dict else r.get("pipeline_label", "MISSING")

            if true_side == "Left":
                left_total += 1
                if pred_side == "Left":
                    left_correct += 1
            elif true_side == "Right":
                right_total += 1
                if pred_side == "Right":
                    right_correct += 1

    left_acc = (left_correct / left_total) if left_total > 0 else 0.0
    right_acc = (right_correct / right_total) if right_total > 0 else 0.0
    overall_acc = ((left_correct + right_correct) / (left_total + right_total)) if (left_total + right_total) > 0 else 0.0

    print(f"  Left Hand Accuracy : {left_correct}/{left_total} ({left_acc * 100.0:.1f}%)")
    print(f"  Right Hand Accuracy: {right_correct}/{right_total} ({right_acc * 100.0:.1f}%)")
    print(f"  Overall Accuracy   : {left_correct + right_correct}/{left_total + right_total} ({overall_acc * 100.0:.1f}%)")

    if (left_total + right_total) >= args.min_labels and overall_acc < args.min_accuracy:
        failures.append(f"Chirality accuracy {overall_acc:.3f} < required {args.min_accuracy:.3f}")

    passed = len(failures) == 0
    for f_msg in failures:
        print(f"  [FAIL] {f_msg}")
    print(f"G05 VERDICT: {'PASS' if passed else 'FAIL'}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "gate": "g05_chirality",
            "status": "PASS" if passed else "FAIL",
            "total_human_labels": total_rows,
            "left_accuracy": round(left_acc, 4),
            "right_accuracy": round(right_acc, 4),
            "overall_accuracy": round(overall_acc, 4),
            "mid_clip_identity_swaps": swaps,
            "failures": failures,
        }, indent=2) + "\n", encoding="utf-8")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
