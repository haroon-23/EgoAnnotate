#!/usr/bin/env python3
"""Interactive OpenCV tool for human ground-truth labelling of wearer hand chirality (Left vs Right).

Displays sampled video frames with a crosshair on the detected wrist, allowing the user to press:
  [L] -> This highlighted wrist is the wearer's LEFT hand
  [R] -> This highlighted wrist is the wearer's RIGHT hand
  [S] -> Skip / ambiguous frame
  [Q] -> Save and quit

Writes CSV with columns:
  clip_id,frame_idx,true_side_of_wearer_hand_in_image,wrist_x_norm,wrist_y_norm,pipeline_label
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np


def collect_candidate_instances(
    video_path: Path,
    annotations_path: Path,
    max_samples: int = 60,
) -> List[Dict[str, Any]]:
    ann = json.loads(annotations_path.read_text(encoding="utf-8"))
    frames = ann.get("frames", [])
    clip_id = ann.get("metadata", {}).get("episode_id", video_path.stem)

    instances: List[Dict[str, Any]] = []
    for f in frames:
        f_idx = int(f["frame_idx"])
        ts = float(f.get("timestamp", 0.0))
        for side_key, default_lbl in [("left_hand", "Left"), ("right_hand", "Right")]:
            h = f.get(side_key)
            if h and not h.get("is_interpolated", False):
                lms = h.get("landmarks", [])
                if lms:
                    wx, wy = float(lms[0]["x"]), float(lms[0]["y"])
                    instances.append({
                        "clip_id": clip_id,
                        "video_path": str(video_path),
                        "frame_idx": f_idx,
                        "timestamp": ts,
                        "wrist_x_norm": round(wx, 4),
                        "wrist_y_norm": round(wy, 4),
                        "landmarks": lms,
                        "pipeline_label": h.get("handedness", default_lbl),
                    })

    if len(instances) > max_samples:
        idxs = np.linspace(0, len(instances) - 1, max_samples, dtype=int)
        instances = [instances[i] for i in idxs]
    return instances


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive Human Hand Chirality Labelling Tool")
    parser.add_argument("--video", type=str, required=True, help="Path to raw .mp4 file")
    parser.add_argument("--annotations", type=str, required=True, help="Path to episode annotations.json")
    parser.add_argument("--out", type=str, default="data/chirality_labels.csv", help="Output CSV path")
    parser.add_argument("--num-samples", type=int, default=60, help="Number of hand instances to sample")
    args = parser.parse_args()

    video_path = Path(args.video)
    ann_path = Path(args.annotations)
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    instances = collect_candidate_instances(video_path, ann_path, max_samples=args.num_samples)
    if not instances:
        print("No valid un-interpolated hand instances found in annotations.")
        return 1

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return 1
    orig_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)

    existing_rows = []
    if out_csv.exists():
        with open(out_csv, "r", encoding="utf-8") as f_in:
            existing_rows = list(csv.DictReader(f_in))

    new_rows = []
    print(f"Loaded {len(instances)} candidate hand instances from {video_path.name}.")
    print("Controls in OpenCV window: [L] = Wearer's LEFT hand | [R] = Wearer's RIGHT hand | [S] = Skip | [Q] = Quit & Save")

    for idx, inst in enumerate(instances):
        raw_f = max(0, min(total_raw - 1, int(round(inst["timestamp"] * orig_fps))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, raw_f)
        ret, bgr = cap.read()
        if not ret or bgr is None:
            continue

        h, w = bgr.shape[:2]
        canvas = bgr.copy()
        for lm in inst["landmarks"]:
            px, py = int(round(lm["x"] * w)), int(round(lm["y"] * h))
            cv2.circle(canvas, (px, py), 4, (0, 255, 255), -1)

        wx_px = int(round(inst["wrist_x_norm"] * w))
        wy_px = int(round(inst["wrist_y_norm"] * h))
        cv2.circle(canvas, (wx_px, wy_px), 16, (0, 0, 255), 3)
        cv2.putText(
            canvas,
            f"[{idx+1}/{len(instances)}] f={inst['frame_idx']} | Press L (Left), R (Right), S (Skip), Q (Quit)",
            (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
        )
        cv2.imshow("Chirality Ground Truth Labeler", canvas)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        elif key in (ord("l"), ord("L")):
            true_side = "Left"
        elif key in (ord("r"), ord("R")):
            true_side = "Right"
        else:
            continue

        new_rows.append({
            "clip_id": inst["clip_id"],
            "frame_idx": inst["frame_idx"],
            "true_side_of_wearer_hand_in_image": true_side,
            "wrist_x_norm": inst["wrist_x_norm"],
            "wrist_y_norm": inst["wrist_y_norm"],
            "pipeline_label": inst["pipeline_label"],
        })

    cap.release()
    cv2.destroyAllWindows()

    all_rows = existing_rows + new_rows
    fieldnames = [
        "clip_id", "frame_idx", "true_side_of_wearer_hand_in_image",
        "wrist_x_norm", "wrist_y_norm", "pipeline_label",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved {len(new_rows)} new labels ({len(all_rows)} total) to {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
