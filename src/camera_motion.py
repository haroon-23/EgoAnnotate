"""Per-episode camera ego-motion estimation via background sparse optical flow."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import cv2
import numpy as np


@dataclass
class CameraMotionResult:
    mean_bg_speed_px_per_sec: float
    max_bg_speed_px_per_sec: float
    camera_motion_flag: bool
    threshold_px_per_sec: float
    frames_evaluated: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_bg_speed_px_per_sec": round(self.mean_bg_speed_px_per_sec, 3),
            "max_bg_speed_px_per_sec": round(self.max_bg_speed_px_per_sec, 3),
            "camera_motion_flag": bool(self.camera_motion_flag),
            "threshold_px_per_sec": round(self.threshold_px_per_sec, 3),
            "frames_evaluated": int(self.frames_evaluated),
        }


def estimate_camera_motion(
    image_paths: List[str],
    frames: List[Any],
    fps: float = 30.0,
    threshold_px_per_sec: float = 18.0,
) -> CameraMotionResult:
    """Estimate background camera motion by masking out regions near tracked hands."""
    if len(image_paths) < 2 or fps <= 0:
        return CameraMotionResult(0.0, 0.0, False, threshold_px_per_sec, 0)

    speeds: List[float] = []
    prev_gray = None
    step = max(1, len(image_paths) // 60)  # bound runtime on long clips

    for idx in range(0, len(image_paths), step):
        img = cv2.imread(image_paths[idx])
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        if prev_gray is not None and prev_gray.shape == gray.shape:
            mask = np.full((h, w), 255, dtype=np.uint8)
            if idx < len(frames):
                fr = frames[idx]
                for hand in (getattr(fr, "left_hand", None), getattr(fr, "right_hand", None)):
                    if hand is not None:
                        x_min = max(0, int(( float(np.min(hand.x)) - 0.08 ) * w))
                        x_max = min(w, int(( float(np.max(hand.x)) + 0.08 ) * w))
                        y_min = max(0, int(( float(np.min(hand.y)) - 0.08 ) * h))
                        y_max = min(h, int(( float(np.max(hand.y)) + 0.08 ) * h))
                        mask[y_min:y_max, x_min:x_max] = 0

            pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=80, qualityLevel=0.01, minDistance=7, mask=mask)
            if pts is not None and len(pts) >= 6:
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
                if next_pts is not None and status is not None:
                    valid = status.ravel() == 1
                    if np.any(valid):
                        disp = np.linalg.norm((next_pts[valid] - pts[valid]).reshape(-1, 2), axis=1)
                        # Normalize to 224px reference width so threshold is resolution-independent
                        scale = 224.0 / max(float(w), 1.0)
                        median_disp = float(np.median(disp)) * scale
                        dt = float(step) / float(fps)
                        speeds.append(median_disp / max(dt, 1e-4))
        prev_gray = gray

    if not speeds:
        return CameraMotionResult(0.0, 0.0, False, threshold_px_per_sec, 0)

    mean_spd = float(np.mean(speeds))
    max_spd = float(np.max(speeds))
    return CameraMotionResult(
        mean_bg_speed_px_per_sec=mean_spd,
        max_bg_speed_px_per_sec=max_spd,
        camera_motion_flag=bool(mean_spd >= threshold_px_per_sec),
        threshold_px_per_sec=threshold_px_per_sec,
        frames_evaluated=len(speeds),
    )
