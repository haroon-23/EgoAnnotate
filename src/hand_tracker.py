"""MediaPipe hand landmark detection and tracking with temporal smoothing."""

from __future__ import annotations

import logging
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from .datatypes import HandLandmarks

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    try:
        from mediapipe.tasks.python import vision
    except (ImportError, AttributeError):
        class _DummyHandLandmarker:
            @classmethod
            def create_from_options(cls, *args, **kwargs):
                pass
        class _DummyVision:
            HandLandmarker = _DummyHandLandmarker
        vision = _DummyVision
except ImportError:
    mp = None
    class _DummyHandLandmarker:
        @classmethod
        def create_from_options(cls, *args, **kwargs):
            pass
    class _DummyVision:
        HandLandmarker = _DummyHandLandmarker
    vision = _DummyVision


@dataclass
class HandTrackerConfig:
    """Configuration for the HandTracker."""
    model_path: str = "models/hand_landmarker.task"
    running_mode: str = "VIDEO"  # "VIDEO" or "IMAGE"
    num_hands: int = 2
    min_detection_confidence: float = 0.3
    min_tracking_confidence: float = 0.3
    smoothing_window: int = 5
    max_gap_frames: int = 5
    max_gap_distance: float = 0.15


class HandTracker:
    """Pass 1: temporal tracking. Pass 2: static-image rescue on lost frames.
    Pass 3: <=5-frame gap interpolation. All losses measured and reported."""

    # MediaPipe hand landmark connections for drawing the skeleton
    _HAND_CONNECTIONS = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle
        (9, 10), (10, 11), (11, 12),
        # Ring
        (13, 14), (14, 15), (15, 16),
        # Pinky
        (17, 18), (18, 19), (19, 20),
        # Palm base
        (5, 9), (9, 13), (13, 17), (0, 17)
    ]

    def __init__(self, config: Optional[Any] = None, detection_conf: float = 0.3, tracking_conf: float = 0.3):
        if config is not None:
            if hasattr(config, "min_detection_confidence"):
                detection_conf = config.min_detection_confidence
            if hasattr(config, "min_tracking_confidence"):
                tracking_conf = config.min_tracking_confidence
        self.config = config or HandTrackerConfig(
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )

        import mediapipe as mp
        self._mp = mp
        self._video = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf, model_complexity=1)
        self._static = mp.solutions.hands.Hands(
            static_image_mode=True, max_num_hands=2,
            min_detection_confidence=0.25, model_complexity=1)

        self.landmarker = self._video
        self.last_metrics: Dict[str, float] = {}

        # Set up smoothing buffers for Left and Right hands.
        smooth_win = getattr(self.config, "smoothing_window", 5)
        self.buffers: Dict[str, deque] = {
            "Left": deque(maxlen=smooth_win),
            "Right": deque(maxlen=smooth_win),
        }

    @staticmethod
    def _empty(i: int) -> Dict[str, Any]:
        return {"frame_idx": i, "left_present": False, "right_present": False,
                "left_keypoints": [0.0]*63, "right_keypoints": [0.0]*63,
                "left_interpolated": False, "right_interpolated": False,
                "rescued": False}

    def _fill(self, rec: Dict[str, Any], res: Any, only_missing: bool = False) -> None:
        if res is None:
            return
        landmarks = getattr(res, "multi_hand_landmarks", None) or getattr(res, "hand_landmarks", None)
        handedness = getattr(res, "multi_handedness", None) or getattr(res, "handedness", None)
        if not landmarks or not handedness:
            return

        for lm, handed in zip(landmarks, handedness):
            if hasattr(handed, "classification"):
                side = handed.classification[0].label.lower()  # keep existing mirror convention
            elif isinstance(handed, list) and len(handed) > 0 and hasattr(handed[0], "category_name"):
                side = handed[0].category_name.lower()
            else:
                side = "right"

            key = f"{side}_present"
            if only_missing and rec[key]:
                continue
            rec[key] = True

            if hasattr(lm, "landmark"):
                pts = lm.landmark
            elif isinstance(lm, list):
                pts = lm
            else:
                continue

            rec[f"{side}_keypoints"] = [c for p in pts for c in (p.x, p.y, p.z)]

    def _process_frame(self, hands_obj: Any, frame_rgb: np.ndarray) -> Any:
        if hasattr(hands_obj, "process"):
            return hands_obj.process(frame_rgb)
        elif hasattr(hands_obj, "detect"):
            return hands_obj.detect(frame_rgb)
        return None

    def track(self, frames: List[np.ndarray]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        n = len(frames)
        if n == 0:
            metrics = {"lost_pct": 0.0, "rescued_pct": 0.0, "interpolated_pct": 0.0}
            self.last_metrics = metrics
            return [], metrics

        recs = [self._empty(i) for i in range(n)]
        for i, frm in enumerate(frames):                       # Pass 1
            self._fill(recs[i], self._process_frame(self._video,
                cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)))
        rescued = 0
        for i in range(n):                                     # Pass 2
            if recs[i]["left_present"] and recs[i]["right_present"]:
                continue
            before = (recs[i]["left_present"], recs[i]["right_present"])
            self._fill(recs[i], self._process_frame(self._static,
                cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)), only_missing=True)
            if (recs[i]["left_present"], recs[i]["right_present"]) != before:
                recs[i]["rescued"] = True
                rescued += 1
        for side in ("left", "right"):                         # Pass 3
            pres = [r[f"{side}_present"] for r in recs]
            i = 0
            while i < n:
                if not pres[i]:
                    j = i
                    while j < n and not pres[j]:
                        j += 1
                    gap = j - i
                    if 0 < i and j < n and gap <= 5:
                        a, b = recs[i-1], recs[j]
                        for k in range(i, j):
                            t = (k - (i-1)) / (j - (i-1))
                            ka, kb = a[f"{side}_keypoints"], b[f"{side}_keypoints"]
                            recs[k][f"{side}_keypoints"] = [
                                xa + t*(xb-xa) for xa, xb in zip(ka, kb)]
                            recs[k][f"{side}_present"] = True
                            recs[k][f"{side}_interpolated"] = True
                    i = j
                else:
                    i += 1
        detected = [r["left_present"] or r["right_present"] for r in recs]
        active_mask = [False] * n
        for i in range(n):
            if detected[i]:
                for k in range(max(0, i - 10), min(n, i + 11)):
                    active_mask[k] = True

        active_frames = sum(1 for a in active_mask if a)
        lost = sum(1 for r in recs if not r["left_present"] or not r["right_present"])
        lost_active = sum(1 for i, r in enumerate(recs) if active_mask[i] and (not r["left_present"] or not r["right_present"]))

        interp = sum(1 for r in recs if r["left_interpolated"] or r["right_interpolated"])
        metrics = {
            "lost_pct": 100.0 * lost / max(n, 1),
            "lost_pct_active": 100.0 * lost_active / max(active_frames, 1) if active_frames > 0 else 0.0,
            "rescued_pct": 100.0 * rescued / max(n, 1),
            "interpolated_pct": 100.0 * interp / max(n, 1),
        }
        self.last_metrics = metrics
        return recs, metrics

    def track_frames(self, image_paths: List[str]) -> List[Dict[str, Optional[HandLandmarks]]]:
        """Process a sequence of images at native resolution and extract hand landmarks."""
        frames = []
        for p in image_paths:
            img = cv2.imread(p)
            if img is None:
                raise RuntimeError(f"Cannot read frame image: {p}")
            frames.append(img)

        recs, metrics = self.track(frames)
        self.last_metrics = metrics

        results: List[Dict[str, Optional[HandLandmarks]]] = []
        for rec in recs:
            frm_res: Dict[str, Optional[HandLandmarks]] = {"left": None, "right": None}
            for side in ("left", "right"):
                if rec[f"{side}_present"]:
                    kp = np.array(rec[f"{side}_keypoints"], dtype=np.float32).reshape(21, 3)
                    frm_res[side] = HandLandmarks(
                        x=kp[:, 0],
                        y=kp[:, 1],
                        z=kp[:, 2],
                        confidence=0.8,
                        handedness=side.capitalize(),
                        is_interpolated=rec.get(f"{side}_interpolated", False),
                    )
            results.append(frm_res)
        return results

    def draw_landmarks(self, image: np.ndarray, hands: Dict[str, Optional[HandLandmarks]]) -> np.ndarray:
        """Draw hand skeleton overlays onto an image."""
        output = image.copy()
        h, w, _ = output.shape

        colors = {
            "left": (0, 255, 0),     # Green
            "right": (255, 0, 255)   # Purple
        }

        for side, landmarks in hands.items():
            if landmarks is None:
                continue

            color = colors.get(side, (0, 255, 255))
            
            # Convert normalized coords to pixel coords
            pts_x = (landmarks.x * w).astype(np.int32)
            pts_y = (landmarks.y * h).astype(np.int32)

            # Draw connections
            for p1, p2 in self._HAND_CONNECTIONS:
                x1, y1 = pts_x[p1], pts_y[p1]
                x2, y2 = pts_x[p2], pts_y[p2]
                
                # Check bounds
                if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                    cv2.line(output, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Draw joints
            for i in range(21):
                px, py = pts_x[i], pts_y[i]
                if 0 <= px < w and 0 <= py < h:
                    radius = 5 if i in [4, 8, 12, 16, 20] else 3
                    cv2.circle(output, (px, py), radius, color, -1, cv2.LINE_AA)
            
            # Add text label near wrist
            wrist_x, wrist_y = pts_x[0], pts_y[0]
            label = side.upper()
            cv2.putText(
                output, 
                label, 
                (wrist_x - 20, wrist_y + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                color, 
                2, 
                cv2.LINE_AA
            )

        return output
