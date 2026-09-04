"""MediaPipe hand landmark detection and tracking with temporal smoothing."""

from __future__ import annotations

import logging
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .datatypes import HandLandmarks

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError:
    mp = None
    logger.error(
        "MediaPipe is not installed. Hand tracking will fail. "
        "Install via: pip install mediapipe>=0.10.0"
    )


@dataclass
class HandTrackerConfig:
    """Configuration for the HandTracker."""
    model_path: str = "models/hand_landmarker.task"
    running_mode: str = "VIDEO"  # "VIDEO" or "IMAGE"
    num_hands: int = 2
    min_detection_confidence: float = 0.2
    min_tracking_confidence: float = 0.3
    smoothing_window: int = 5
    max_gap_frames: int = 5
    max_gap_distance: float = 0.15


class HandTracker:
    """Detects and tracks hand landmarks across video frames.

    Uses MediaPipe Tasks Vision API (HandLandmarker) in VIDEO mode to ensure
    temporal consistency, followed by a rolling average smoothing filter and
    a gap-interpolation layer to handle brief occlusions.
    """

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

    def __init__(self, config: HandTrackerConfig):
        """Initialise the HandTracker and load the MediaPipe model."""
        if mp is None:
            raise RuntimeError(
                "MediaPipe is required for HandTracker. "
                "Run: pip install mediapipe>=0.10.0"
            )

        self.config = config
        self._ensure_model_exists(config.model_path)

        # Set up MediaPipe HandLandmarker Options
        base_options = mp_python.BaseOptions(model_asset_path=config.model_path)
        
        running_mode = (
            vision.RunningMode.VIDEO
            if config.running_mode == "VIDEO"
            else vision.RunningMode.IMAGE
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_hands=config.num_hands,
            min_hand_detection_confidence=config.min_detection_confidence,
            min_hand_presence_confidence=config.min_tracking_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )

        self.landmarker = vision.HandLandmarker.create_from_options(options)

        # Set up smoothing buffers for Left and Right hands.
        # Each hand has a deque storing the last N raw landmark arrays of shape (21, 3).
        self.buffers: Dict[str, deque] = {
            "Left": deque(maxlen=config.smoothing_window),
            "Right": deque(maxlen=config.smoothing_window),
        }

    def _ensure_model_exists(self, model_path: str) -> None:
        """Download the MediaPipe task model if it does not exist."""
        path = Path(model_path)
        if not path.exists():
            logger.info("Downloading HandLandmarker model to %s...", model_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/"
                "hand_landmarker.task"
            )
            urllib.request.urlretrieve(url, model_path)
            logger.info("Download complete.")

    def track_frames(self, image_paths: List[str]) -> List[Dict[str, Optional[HandLandmarks]]]:
        """Process a sequence of images and extract smoothed hand landmarks.

        Args:
            image_paths: List of absolute paths to image frames.

        Returns:
            List of dictionaries, one per frame, containing ``"left"`` and 
            ``"right"`` keys mapping to :class:`HandLandmarks` or ``None``.
        """
        results: List[Dict[str, Optional[HandLandmarks]]] = []

        # Reset smoothing buffers at the start of a sequence
        self.buffers["Left"].clear()
        self.buffers["Right"].clear()

        for frame_idx, path in enumerate(image_paths):
            frame_result: Dict[str, Optional[HandLandmarks]] = {"left": None, "right": None}

            # Load and convert image
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                logger.warning("Could not read image %s, skipping tracking.", path)
                results.append(frame_result)
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            # Detect landmarks
            if self.config.running_mode == "VIDEO":
                # Assuming 30fps for the timestamp in milliseconds
                timestamp_ms = int(frame_idx * 33.33)
                detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            else:
                detection_result = self.landmarker.detect(mp_image)

            # Keep track of which hands we found in this frame to clear buffers for missing hands
            found_hands = {"Left": False, "Right": False}

            if detection_result and detection_result.hand_landmarks:
                for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                    handedness_list = detection_result.handedness[idx]
                    category = handedness_list[0]
                    handedness_str = category.category_name
                    confidence = category.score

                    if handedness_str not in found_hands:
                        continue
                    
                    found_hands[handedness_str] = True

                    # Extract raw (x, y, z) into arrays
                    raw_x = np.array([lm.x for lm in hand_landmarks], dtype=np.float32)
                    raw_y = np.array([lm.y for lm in hand_landmarks], dtype=np.float32)
                    raw_z = np.array([lm.z for lm in hand_landmarks], dtype=np.float32)
                    
                    raw_pts = np.stack([raw_x, raw_y, raw_z], axis=1) # (21, 3)
                    
                    # Apply smoothing
                    self.buffers[handedness_str].append(raw_pts)
                    smoothed_pts = np.mean(self.buffers[handedness_str], axis=0) # (21, 3)

                    hlm = HandLandmarks(
                        x=smoothed_pts[:, 0],
                        y=smoothed_pts[:, 1],
                        z=smoothed_pts[:, 2],
                        confidence=confidence,
                        handedness=handedness_str,
                        is_interpolated=False,
                    )

                    frame_result[handedness_str.lower()] = hlm

            # Clear buffers for hands that disappeared to prevent dragging old locations
            for hand_name, found in found_hands.items():
                if not found:
                    self.buffers[hand_name].clear()

            results.append(frame_result)

        # Apply gap interpolation for brief tracking losses
        self._interpolate_gaps(results)

        return results

    def _interpolate_gaps(self, results: List[Dict[str, Optional[HandLandmarks]]]) -> None:
        """Linearly interpolate missing hand landmarks for gaps <= max_gap_frames."""
        n_frames = len(results)
        if n_frames < 3:
            return

        for side in ("left", "right"):
            i = 0
            while i < n_frames:
                if results[i][side] is None:
                    start = i
                    while i < n_frames and results[i][side] is None:
                        i += 1
                    end = i - 1
                    gap_len = end - start + 1

                    # Check if gap is eligible for interpolation
                    if (
                        gap_len <= self.config.max_gap_frames
                        and start > 0
                        and end < n_frames - 1
                    ):
                        prev_hlm = results[start - 1][side]
                        next_hlm = results[end + 1][side]

                        if prev_hlm is not None and next_hlm is not None:
                            # Spatial distance check between wrist positions (landmark index 0)
                            dx = float(next_hlm.x[0] - prev_hlm.x[0])
                            dy = float(next_hlm.y[0] - prev_hlm.y[0])
                            wrist_dist = float(np.sqrt(dx * dx + dy * dy))

                            if wrist_dist <= self.config.max_gap_distance:
                                # Interpolate keypoints for all gap frames
                                total_steps = end + 1 - (start - 1)
                                for step_idx, frame_k in enumerate(range(start, end + 1), start=1):
                                    alpha = float(step_idx) / float(total_steps)

                                    interp_x = (1.0 - alpha) * prev_hlm.x + alpha * next_hlm.x
                                    interp_y = (1.0 - alpha) * prev_hlm.y + alpha * next_hlm.y
                                    interp_z = (1.0 - alpha) * prev_hlm.z + alpha * next_hlm.z
                                    interp_conf = float((1.0 - alpha) * prev_hlm.confidence + alpha * next_hlm.confidence)

                                    results[frame_k][side] = HandLandmarks(
                                        x=interp_x,
                                        y=interp_y,
                                        z=interp_z,
                                        confidence=interp_conf,
                                        handedness=prev_hlm.handedness,
                                        is_interpolated=True,
                                    )
                else:
                    i += 1

    def draw_landmarks(self, image: np.ndarray, hands: Dict[str, Optional[HandLandmarks]]) -> np.ndarray:
        """Draw hand skeleton overlays onto an image.

        Args:
            image: BGR image array.
            hands: Dictionary with ``"left"`` and ``"right"`` keys mapped to HandLandmarks.

        Returns:
            BGR image with annotations drawn over it.
        """
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
                    # Fingertips (4, 8, 12, 16, 20) are drawn slightly larger
                    radius = 5 if i in [4, 8, 12, 16, 20] else 3
                    cv2.circle(output, (px, py), radius, color, -1, cv2.LINE_AA)
            
            # Add text label near the wrist (landmark 0)
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
