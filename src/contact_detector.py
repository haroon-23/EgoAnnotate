"""Hand-object contact detection with real geometry.

Determines contact state between tracked hands and detected objects based on spatial proximity
of fingertips to object bounding boxes, with depth consistency heuristic and temporal smoothing.

MediaPipe z is relative depth, not metric 3D. The depth check is a heuristic to filter
false positives when a hand passes in front of an object without touching. It is NOT
ground truth 3D contact detection.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .datatypes import ContactState, HandLandmarks, ObjectAnnotation

logger = logging.getLogger(__name__)


@dataclass
class ContactDetectorConfig:
    """Configuration for the ContactDetector."""
    proximity_threshold_px: int = 25
    fingertip_indices: List[int] = field(
        default_factory=lambda: [4, 8, 12, 16, 20]
    )
    # Temporal smoothing
    smoothing_window: int = 3
    # Depth consistency (MediaPipe z is relative, not metric)
    depth_outlier_stdev: float = 2.0  # Fingertip z > mean + N*std = outlier
    # Minimum confidence to consider "in contact"
    min_confidence: float = 0.3


class ContactDetector:
    """Detects proximity contact between hand fingertips and object bounding boxes.

    Uses real bbox geometry from Grounding DINO + depth consistency heuristic +
    temporal smoothing to produce a confidence score [0, 1].
    """

    def __init__(self, config: ContactDetectorConfig):
        """Initialise the ContactDetector."""
        self.config = config
        # Convert pixel threshold to normalized coordinates assuming 224x224
        self.threshold = config.proximity_threshold_px / 224.0

        # Temporal smoothing buffers: one per hand
        self._left_buffer: deque = deque(maxlen=config.smoothing_window)
        self._right_buffer: deque = deque(maxlen=config.smoothing_window)

    def detect_contact(
        self,
        hand: Optional[HandLandmarks],
        objects: List[ObjectAnnotation],
        image_wh: Tuple[int, int] = (224, 224),
        hand_side: str = "right",  # "left" or "right"
    ) -> Optional[ContactState]:
        """Detect contact between fingertips of a single hand and any visible objects.

        Args:
            hand: HandLandmarks for the active hand, or None.
            objects: List of ObjectAnnotations in the frame.
            image_wh: Width and height of the image frame.
            hand_side: "left" or "right" — used for temporal smoothing buffer selection.

        Returns:
            ContactState with fingers, object_name, in_contact, and confidence [0, 1],
            or None if hand is None.
        """
        if hand is None:
            return None

        # Get 3D fingertip positions (shape (5, 3): x, y, z)
        tips = hand.fingertip_positions()

        # --- 1. Proximity checking using REAL bboxes (primary path) ---
        best_object_name: Optional[str] = None
        best_fingers = np.zeros(5, dtype=bool)
        best_proximity_conf = 0.0
        best_bbox = None

        for obj in objects:
            if obj.bbox is None:
                continue  # Skip objects without real bbox

            # bbox is [x_min, y_min, x_max, y_max] in normalized coordinates
            x_min, y_min, x_max, y_max = obj.bbox
            bw = x_max - x_min
            bh = y_max - y_min

            obj_fingers = np.zeros(5, dtype=bool)
            finger_distances = np.full(5, np.inf)  # Track min distance to bbox edge

            for f_idx in range(5):
                px = tips[f_idx, 0]
                py = tips[f_idx, 1]

                # Distance from point to bbox edge (negative = inside)
                dist = self._distance_to_bbox_edge(px, py, x_min, y_min, bw, bh)
                finger_distances[f_idx] = dist

                if dist <= self.threshold:
                    obj_fingers[f_idx] = True

            contact_count = int(np.sum(obj_fingers))
            if contact_count > 0:
                # Compute proximity confidence based on how deep inside the expanded bbox
                # Fingers inside the original bbox get higher confidence
                inside_count = 0
                for f_idx in range(5):
                    px, py = tips[f_idx, 0], tips[f_idx, 1]
                    if x_min <= px <= x_max and y_min <= py <= y_max:
                        inside_count += 1

                # Proximity confidence: ratio of fingers inside to total fingers in contact
                # Weighted by how many fingers are near the object
                proximity_conf = (inside_count + 0.5 * (contact_count - inside_count)) / max(contact_count, 1)

                if proximity_conf > best_proximity_conf:
                    best_proximity_conf = proximity_conf
                    best_object_name = obj.name
                    best_fingers = obj_fingers
                    best_bbox = obj.bbox

        # --- 2. Depth consistency heuristic (MediaPipe z is relative, NOT metric 3D) ---
        depth_consistency = self._compute_depth_consistency_score(tips)

        # --- 3. Fallback: if no bbox match but Gemini said touched ---
        used_fallback = False
        if best_proximity_conf == 0.0:
            for obj in objects:
                if obj.bbox is None and obj.touched:
                    # Low confidence fallback: all fingers, no geometry
                    best_object_name = obj.name
                    best_fingers = np.ones(5, dtype=bool)
                    best_proximity_conf = 0.3  # Low base confidence
                    depth_consistency = 0.5  # Unknown depth
                    used_fallback = True
                    break

        # --- 4. Combine proximity + depth ---
        raw_confidence = best_proximity_conf * depth_consistency
        # Fallback always counts as contact (even if low confidence)
        in_contact = raw_confidence >= self.config.min_confidence or used_fallback

        # --- 5. Temporal smoothing ---
        buffer = self._left_buffer if hand_side == "left" else self._right_buffer
        buffer.append(raw_confidence)
        smoothed_confidence = float(np.mean(buffer)) if buffer else raw_confidence

        # Final in_contact after smoothing - fallback overrides smoothing
        final_in_contact = smoothed_confidence >= self.config.min_confidence or used_fallback
        if not final_in_contact:
            best_fingers = np.zeros(5, dtype=bool)
            best_object_name = None

        return ContactState(
            fingers=best_fingers,
            object_name=best_object_name,
            in_contact=final_in_contact,
            confidence=smoothed_confidence,
        )

    def _distance_to_bbox_edge(
        self, px: float, py: float, bx: float, by: float, bw: float, bh: float
    ) -> float:
        """Distance from point to nearest bbox edge. Returns 0.0 when point is inside bbox."""
        x_min, y_min, x_max, y_max = bx, by, bx + bw, by + bh

        # Distance to each edge
        dx = max(x_min - px, 0, px - x_max)
        dy = max(y_min - py, 0, py - y_max)

        return np.sqrt(dx * dx + dy * dy)

    def _point_near_bbox(
        self, px: float, py: float, bx: float, by: float, bw: float, bh: float
    ) -> bool:
        """Check if a point is within the bounding box expanded by the proximity threshold."""
        x_min = bx - self.threshold
        x_max = bx + bw + self.threshold
        y_min = by - self.threshold
        y_max = by + bh + self.threshold

        return x_min <= px <= x_max and y_min <= py <= y_max

    def _compute_depth_consistency_score(self, tips: np.ndarray) -> float:
        """Compute depth consistency score from fingertip z values.

        MediaPipe z is RELATIVE depth (smaller = closer to camera), not metric 3D.
        This is a HEURISTIC to filter false positives when a hand passes in front
        of an object without touching.

        If the spread of fingertip depths is too large, it's likely a false positive
        (the hand is passing in front of, not touching, the object).

        Args:
            tips: (5, 3) array of fingertip positions [x, y, z]

        Returns:
            Score [0, 1] where 1 = all tips aligned in depth, 0 = wild spread.
        """
        z_vals = tips[:, 2]  # Relative depth for 5 fingertips

        # Use median and MAD (median absolute deviation) for robustness
        median_z = float(np.median(z_vals))
        mad = float(np.median(np.abs(z_vals - median_z)))

        if mad < 1e-6:
            return 1.0  # All tips at same depth

        # MAD is scaled to be comparable to std for normal distribution
        # MAD * 1.4826 ≈ std for normal distribution
        scaled_mad = mad * 1.4826

        # Check if any fingertip deviates too far from median
        max_dev = float(np.max(np.abs(z_vals - median_z)))

        # Threshold: N * scaled_MAD
        threshold = self.config.depth_outlier_stdev * scaled_mad

        if max_dev > threshold:
            # Linear penalty
            penalty = threshold / max_dev
            return max(0.2, penalty)

        return 1.0

    def reset_smoothing(self, hand_side: str = "both") -> None:
        """Clear temporal smoothing buffers."""
        if hand_side in ("left", "both"):
            self._left_buffer.clear()
        if hand_side in ("right", "both"):
            self._right_buffer.clear()