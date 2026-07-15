"""Hand-object contact detection.

Determines contact state between tracked hands and detected objects based on spatial proximity
of fingertips to object bounding boxes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .datatypes import ContactState, HandLandmarks, ObjectAnnotation


@dataclass
class ContactDetectorConfig:
    """Configuration for the ContactDetector."""
    proximity_threshold_px: int = 25
    fingertip_indices: List[int] = field(
        default_factory=lambda: [4, 8, 12, 16, 20]
    )


class ContactDetector:
    """Detects proximity contact between hand fingertips and object bounding boxes."""

    def __init__(self, config: ContactDetectorConfig):
        """Initialise the ContactDetector and normalise the threshold."""
        self.config = config
        # Convert pixel threshold to normalized coordinates assuming a default image size of 224x224
        self.threshold = config.proximity_threshold_px / 224.0

    def detect_contact(
        self,
        hand: Optional[HandLandmarks],
        objects: List[ObjectAnnotation],
        image_wh: Tuple[int, int] = (224, 224),
    ) -> Optional[ContactState]:
        """Detect contact between fingertips of a single hand and any visible objects.

        Args:
            hand: HandLandmarks for the active hand, or None.
            objects: List of ObjectAnnotations in the frame.
            image_wh: Width and height of the image frame.

        Returns:
            ContactState indicating which fingers are in contact with an object, or None if hand is None.
        """
        if hand is None:
            return None

        # Get 3D fingertip positions (shape (5, 3))
        tips = hand.fingertip_positions()

        best_object_name: Optional[str] = None
        best_fingers = np.zeros(5, dtype=bool)
        best_contact_count = 0

        # 1. Proximity checking for objects with bounding boxes
        for obj in objects:
            if obj.bbox is not None:
                # bbox is [x_min, y_min, x_max, y_max] in normalised coordinates
                x_min, y_min, x_max, y_max = obj.bbox
                bw = x_max - x_min
                bh = y_max - y_min

                obj_fingers = np.zeros(5, dtype=bool)
                for f_idx in range(5):
                    px = tips[f_idx, 0]
                    py = tips[f_idx, 1]

                    if self._point_near_bbox(px, py, x_min, y_min, bw, bh):
                        obj_fingers[f_idx] = True

                contact_count = int(np.sum(obj_fingers))
                if contact_count > 0:
                    # Prefer the object with the most fingers in contact
                    if contact_count > best_contact_count:
                        best_object_name = obj.name
                        best_fingers = obj_fingers
                        best_contact_count = contact_count

        # 2. Fallback for objects without a bounding box but marked as touched (e.g. from VLM)
        if best_contact_count == 0:
            for obj in objects:
                if obj.bbox is None and obj.touched:
                    best_object_name = obj.name
                    # Mark all fingers as potential contact
                    best_fingers = np.ones(5, dtype=bool)
                    best_contact_count = 5
                    break

        # If contact was found, return active ContactState; otherwise, return a clean non-contact state
        if best_contact_count > 0:
            return ContactState(
                fingers=best_fingers,
                object_name=best_object_name,
                in_contact=True,
            )
        else:
            return ContactState(
                fingers=np.zeros(5, dtype=bool),
                object_name=None,
                in_contact=False,
            )

    def _point_near_bbox(
        self, px: float, py: float, bx: float, by: float, bw: float, bh: float
    ) -> bool:
        """Check if a point is within the bounding box expanded by the proximity threshold.

        All coordinates (px, py, bx, by, bw, bh) and the threshold are in normalised [0, 1] space.
        """
        # Expand bbox in all directions by the normalised threshold
        x_min = bx - self.threshold
        x_max = bx + bw + self.threshold
        y_min = by - self.threshold
        y_max = by + bh + self.threshold

        # Check inclusion of the point
        return x_min <= px <= x_max and y_min <= py <= y_max
