"""Grasp type classification from hand landmarks.

Classifies the type of grasp (precision pinch, power wrap, hook, open, or unknown)
based on hand geometric features such as finger curling and thumb-index distance.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .datatypes import GraspType, HandLandmarks, AnnotationFrame


class TemporalGraspVoter:
    """Majority vote over a 5-frame window inside contiguous tracked blocks.
    'unknown' inside a hold inherits the nearest confident label (flagged)."""
    def __init__(self, window=5):
        self.w = window

    def smooth(self, labels, present):
        n = len(labels)
        out = [None]*n
        i = 0
        while i < n:
            if not present[i]:
                out[i] = (None, False); i += 1; continue
            j = i
            while j < n and present[j]:
                j += 1
            for k in range(i, j):
                lo, hi = max(i, k-self.w//2), min(j, k+self.w//2+1)
                votes = Counter(l for l in labels[lo:hi]
                                if l not in (None, "unknown"))
                if votes:
                    out[k] = (votes.most_common(1)[0][0], False)
                elif labels[k] not in (None, "unknown"):
                    out[k] = (labels[k], False)
                else:
                    inherit = next((labels[m] for m in range(k, j)
                                    if labels[m] not in (None, "unknown")),
                                   next((labels[m] for m in range(k-1, i-1, -1)
                                         if labels[m] not in (None, "unknown")), None))
                    out[k] = (inherit, inherit is not None)
            i = j
        return out


@dataclass
class GraspClassifierConfig:
    """Configuration for the GraspClassifier."""
    pinch_threshold: float = 0.05
    wrap_threshold: float = 0.15
    pip_indices: List[int] = field(
        default_factory=lambda: [6, 10, 14, 18]
    )
    tip_indices: List[int] = field(
        default_factory=lambda: [8, 12, 16, 20]
    )


class GraspClassifier:
    """Classifies grasp types based on hand skeletal geometry."""

    def __init__(self, config: GraspClassifierConfig):
        """Initialise the GraspClassifier with config."""
        self.config = config

    def smooth_sequence(self, frames: List[AnnotationFrame]) -> None:
        """Apply TemporalGraspVoter smoothing across all frames in an episode."""
        voter = TemporalGraspVoter(window=5)
        for handedness in ["left", "right"]:
            labels = [
                getattr(f, f"{handedness}_grasp").type
                if getattr(f, f"{handedness}_grasp") is not None else None
                for f in frames
            ]
            present = [getattr(f, f"{handedness}_hand") is not None for f in frames]
            results = voter.smooth(labels, present)
            for f, (label, inherited) in zip(frames, results):
                g = getattr(f, f"{handedness}_grasp")
                if g is not None and label is not None:
                    g.type = label

    def classify(self, hand: Optional[HandLandmarks]) -> Optional[GraspType]:
        """Classify the grasp type for the given hand.

        Args:
            hand: HandLandmarks for the hand, or None.

        Returns:
            GraspType object containing the classification results, or None if hand is None.
        """
        if hand is None:
            return None

        # 1. Compute normalized distance between thumb tip (landmark 4) and index tip (landmark 8)
        p4 = np.array([hand.x[4], hand.y[4], hand.z[4]])
        p8 = np.array([hand.x[8], hand.y[8], hand.z[8]])
        dist = float(np.linalg.norm(p4 - p8))

        # 2. Count curled fingers (index, middle, ring, pinky)
        num_curled = self._count_curled_fingers(hand)

        # 3. Classify grasp type
        if dist < self.config.pinch_threshold:
            grasp_type_str = "precision_pinch"
        elif num_curled >= 3 and dist > self.config.wrap_threshold:
            grasp_type_str = "power_wrap"
        elif num_curled >= 2 and dist > self.config.wrap_threshold * 0.7:
            # Reconciles "hook or power_wrap" rule: if curled is 2, it is a hook.
            # If curled is >=3 but distance is smaller than wrap_threshold, it's also categorized as hook.
            grasp_type_str = "hook"
        elif num_curled == 0:
            grasp_type_str = "open"
        else:
            grasp_type_str = "unknown"

        return GraspType(
            type=grasp_type_str,
            confidence=0.7,
            thumb_index_distance=dist,
            num_curled_fingers=num_curled,
        )

    def _count_curled_fingers(self, hand: HandLandmarks) -> int:
        """Count the number of non-thumb fingers that are curled.

        A finger is considered curled if its tip landmark is closer to the wrist (landmark 0)
        than its corresponding PIP joint landmark.
        """
        w = np.array([hand.x[0], hand.y[0], hand.z[0]])
        curled_count = 0

        # Loop over the 4 fingers: index, middle, ring, pinky
        for pip_idx, tip_idx in zip(self.config.pip_indices, self.config.tip_indices):
            pip_pt = np.array([hand.x[pip_idx], hand.y[pip_idx], hand.z[pip_idx]])
            tip_pt = np.array([hand.x[tip_idx], hand.y[tip_idx], hand.z[tip_idx]])

            d_pip = np.linalg.norm(pip_pt - w)
            d_tip = np.linalg.norm(tip_pt - w)

            if d_tip < d_pip:
                curled_count += 1

        return curled_count
