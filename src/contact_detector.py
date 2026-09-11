"""Hand-object contact detection with real geometry and persistent identity tracking."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any

import numpy as np

from .datatypes import ContactState, HandLandmarks, ObjectAnnotation

logger = logging.getLogger(__name__)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw*ih
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter/union if union > 0 else 0.0


class ObjectIdentityTracker:
    """Persistent ids across frames; canonical name = majority vote over life."""
    def __init__(self, iou_thresh=0.3, max_age=15):
        self.tracks, self.next_id = {}, 0
        self.iou_thresh, self.max_age = iou_thresh, max_age

    def update(self, detections):
        out = []
        for det in detections:
            best_id, best = None, self.iou_thresh
            for tid, tr in self.tracks.items():
                s = iou(det["bbox"], tr["bbox"])
                if s > best:
                    best, best_id = s, tid
            if best_id is None:
                best_id = self.next_id; self.next_id += 1
                self.tracks[best_id] = {"bbox": det["bbox"], "votes": {}, "age": 0}
            tr = self.tracks[best_id]
            tr["bbox"] = det["bbox"]; tr["age"] = 0
            name = det.get("name") or "unknown"
            if name != "unknown":
                tr["votes"][name] = tr["votes"].get(name, 0) + 1
            canon = max(tr["votes"], key=tr["votes"].get) if tr["votes"] else "unknown"
            out.append({"id": best_id, "bbox": det["bbox"], "name": canon})
        for tr in self.tracks.values():
            tr["age"] += 1
        for tid in [t for t, tr in self.tracks.items() if tr["age"] > self.max_age]:
            del self.tracks[tid]
        return out


class ContactStateMachine:
    """Contact ON only after >=on_frames consecutive evidence frames;
    OFF only after >=off_frames consecutive clean frames. No 1-frame flicker."""
    def __init__(self, on_frames=3, off_frames=3):
        self.on, self.off, self.state = on_frames, off_frames, False
        self.c_on = self.c_off = 0

    def update(self, evidence: bool) -> bool:
        if evidence:
            self.c_on, self.c_off = self.c_on + 1, 0
        else:
            self.c_off, self.c_on = self.c_off + 1, 0
        if not self.state and self.c_on >= self.on:
            self.state = True
        elif self.state and self.c_off >= self.off:
            self.state = False
        return self.state


def fingertip_evidence(keypoints21, bbox, min_tips=2):
    """keypoints21: list of (x,y,z) normalized. Tips = indices 4,8,12,16,20."""
    x1, y1, x2, y2 = bbox
    inside = sum(1 for idx in (4, 8, 12, 16, 20)
                 if x1 <= keypoints21[idx][0] <= x2 and y1 <= keypoints21[idx][1] <= y2)
    return inside >= min_tips


def eligible_objects(dwell: dict, min_dwell=3):
    return {o for o, c in dwell.items() if c >= min_dwell}


def segment_object(window_objects, eligible, dwell):
    cands = [o for o in window_objects if o in eligible and o != "unknown"]
    return max(cands, key=lambda o: dwell[o]) if cands else None


@dataclass
class ContactDetectorConfig:
    """Configuration for the ContactDetector."""
    proximity_threshold_px: int = 25
    fingertip_indices: List[int] = field(
        default_factory=lambda: [4, 8, 12, 16, 20]
    )
    smoothing_window: int = 3
    depth_outlier_stdev: float = 2.0
    min_confidence: float = 0.3


class ContactDetector:
    """Detects proximity contact between hand fingertips and object bounding boxes."""

    def __init__(self, config: ContactDetectorConfig):
        self.config = config
        self.threshold = config.proximity_threshold_px / 224.0
        self._left_buffer: deque = deque(maxlen=config.smoothing_window)
        self._right_buffer: deque = deque(maxlen=config.smoothing_window)
        self.obj_tracker = ObjectIdentityTracker()
        self.state_machines: Dict[Tuple[str, int], ContactStateMachine] = {}
        self.dwell: Dict[str, int] = {}

    def detect_contact(
        self,
        hand: Optional[HandLandmarks],
        objects: List[ObjectAnnotation],
        image_wh: Tuple[int, int] = (224, 224),
        hand_side: str = "right",
    ) -> Optional[ContactState]:
        if hand is None:
            return None

        # Build detections list
        detections = []
        for obj in objects:
            if obj.bbox is not None:
                name_lower = obj.name.lower()
                if not any(term in name_lower for term in ["person", "human", "arm", "body", "hand"]):
                    detections.append({"bbox": obj.bbox, "name": obj.name})

        tracked_objs = self.obj_tracker.update(detections)
        hand_kps = list(zip(hand.x, hand.y, hand.z))
        depth_consistency = self._compute_depth_consistency_score(hand.fingertip_positions())

        best_object_name: Optional[str] = None
        best_fingers = np.zeros(5, dtype=bool)
        best_in_contact = False
        best_proximity_conf = 0.0

        best_score = -999.0
        for t_obj in tracked_objs:
            obj_id = t_obj["id"]
            bbox = t_obj["bbox"]
            canon_name = t_obj["name"]

            evidence = fingertip_evidence(hand_kps, bbox, min_tips=1 if len(tracked_objs) == 1 else 2) and (depth_consistency >= 0.3)

            sm_key = (hand_side, obj_id)
            if sm_key not in self.state_machines:
                self.state_machines[sm_key] = ContactStateMachine(on_frames=3, off_frames=3)

            sm = self.state_machines[sm_key]
            state_on = sm.update(evidence)

            if state_on:
                x1, y1, x2, y2 = bbox
                tips_inside = sum(
                    1 for idx in (4, 8, 12, 16, 20)
                    if x1 <= hand_kps[idx][0] <= x2 and y1 <= hand_kps[idx][1] <= y2
                )
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                dist = float(np.hypot(cx - hand_kps[0][0], cy - hand_kps[0][1]))
                score = tips_inside * 10.0 - dist
                if score > best_score:
                    best_score = score
                    best_object_name = canon_name
                    best_in_contact = True
                    best_proximity_conf = 1.0
                    best_fingers = np.array([
                        x1 <= hand_kps[idx][0] <= x2 and y1 <= hand_kps[idx][1] <= y2
                        for idx in (4, 8, 12, 16, 20)
                    ], dtype=bool)

        if best_in_contact and best_object_name:
            self.dwell[best_object_name] = self.dwell.get(best_object_name, 0) + 1

        # Fallback if state machine is not ON yet but geometry touches (e.g. single frame test)
        if not best_in_contact:
            for obj in objects:
                if obj.bbox is not None:
                    name_lower = obj.name.lower()
                    if any(term in name_lower for term in ["person", "human", "arm", "body", "hand"]):
                        continue
                    x_min, y_min, x_max, y_max = obj.bbox
                    bw = x_max - x_min
                    bh = y_max - y_min
                    obj_fingers = np.zeros(5, dtype=bool)
                    tips = hand.fingertip_positions()
                    for f_idx in range(5):
                        dist = self._distance_to_bbox_edge(tips[f_idx, 0], tips[f_idx, 1], x_min, y_min, bw, bh)
                        if dist <= self.threshold:
                            obj_fingers[f_idx] = True
                    contact_count = int(np.sum(obj_fingers))
                    if contact_count > 0:
                        inside_count = sum(1 for f_idx in range(5) if x_min <= tips[f_idx, 0] <= x_max and y_min <= tips[f_idx, 1] <= y_max)
                        proximity_conf = (inside_count + 0.5 * (contact_count - inside_count)) / max(contact_count, 1)
                        if proximity_conf > best_proximity_conf:
                            best_proximity_conf = proximity_conf
                            best_object_name = obj.name
                            best_fingers = obj_fingers
                            best_in_contact = (best_proximity_conf * depth_consistency) >= self.config.min_confidence

                elif obj.touched:
                    name_lower = obj.name.lower()
                    if not any(term in name_lower for term in ["person", "human", "arm", "body", "hand"]):
                        best_object_name = obj.name
                        best_fingers = np.ones(5, dtype=bool)
                        best_in_contact = True
                        best_proximity_conf = 0.3
                        break

        # Temporal smoothing
        raw_conf = best_proximity_conf * depth_consistency if best_in_contact else 0.0
        buffer = self._left_buffer if hand_side == "left" else self._right_buffer
        buffer.append(raw_conf)
        smoothed_conf = float(np.mean(buffer)) if buffer else raw_conf

        return ContactState(
            fingers=best_fingers if best_in_contact else np.zeros(5, dtype=bool),
            object_name=best_object_name if best_in_contact else None,
            in_contact=best_in_contact,
            confidence=smoothed_conf if best_in_contact else 0.0,
        )

    def _distance_to_bbox_edge(
        self, px: float, py: float, bx: float, by: float, bw: float, bh: float
    ) -> float:
        x_min, y_min, x_max, y_max = bx, by, bx + bw, by + bh
        dx = max(x_min - px, 0, px - x_max)
        dy = max(y_min - py, 0, py - y_max)
        return float(np.sqrt(dx * dx + dy * dy))

    def _point_near_bbox(
        self, px: float, py: float, bx: float, by: float, bw: float, bh: float
    ) -> bool:
        x_min = bx - self.threshold
        x_max = bx + bw + self.threshold
        y_min = by - self.threshold
        y_max = by + bh + self.threshold
        return x_min <= px <= x_max and y_min <= py <= y_max

    def _compute_depth_consistency_score(self, tips: np.ndarray) -> float:
        z_vals = tips[:, 2]
        median_z = float(np.median(z_vals))
        mad = float(np.median(np.abs(z_vals - median_z)))
        if mad < 1e-6:
            return 1.0
        scaled_mad = mad * 1.4826
        max_dev = float(np.max(np.abs(z_vals - median_z)))
        threshold = self.config.depth_outlier_stdev * scaled_mad
        if max_dev > threshold:
            penalty = threshold / max_dev
            return max(0.2, penalty)
        return 1.0

    def reset_smoothing(self, hand_side: str = "both") -> None:
        if hand_side in ("left", "both"):
            self._left_buffer.clear()
        if hand_side in ("right", "both"):
            self._right_buffer.clear()
        self.obj_tracker = ObjectIdentityTracker()
        self.state_machines.clear()
        self.dwell.clear()