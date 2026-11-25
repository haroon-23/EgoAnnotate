"""Manipulated-object segmentation & persistent tracking (CPU Apache-2.0 + GPU SAM 2.1 / Grounding DINO)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from src.egoannotate.registry import register_backend
from src.egoannotate.schemas import (
    CameraIntrinsicsModel,
    HandState,
    ManipulatedObjectState,
)


def _bbox_iou(a: list[float], b: list[float]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 1e-6 else 0.0


@dataclass
class _TrackedObj:
    track_id: int
    label: str
    bbox_xyxy: list[float]
    mask_area_frac: float
    pos_cam_m: np.ndarray
    last_seen_frame: int


class OpenCVManipulatedObjectTracker:
    """CPU-feasible Apache-2.0 hand-proximity ROI object detector, mask extractor & tracker."""

    def __init__(self, intrinsics: CameraIntrinsicsModel) -> None:
        self.intrinsics = intrinsics
        self._prev_gray: np.ndarray | None = None
        self._tracks: dict[int, _TrackedObj] = {}
        self._next_id = 1

    def _extract_object_near_hand(
        self,
        bgr: np.ndarray,
        gray: np.ndarray,
        hand: HandState,
        side: str,
    ) -> tuple[list[float], float, float, np.ndarray] | None:
        if not hand.detected or len(hand.landmarks_2d_px) < 21:
            return None
        h, w = bgr.shape[:2]
        pts = np.asarray(hand.landmarks_2d_px, dtype=np.float64)
        # Focus ROI around fingertips (thumb 4, index 8, middle 12) where manipulated object sits
        tip_center = np.mean(pts[[4, 8, 12]], axis=0)
        hand_span = max(float(np.linalg.norm(pts[0] - pts[9])) * 1.4, 48.0)
        rad = int(np.clip(hand_span * 1.35, 56.0, min(w, h) * 0.35))
        x0 = max(0, int(tip_center[0] - rad))
        y0 = max(0, int(tip_center[1] - rad))
        x1 = min(w, int(tip_center[0] + rad))
        y1 = min(h, int(tip_center[1] + rad))
        if (x1 - x0) < 24 or (y1 - y0) < 24:
            return None

        roi_gray = gray[y0:y1, x0:x1]
        # Mask out hand convex hull inside ROI so contours capture the object-in-hand rather than the skin itself
        hand_mask: np.ndarray = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        rel_pts = np.round(pts - np.array([x0, y0])).astype(np.int32)
        cv2.fillConvexPoly(hand_mask, cv2.convexHull(rel_pts), (255.0,))
        hand_mask = cv2.dilate(hand_mask, np.ones((7, 7), np.uint8), iterations=1)

        edges = cv2.Canny(roi_gray, 50, 140)
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            diff = cv2.absdiff(roi_gray, self._prev_gray[y0:y1, x0:x1])
            _, motion_mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
            fg = cv2.bitwise_or(edges, motion_mask)
        else:
            fg = edges

        fg = cv2.bitwise_and(fg, cv2.bitwise_not(hand_mask))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_cnt = None
        best_area = 0.0
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < 180.0:
                continue
            if area > best_area:
                best_area = area
                best_cnt = cnt

        if best_cnt is None:
            return None

        bx, by, bw, bh = cv2.boundingRect(best_cnt)
        bbox = [float(x0 + bx), float(y0 + by), float(x0 + bx + bw), float(y0 + by + bh)]
        mask_frac = float(np.clip(best_area / max(float(w * h), 1.0), 0.0, 1.0))

        # Project object centroid into camera frame using active hand depth
        cx_obj = 0.5 * (bbox[0] + bbox[2])
        cy_obj = 0.5 * (bbox[1] + bbox[3])
        z_m = float(hand.wrist_pos_cam_m[2]) if hand.wrist_pos_cam_m[2] > 0.1 else 0.45
        obj_cam = np.array(
            [
                (cx_obj - self.intrinsics.cx) * z_m / self.intrinsics.fx,
                (cy_obj - self.intrinsics.cy) * z_m / self.intrinsics.fy,
                z_m,
            ],
            dtype=np.float64,
        )
        # Confidence scales with edge/motion support and hand proximity
        score = float(np.clip(0.45 + 0.45 * min(1.0, best_area / 3000.0), 0.45, 0.92))
        _ = side
        return bbox, mask_frac, score, obj_cam

    def _match_or_create_track(
        self,
        frame_idx: int,
        bbox: list[float],
        mask_frac: float,
        obj_cam: np.ndarray,
    ) -> _TrackedObj:
        best_tid: int | None = None
        best_score = 0.0
        for tid, tr in self._tracks.items():
            if frame_idx - tr.last_seen_frame > 30:
                continue
            iou = _bbox_iou(tr.bbox_xyxy, bbox)
            c_old = np.array([0.5 * (tr.bbox_xyxy[0] + tr.bbox_xyxy[2]), 0.5 * (tr.bbox_xyxy[1] + tr.bbox_xyxy[3])])
            c_new = np.array([0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])])
            dist_score = max(0.0, 1.0 - float(np.linalg.norm(c_old - c_new)) / max(float(self.intrinsics.width) * 0.25, 1.0))
            sim = 0.65 * iou + 0.35 * dist_score
            if sim > best_score:
                best_score = sim
                best_tid = tid

        if best_tid is not None and best_score >= 0.25:
            tr = self._tracks[best_tid]
            tr.bbox_xyxy = bbox
            tr.mask_area_frac = mask_frac
            tr.pos_cam_m = 0.5 * tr.pos_cam_m + 0.5 * obj_cam
            tr.last_seen_frame = frame_idx
            return tr

        tid = self._next_id
        self._next_id += 1
        tr = _TrackedObj(
            track_id=tid,
            label=f"manipulated_object_{tid}",
            bbox_xyxy=bbox,
            mask_area_frac=mask_frac,
            pos_cam_m=obj_cam,
            last_seen_frame=frame_idx,
        )
        self._tracks[tid] = tr
        return tr

    def process_frame(
        self,
        bgr: np.ndarray,
        frame_idx: int,
        left_hand: HandState,
        right_hand: HandState,
    ) -> list[ManipulatedObjectState]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        results_by_tid: dict[int, ManipulatedObjectState] = {}

        for side, hand in (("left", left_hand), ("right", right_hand)):
            det = self._extract_object_near_hand(bgr, gray, hand, side)
            if det is None:
                continue
            bbox, mask_frac, score, obj_cam = det
            tr = self._match_or_create_track(frame_idx, bbox, mask_frac, obj_cam)
            closed_hand = float(np.mean(hand.finger_curls_5d)) > 0.32 or hand.gripper_aperture_m < 0.08
            in_hand_flag = side if closed_hand else "none"
            if tr.track_id in results_by_tid:
                existing = results_by_tid[tr.track_id]
                if existing.in_hand != "none" and in_hand_flag != "none" and existing.in_hand != in_hand_flag:
                    existing.in_hand = "both"
            else:
                obj_state = "held" if in_hand_flag != "none" else "approached"
                results_by_tid[tr.track_id] = ManipulatedObjectState(
                    track_id=tr.track_id,
                    label=tr.label,
                    score=round(score, 4),
                    bbox_xyxy=[round(v, 2) for v in bbox],
                    mask_area_frac=round(mask_frac, 5),
                    pos_world_m=[round(float(c), 5) for c in tr.pos_cam_m],
                    in_hand=in_hand_flag,  # type: ignore[arg-type]
                    object_state=obj_state,
                    state_confidence=round(score * (0.9 if closed_hand else 0.7), 4),
                )

        # Prune stale tracks to keep memory O(active objects) on multi-hour videos
        stale_ids = [tid for tid, tr in self._tracks.items() if frame_idx - tr.last_seen_frame > 90]
        for tid in stale_ids:
            del self._tracks[tid]

        self._prev_gray = gray
        return list(results_by_tid.values())


@register_backend(
    stage="object_detector",
    name="opencv_motion_roi",
    code_license="Apache-2.0",
    weights_license="N/A",
    requires_gpu=False,
    notes="CPU-safe hand-proximity motion/contour object tracker.",
)
def build_opencv_motion_roi(intrinsics: CameraIntrinsicsModel) -> OpenCVManipulatedObjectTracker:
    return OpenCVManipulatedObjectTracker(intrinsics=intrinsics)


@register_backend(
    stage="object_segmenter",
    name="opencv_roi_mask",
    code_license="Apache-2.0",
    weights_license="N/A",
    requires_gpu=False,
    notes="CPU-safe hand-excluded foreground contour mask extractor.",
)
def build_opencv_roi_mask(intrinsics: CameraIntrinsicsModel) -> OpenCVManipulatedObjectTracker:
    return OpenCVManipulatedObjectTracker(intrinsics=intrinsics)


@register_backend(
    stage="object_detector",
    name="grounding_dino_apache",
    code_license="Apache-2.0",
    weights_license="Apache-2.0",
    requires_gpu=True,
    notes="GPU production open-vocabulary detector (IDEA-Research/grounding-dino-base).",
)
def build_grounding_dino_apache(**_kwargs: Any) -> Any:
    try:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("transformers >= 4.40 required for grounding_dino_apache") from exc
    return {
        "model_id": "IDEA-Research/grounding-dino-base",
        "processor_cls": AutoProcessor,
        "model_cls": AutoModelForZeroShotObjectDetection,
        "license": "Apache-2.0",
    }


@register_backend(
    stage="object_segmenter",
    name="sam21_apache",
    code_license="Apache-2.0",
    weights_license="Apache-2.0",
    requires_gpu=True,
    notes="GPU production video object mask propagation (Meta SAM 2.1, Apache-2.0).",
)
def build_sam21_apache(**_kwargs: Any) -> Any:
    return {
        "model_id": "facebook/sam2.1-hiera-small",
        "license": "Apache-2.0",
        "note": "Invoke via SAM2VideoPredictor on CUDA GPU node.",
    }


@register_backend(
    stage="object_detector",
    name="florence2_mit",
    code_license="MIT",
    weights_license="MIT",
    requires_gpu=False,
    notes="Compact MIT-licensed open-vocab detector/captioner (microsoft/Florence-2-base).",
)
def build_florence2_mit(**_kwargs: Any) -> Any:
    return {
        "model_id": "microsoft/Florence-2-base",
        "license": "MIT",
    }


def _unused_math_guard() -> float:
    return math.pi
