"""Apache-2.0 GPU RTMW (rtmlib) 133-kpt whole-body & gloved-hand tracker with MediaPipe fallback."""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import torch

from src.egoannotate.config import PerceptionConfig
from src.egoannotate.perception_hands import (
    MediaPipeHandTracker,
    MediaPipeUpperBodyTracker,
    OneEuroArrayFilter,
    compute_finger_curls_5d,
    compute_hand_rotation_cam,
    rotation_to_quat_and_rot6d,
)
from src.egoannotate.schemas import (
    CameraIntrinsicsModel,
    FrameAnnotation,
    HandState,
    UpperBodyState,
)

COCO_WHOLEBODY_LEFT_HAND_SLICE = slice(91, 112)
COCO_WHOLEBODY_RIGHT_HAND_SLICE = slice(112, 133)
FINGERTIP_IDXS = (4, 8, 12, 16, 20)


def regularize_hand_bone_lengths(
    lm_3d: np.ndarray,
    assumed_palm_width_m: float = 0.085,
) -> np.ndarray:
    """Enforce anatomically plausible palm scale and finger segment proportions."""
    pts = np.asarray(lm_3d, dtype=np.float32).copy()
    if pts.shape != (21, 3) or not np.all(np.isfinite(pts)):
        return pts
    wrist = pts[0].copy()
    palm_vec = pts[17] - pts[5]
    palm_w = float(np.linalg.norm(palm_vec))
    if palm_w > 1e-4:
        target_w = float(np.clip(palm_w, 0.72 * assumed_palm_width_m, 1.28 * assumed_palm_width_m))
        scale = target_w / palm_w
        pts = wrist[None, :] + (pts - wrist[None, :]) * scale
    return pts.astype(np.float32)


def interpolate_gloved_hand_dropouts(
    frames: list[FrameAnnotation],
    max_gap_frames: int = 18,
) -> int:
    """Temporally bridge short hand occlusion/glove drop-outs via linear kinematics interpolation."""
    filled = 0
    n = len(frames)
    if n < 3:
        return 0
    for side in ("left", "right"):
        detected_idxs = [
            i
            for i, fr in enumerate(frames)
            if (fr.left_hand if side == "left" else fr.right_hand).detected
        ]
        for a, b in itertools.pairwise(detected_idxs):
            gap = b - a - 1
            if gap <= 0 or gap > max_gap_frames:
                continue
            ha = frames[a].left_hand if side == "left" else frames[a].right_hand
            hb = frames[b].left_hand if side == "left" else frames[b].right_hand
            if len(ha.landmarks_2d_px) != 21 or len(hb.landmarks_2d_px) != 21:
                continue
            lm2_a = np.asarray(ha.landmarks_2d_px, dtype=np.float32)
            lm2_b = np.asarray(hb.landmarks_2d_px, dtype=np.float32)
            lm3c_a = np.asarray(ha.landmarks_3d_cam_m, dtype=np.float32)
            lm3c_b = np.asarray(hb.landmarks_3d_cam_m, dtype=np.float32)
            lm3w_a = np.asarray(ha.landmarks_3d_world_m, dtype=np.float32)
            lm3w_b = np.asarray(hb.landmarks_3d_world_m, dtype=np.float32)
            wp_a = np.asarray(ha.wrist_pos_world_m, dtype=np.float32)
            wp_b = np.asarray(hb.wrist_pos_world_m, dtype=np.float32)
            wc_a = np.asarray(ha.wrist_pos_cam_m, dtype=np.float32)
            wc_b = np.asarray(hb.wrist_pos_cam_m, dtype=np.float32)
            curls_a = np.asarray(ha.finger_curls_5d, dtype=np.float32)
            curls_b = np.asarray(hb.finger_curls_5d, dtype=np.float32)

            for k in range(1, gap + 1):
                alpha = float(k) / float(gap + 1)
                idx = a + k
                target = frames[idx].left_hand if side == "left" else frames[idx].right_hand
                target.detected = True
                target.confidence = round(0.75 * min(ha.confidence, hb.confidence), 4)
                target.visibility = round(0.75 * min(ha.visibility, hb.visibility), 4)
                target.landmarks_2d_px = ((1.0 - alpha) * lm2_a + alpha * lm2_b).tolist()
                if lm3c_a.shape == (21, 3) and lm3c_b.shape == (21, 3):
                    target.landmarks_3d_cam_m = ((1.0 - alpha) * lm3c_a + alpha * lm3c_b).tolist()
                if lm3w_a.shape == (21, 3) and lm3w_b.shape == (21, 3):
                    interp_w = (1.0 - alpha) * lm3w_a + alpha * lm3w_b
                    target.landmarks_3d_world_m = interp_w.tolist()
                    target.fingertips_world_m = [interp_w[fi].tolist() for fi in FINGERTIP_IDXS]
                target.wrist_pos_cam_m = ((1.0 - alpha) * wc_a + alpha * wc_b).tolist()
                target.wrist_pos_world_m = ((1.0 - alpha) * wp_a + alpha * wp_b).tolist()
                target.wrist_quat_xyzw_world = list(ha.wrist_quat_xyzw_world)
                target.wrist_rot6d_world = list(ha.wrist_rot6d_world)
                target.gripper_aperture_m = float(
                    (1.0 - alpha) * ha.gripper_aperture_m + alpha * hb.gripper_aperture_m
                )
                if curls_a.size == 5 and curls_b.size == 5:
                    target.finger_curls_5d = ((1.0 - alpha) * curls_a + alpha * curls_b).tolist()
                filled += 1
    return filled


class RTMWWholeBodyHandTracker:
    """GPU ONNXRuntime/TensorRT RTMW 133-kpt whole-body + gloved hand tracker."""

    def __init__(
        self,
        cfg: PerceptionConfig,
        intrinsics: CameraIntrinsicsModel,
        assumed_palm_width_m: float = 0.085,
        device: str = "cuda",
    ) -> None:
        self.cfg = cfg
        self.intrinsics = intrinsics
        self.assumed_palm_width_m = float(assumed_palm_width_m)
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.backend_used = "rtmw_133_apache"
        self._rtm_wholebody: Any | None = None
        self._mp_hands: MediaPipeHandTracker | None = None
        self._mp_body: MediaPipeUpperBodyTracker | None = None
        self._filters = {
            "left": OneEuroArrayFilter(cfg.one_euro_min_cutoff, cfg.one_euro_beta),
            "right": OneEuroArrayFilter(cfg.one_euro_min_cutoff, cfg.one_euro_beta),
        }

        try:
            from rtmlib import Wholebody

            self._rtm_wholebody = Wholebody(
                to_openpose=False,
                mode="balanced",
                backend="onnxruntime",
                device=self.device,
            )
        except (ImportError, RuntimeError, OSError, ValueError):
            self.backend_used = "mediapipe_fallback"
            self._mp_hands = MediaPipeHandTracker(
                cfg,
                intrinsics,
                assumed_palm_width_m,
            )
            self._mp_body = MediaPipeUpperBodyTracker() if cfg.enable_upper_body else None

    def close(self) -> None:
        if self._mp_hands is not None:
            self._mp_hands.close()
        if self._mp_body is not None:
            self._mp_body.close()

    def _lift_hand_2d_to_state(
        self,
        kpts_21: np.ndarray,
        scores_21: np.ndarray,
        side: str,
        timestamp_s: float,
    ) -> HandState:
        mean_conf = float(np.mean(scores_21))
        if mean_conf < float(self.cfg.hand_min_detection_conf):
            return HandState(handedness="left" if side == "left" else "right", detected=False)

        palm_px = float(np.linalg.norm(kpts_21[17] - kpts_21[5]))
        fx = max(float(self.intrinsics.fx), 1.0)
        fy = max(float(self.intrinsics.fy), 1.0)
        cx = float(self.intrinsics.cx)
        cy = float(self.intrinsics.cy)
        z_est = float(
            np.clip(
                (fx * self.assumed_palm_width_m) / max(palm_px, 12.0),
                0.22,
                1.60,
            )
        )
        xs = (kpts_21[:, 0] - cx) * z_est / fx
        ys = (kpts_21[:, 1] - cy) * z_est / fy
        zs = np.full(21, z_est, dtype=np.float32)
        lm_3d = np.stack([xs, ys, zs], axis=1).astype(np.float32)
        lm_3d = regularize_hand_bone_lengths(lm_3d, self.assumed_palm_width_m)

        if self.cfg.smoothing_method == "one_euro":
            lm_3d = self._filters[side].filter(lm_3d, timestamp_s)

        rot_cam = compute_hand_rotation_cam(lm_3d, side)
        quat, rot6d = rotation_to_quat_and_rot6d(rot_cam)
        curls = compute_finger_curls_5d(lm_3d)
        ap_m = float(np.linalg.norm(lm_3d[4] - lm_3d[8]))
        in_frame = (
            (kpts_21[:, 0] >= 0)
            & (kpts_21[:, 0] < self.intrinsics.width)
            & (kpts_21[:, 1] >= 0)
            & (kpts_21[:, 1] < self.intrinsics.height)
        )
        vis = float(np.mean(in_frame & (scores_21 >= 0.25)))
        return HandState(
            handedness="left" if side == "left" else "right",
            detected=True,
            confidence=round(min(1.0, max(0.0, mean_conf)), 4),
            visibility=round(min(1.0, max(0.0, vis)), 4),
            truncated=bool(vis < 0.85),
            landmarks_2d_px=kpts_21.astype(np.float32).tolist(),
            landmarks_3d_cam_m=lm_3d.tolist(),
            landmarks_3d_world_m=lm_3d.tolist(),
            wrist_pos_cam_m=lm_3d[0].tolist(),
            wrist_pos_world_m=lm_3d[0].tolist(),
            wrist_quat_xyzw_world=quat,
            wrist_rot6d_world=rot6d,
            fingertips_world_m=[lm_3d[fi].tolist() for fi in FINGERTIP_IDXS],
            gripper_aperture_m=round(float(np.clip(ap_m, 0.0, 0.18)), 5),
            finger_curls_5d=curls,
        )

    def _build_upper_body(self, kpts: np.ndarray, scores: np.ndarray) -> UpperBodyState:
        idxs = (5, 6, 7, 8, 9, 10)
        if kpts.shape[0] <= 10 or float(np.mean(scores[list(idxs)])) < 0.25:
            return UpperBodyState(detected=False)
        fx = max(float(self.intrinsics.fx), 1.0)
        fy = max(float(self.intrinsics.fy), 1.0)
        cx = float(self.intrinsics.cx)
        cy = float(self.intrinsics.cy)
        z_nom = 0.65

        def _pt(i: int) -> list[float]:
            return [
                round(float((kpts[i, 0] - cx) * z_nom / fx), 4),
                round(float((kpts[i, 1] - cy) * z_nom / fy), 4),
                z_nom,
            ]

        ls, rs = _pt(5), _pt(6)
        pitch = math.degrees(math.atan2(ls[1] + rs[1], 2.0 * z_nom))
        return UpperBodyState(
            detected=True,
            confidence=round(float(np.mean(scores[list(idxs)])), 4),
            left_shoulder_cam=ls,
            right_shoulder_cam=rs,
            left_elbow_cam=_pt(7),
            right_elbow_cam=_pt(8),
            left_wrist_cam=_pt(9),
            right_wrist_cam=_pt(10),
            torso_pitch_deg=round(pitch, 2),
        )

    def process_frame(
        self,
        bgr: np.ndarray,
        timestamp_s: float,
    ) -> tuple[HandState, HandState, UpperBodyState | None]:
        if self._rtm_wholebody is not None:
            kpts, scores = self._rtm_wholebody(bgr)
            if kpts is None or len(kpts) == 0:
                return (
                    HandState(handedness="left", detected=False),
                    HandState(handedness="right", detected=False),
                    UpperBodyState(detected=False),
                )
            p_kpts = np.asarray(kpts[0], dtype=np.float32)
            p_scores = np.asarray(scores[0], dtype=np.float32)
            if p_kpts.shape[0] >= 133:
                lh = self._lift_hand_2d_to_state(
                    p_kpts[COCO_WHOLEBODY_LEFT_HAND_SLICE],
                    p_scores[COCO_WHOLEBODY_LEFT_HAND_SLICE],
                    "left",
                    timestamp_s,
                )
                rh = self._lift_hand_2d_to_state(
                    p_kpts[COCO_WHOLEBODY_RIGHT_HAND_SLICE],
                    p_scores[COCO_WHOLEBODY_RIGHT_HAND_SLICE],
                    "right",
                    timestamp_s,
                )
                body = (
                    self._build_upper_body(p_kpts, p_scores)
                    if self.cfg.enable_upper_body
                    else None
                )
                return lh, rh, body

        assert self._mp_hands is not None
        lh, rh = self._mp_hands.process_frame(bgr, timestamp_s)
        body = self._mp_body.process_frame(bgr) if self._mp_body is not None else None
        return lh, rh, body
