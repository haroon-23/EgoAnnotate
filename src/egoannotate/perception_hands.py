"""Commercially-licensed hand & upper-body pose extraction (MediaPipe Apache-2.0) + One-Euro smoothing."""

from __future__ import annotations

import math
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation

from src.egoannotate.config import PerceptionConfig
from src.egoannotate.registry import register_backend
from src.egoannotate.schemas import CameraIntrinsicsModel, HandState, UpperBodyState

FINGERTIP_INDICES = (4, 8, 12, 16, 20)
FINGER_CHAINS = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)


class OneEuroArrayFilter:
    """Casiez et al. (2012) One-Euro low-pass filter for jitter-free, low-latency 3D landmarks."""

    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.05, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None
        self._t_prev: float | None = None

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    @staticmethod
    def _alpha(cutoff: float | np.ndarray, dt: float) -> float | np.ndarray:
        tau = 1.0 / (2.0 * math.pi * np.maximum(cutoff, 1e-5))
        return 1.0 / (1.0 + tau / max(dt, 1e-5))

    def filter(self, x: np.ndarray, timestamp_s: float) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if self._x_prev is None or self._t_prev is None or timestamp_s <= self._t_prev:
            self._x_prev = arr.copy()
            self._dx_prev = np.zeros_like(arr)
            self._t_prev = float(timestamp_s)
            return arr
        dt = float(timestamp_s - self._t_prev)
        if dt > 0.5:
            self._x_prev = arr.copy()
            self._dx_prev = np.zeros_like(arr)
            self._t_prev = float(timestamp_s)
            return arr
        dx = (arr - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * (self._dx_prev if self._dx_prev is not None else dx)
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * arr + (1.0 - a) * self._x_prev
        self._x_prev = x_hat.copy()
        self._dx_prev = dx_hat.copy()
        self._t_prev = float(timestamp_s)
        return x_hat


def _normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-6 or not np.isfinite(n):
        return fallback.copy()
    return v / n


def compute_hand_rotation_cam(landmarks_3d_cam: np.ndarray, handedness: str) -> np.ndarray:
    """Compute right-handed 3x3 wrist orientation matrix in camera optical frame."""
    p0 = landmarks_3d_cam[0]
    p5 = landmarks_3d_cam[5]
    p9 = landmarks_3d_cam[9]
    p17 = landmarks_3d_cam[17]

    x_axis = _normalize(p9 - p0, np.array([0.0, -1.0, 0.0]))
    span = (p17 - p5) if handedness == "right" else (p5 - p17)
    across = _normalize(span, np.array([1.0, 0.0, 0.0]))
    z_axis = _normalize(np.cross(x_axis, across), np.array([0.0, 0.0, -1.0]))
    y_axis = _normalize(np.cross(z_axis, x_axis), np.array([1.0, 0.0, 0.0]))
    rot = np.column_stack([x_axis, y_axis, z_axis])
    if float(np.linalg.det(rot)) < 0.0:
        rot[:, 2] = -rot[:, 2]
    return rot


def rotation_to_quat_and_rot6d(rot: np.ndarray) -> tuple[list[float], list[float]]:
    quat = Rotation.from_matrix(rot).as_quat()
    rot6d = np.concatenate([rot[:, 0], rot[:, 1]])
    return [float(x) for x in quat], [float(x) for x in rot6d]


def compute_finger_curls_5d(landmarks_3d: np.ndarray) -> list[float]:
    """Estimate [thumb, index, middle, ring, pinky] flexion in [0.0=open, 1.0=closed]."""
    curls: list[float] = []
    for chain in FINGER_CHAINS:
        p0, p1, p2, p3 = (landmarks_3d[i] for i in chain)
        v1 = _normalize(p1 - p0, np.array([1.0, 0.0, 0.0]))
        v2 = _normalize(p2 - p1, v1)
        v3 = _normalize(p3 - p2, v2)
        cos1 = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
        cos2 = float(np.clip(np.dot(v2, v3), -1.0, 1.0))
        bend_rad = math.acos(cos1) + math.acos(cos2)
        curl = float(np.clip(bend_rad / (math.pi * 0.95), 0.0, 1.0))
        curls.append(curl)
    return curls


def _estimate_hand_mask_area_frac(lm_2d_px: np.ndarray, width: int, height: int) -> float:
    if len(lm_2d_px) < 3:
        return 0.0
    pts = np.round(lm_2d_px).astype(np.int32)
    hull = cv2.convexHull(pts)
    area = float(cv2.contourArea(hull))
    return float(np.clip(area / max(float(width * height), 1.0), 0.0, 1.0))


class MediaPipeHandTracker:
    """Apache-2.0 MediaPipe 21-joint hand detector with persistent left/right tracking."""

    def __init__(
        self,
        cfg: PerceptionConfig,
        intrinsics: CameraIntrinsicsModel,
        assumed_palm_width_m: float = 0.085,
        infer_width: int = 640,
    ) -> None:
        self.cfg = cfg
        self.intrinsics = intrinsics
        self.assumed_palm_width_m = float(assumed_palm_width_m)
        self.infer_width = int(infer_width)
        self._mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=cfg.max_num_hands,
            model_complexity=1,
            min_detection_confidence=cfg.hand_min_detection_conf,
            min_tracking_confidence=cfg.hand_min_tracking_conf,
        )
        self._filters: dict[str, OneEuroArrayFilter] = {
            "left": OneEuroArrayFilter(cfg.one_euro_min_cutoff, cfg.one_euro_beta),
            "right": OneEuroArrayFilter(cfg.one_euro_min_cutoff, cfg.one_euro_beta),
        }
        self._prev_wrist_px: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self._missing_streak: dict[str, int] = {"left": 0, "right": 0}
        self.observed_palm_spans_px: list[float] = []

    def close(self) -> None:
        self._mp_hands.close()

    def _backproject_landmarks(
        self,
        lm_2d_px: np.ndarray,
        world_lm_m: np.ndarray | None,
    ) -> np.ndarray:
        fx = self.intrinsics.fx
        fy = self.intrinsics.fy
        cx = self.intrinsics.cx
        cy = self.intrinsics.cy

        palm_px = float(np.linalg.norm(lm_2d_px[5] - lm_2d_px[17]))
        wrist_mid_px = float(np.linalg.norm(lm_2d_px[9] - lm_2d_px[0]))
        ref_px = max(palm_px, 0.85 * wrist_mid_px, 14.0)
        self.observed_palm_spans_px.append(ref_px)

        z_wrist = float(np.clip((fx * self.assumed_palm_width_m) / ref_px, 0.18, 1.15))
        u0, v0 = float(lm_2d_px[0, 0]), float(lm_2d_px[0, 1])
        wrist_cam = np.array([(u0 - cx) * z_wrist / fx, (v0 - cy) * z_wrist / fy, z_wrist], dtype=np.float64)

        if world_lm_m is not None and len(world_lm_m) == 21:
            mp_palm = float(np.linalg.norm(world_lm_m[5] - world_lm_m[17]))
            mp_mid = float(np.linalg.norm(world_lm_m[9] - world_lm_m[0]))
            mp_ref = max(mp_palm, 0.85 * mp_mid, 0.03)
            scale = float(np.clip(self.assumed_palm_width_m / mp_ref, 0.65, 1.45))
            rel = (world_lm_m - world_lm_m[0]) * scale
            return wrist_cam[None, :] + rel

        pts_3d = np.zeros((21, 3), dtype=np.float64)
        for i in range(21):
            u, v = float(lm_2d_px[i, 0]), float(lm_2d_px[i, 1])
            pts_3d[i] = [(u - cx) * z_wrist / fx, (v - cy) * z_wrist / fy, z_wrist]
        return pts_3d

    def _assign_handedness(
        self,
        candidates: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Persistently match up to 2 detections to ('left', 'right') across crossings/re-entries."""
        if not candidates:
            return {}
        if len(candidates) == 1:
            c = candidates[0]
            label = c["ego_label"]
            prev_same = self._prev_wrist_px[label]
            other = "right" if label == "left" else "left"
            prev_other = self._prev_wrist_px[other]
            if prev_same is None and prev_other is not None:
                dist_other = float(np.linalg.norm(c["wrist_px"] - prev_other))
                if dist_other < 0.08 * self.intrinsics.width and c["score"] < 0.72:
                    label = other
            return {label: c}

        c0, c1 = candidates[0], candidates[1]

        def _cost(cand: dict[str, Any], side: str) -> float:
            label_pen = 0.0 if cand["ego_label"] == side else 0.45 * cand["score"]
            prev = self._prev_wrist_px[side]
            if prev is not None and self._missing_streak[side] < 12:
                dist_norm = float(np.linalg.norm(cand["wrist_px"] - prev)) / max(float(self.intrinsics.width), 1.0)
            else:
                # Weak spatial prior only when no history exists: left hand tends left half, right hand right half
                x_norm = float(cand["wrist_px"][0]) / max(float(self.intrinsics.width), 1.0)
                dist_norm = 0.12 * (x_norm if side == "left" else (1.0 - x_norm))
            return label_pen + dist_norm

        cost_0l_1r = _cost(c0, "left") + _cost(c1, "right")
        cost_1l_0r = _cost(c1, "left") + _cost(c0, "right")
        if cost_0l_1r <= cost_1l_0r:
            return {"left": c0, "right": c1}
        return {"left": c1, "right": c0}

    def process_frame(self, bgr: np.ndarray, timestamp_s: float) -> tuple[HandState, HandState]:
        h, w = bgr.shape[:2]
        if w > self.infer_width:
            scale = self.infer_width / float(w)
            small = cv2.resize(bgr, (self.infer_width, max(16, round(h * scale))), interpolation=cv2.INTER_AREA)
        else:
            small = bgr
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        res = self._mp_hands.process(rgb)

        candidates: list[dict[str, Any]] = []
        if res.multi_hand_landmarks:
            world_list = res.multi_hand_world_landmarks or [None] * len(res.multi_hand_landmarks)
            handed_list = res.multi_handedness or [None] * len(res.multi_hand_landmarks)
            for lm_obj, wlm_obj, hand_cls in zip(res.multi_hand_landmarks, world_list, handed_list):
                lm_2d = np.array([[p.x * w, p.y * h] for p in lm_obj.landmark], dtype=np.float64)
                world_arr = (
                    np.array([[p.x, p.y, p.z] for p in wlm_obj.landmark], dtype=np.float64)
                    if wlm_obj is not None
                    else None
                )
                # MediaPipe assumes selfie-mirror; for outward head-mounted POV invert raw Left/Right
                raw_label = "Right"
                score = 0.7
                if hand_cls is not None and hand_cls.classification:
                    raw_label = hand_cls.classification[0].label
                    score = float(hand_cls.classification[0].score)
                ego_label = "right" if raw_label.lower() == "left" else "left"
                candidates.append(
                    {
                        "lm_2d": lm_2d,
                        "world_arr": world_arr,
                        "wrist_px": lm_2d[0],
                        "ego_label": ego_label,
                        "score": score,
                    }
                )

        assigned = self._assign_handedness(candidates[:2])
        states: dict[str, HandState] = {}

        for side in ("left", "right"):
            if side not in assigned:
                self._missing_streak[side] += 1
                if self._missing_streak[side] > 10:
                    self._filters[side].reset()
                    self._prev_wrist_px[side] = None
                states[side] = HandState(handedness=side)  # type: ignore[arg-type]
                continue

            self._missing_streak[side] = 0
            cand = assigned[side]
            lm_2d = cand["lm_2d"]
            self._prev_wrist_px[side] = lm_2d[0].copy()
            pts_3d_cam = self._backproject_landmarks(lm_2d, cand["world_arr"])
            if self.cfg.smoothing_method == "one_euro":
                pts_3d_cam = self._filters[side].filter(pts_3d_cam, timestamp_s)

            rot_cam = compute_hand_rotation_cam(pts_3d_cam, side)
            quat_cam, rot6d_cam = rotation_to_quat_and_rot6d(rot_cam)
            curls = compute_finger_curls_5d(pts_3d_cam)
            aperture_m = float(np.linalg.norm(pts_3d_cam[4] - pts_3d_cam[8]))
            in_bounds = (
                (lm_2d[:, 0] >= 4.0)
                & (lm_2d[:, 0] <= w - 4.0)
                & (lm_2d[:, 1] >= 4.0)
                & (lm_2d[:, 1] <= h - 4.0)
            )
            vis = float(np.mean(in_bounds))
            truncated = vis < 0.85
            mask_frac = _estimate_hand_mask_area_frac(lm_2d, w, h)
            tips_cam = [pts_3d_cam[i].tolist() for i in FINGERTIP_INDICES]

            states[side] = HandState(
                handedness=side,  # type: ignore[arg-type]
                detected=True,
                confidence=float(np.clip(cand["score"], 0.0, 1.0)),
                visibility=vis,
                truncated=truncated,
                landmarks_2d_px=[[round(float(x), 2), round(float(y), 2)] for x, y in lm_2d],
                landmarks_3d_cam_m=[[round(float(c), 5) for c in row] for row in pts_3d_cam],
                landmarks_3d_world_m=[[round(float(c), 5) for c in row] for row in pts_3d_cam],
                wrist_pos_cam_m=[round(float(c), 5) for c in pts_3d_cam[0]],
                wrist_pos_world_m=[round(float(c), 5) for c in pts_3d_cam[0]],
                wrist_quat_xyzw_world=quat_cam,
                wrist_rot6d_world=rot6d_cam,
                fingertips_world_m=[[round(float(c), 5) for c in tip] for tip in tips_cam],
                gripper_aperture_m=round(aperture_m, 5),
                finger_curls_5d=[round(float(c), 4) for c in curls],
                mask_area_frac=round(mask_frac, 5),
            )

        return states["left"], states["right"]


class MediaPipeUpperBodyTracker:
    """Apache-2.0 MediaPipe Pose Landmarker for arm/forearm/shoulder extraction when visible."""

    def __init__(self, infer_width: int = 384) -> None:
        self.infer_width = int(infer_width)
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        )

    def close(self) -> None:
        self._pose.close()

    def process_frame(self, bgr: np.ndarray) -> UpperBodyState:
        h, w = bgr.shape[:2]
        if w > self.infer_width:
            scale = self.infer_width / float(w)
            small = cv2.resize(bgr, (self.infer_width, max(16, round(h * scale))), interpolation=cv2.INTER_AREA)
        else:
            small = bgr
        res = self._pose.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        if not res.pose_world_landmarks:
            return UpperBodyState(detected=False, confidence=0.0)

        lms = res.pose_world_landmarks.landmark
        # Indices: 11=L_shoulder, 12=R_shoulder, 13=L_elbow, 14=R_elbow, 15=L_wrist, 16=R_wrist
        def _pt(idx: int) -> tuple[list[float] | None, float]:
            p = lms[idx]
            vis = float(getattr(p, "visibility", 0.0))
            if vis < 0.35:
                return None, 0.0
            return [round(float(p.x), 4), round(float(p.y), 4), round(float(p.z), 4)], vis

        ls, v_ls = _pt(11)
        rs, v_rs = _pt(12)
        le, v_le = _pt(13)
        re, v_re = _pt(14)
        lw, v_lw = _pt(15)
        rw, v_rw = _pt(16)
        vis_vals = [v for v in (v_ls, v_rs, v_le, v_re, v_lw, v_rw) if v > 0.0]
        if not vis_vals:
            return UpperBodyState(detected=False, confidence=0.0)

        pitch_deg: float | None = None
        if ls is not None and rs is not None and le is not None and re is not None:
            sh_mid = 0.5 * (np.array(ls) + np.array(rs))
            el_mid = 0.5 * (np.array(le) + np.array(re))
            arm_vec = el_mid - sh_mid
            pitch_deg = round(float(math.degrees(math.atan2(-arm_vec[1], max(abs(arm_vec[2]), 1e-4)))), 2)

        return UpperBodyState(
            detected=True,
            confidence=round(float(np.mean(vis_vals)), 4),
            left_shoulder_cam=ls,
            left_elbow_cam=le,
            left_wrist_cam=lw,
            right_shoulder_cam=rs,
            right_elbow_cam=re,
            right_wrist_cam=rw,
            torso_pitch_deg=pitch_deg,
        )


@register_backend(
    stage="hand_pose",
    name="mediapipe_hands",
    code_license="Apache-2.0",
    weights_license="Apache-2.0",
    requires_gpu=False,
    notes="Default commercially-safe 21-joint hand tracker (no MANO dependency).",
)
def build_mediapipe_hands(
    cfg: PerceptionConfig,
    intrinsics: CameraIntrinsicsModel,
    assumed_palm_width_m: float = 0.085,
) -> MediaPipeHandTracker:
    return MediaPipeHandTracker(cfg=cfg, intrinsics=intrinsics, assumed_palm_width_m=assumed_palm_width_m)


@register_backend(
    stage="body_pose",
    name="mediapipe_pose",
    code_license="Apache-2.0",
    weights_license="Apache-2.0",
    requires_gpu=False,
    notes="Default commercially-safe upper-body & arm pose tracker.",
)
def build_mediapipe_pose() -> MediaPipeUpperBodyTracker:
    return MediaPipeUpperBodyTracker()


@register_backend(
    stage="hand_pose",
    name="hamer_mano_optional",
    code_license="MIT",
    weights_license="MANO-Non-Commercial",
    commercial_ok=False,
    requires_gpu=True,
    notes="Quarantined non-commercial plugin (MANO license restricts commercial use). OFF by default.",
)
def build_hamer_optional(**_kwargs: Any) -> Any:
    raise RuntimeError("HaMeR/MANO optional plugin requires external HaMeR install and explicit non-commercial opt-in.")
