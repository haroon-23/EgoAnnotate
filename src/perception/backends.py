from __future__ import annotations
"""Pluggable Camera-Frame 24-D Hand State & Camera Pose Backends."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple
import numpy as np

STATE_24D_NAMES = [
    "wrist_x", "wrist_y", "wrist_z",
    "rot6_0", "rot6_1", "rot6_2", "rot6_3", "rot6_4", "rot6_5",
    "gripper",
    "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13",
]

ACTION_24D_NAMES = [f"delta_{n}" for n in STATE_24D_NAMES]

FINGER_TRIPLETS: List[Tuple[int, int, int]] = [
    (1, 2, 3), (2, 3, 4),
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
]


@dataclass
class CameraIntrinsicsProfile:
    device_id: str
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: List[float]
    calibrated: bool
    scale_source: str
    is_metric: bool
    source: str = "assumed"
    distortion_model: str = "pinhole"  # "pinhole" | "fisheye"

    @classmethod
    def load(cls, device_id: str = "generic_smartphone", img_w: int = 1024, img_h: int = 576) -> "CameraIntrinsicsProfile":
        dev_file = Path("configs/devices") / f"{device_id}.yaml"
        if dev_file.exists():
            try:
                import yaml
                raw = yaml.safe_load(dev_file.read_text(encoding="utf-8")) or {}
                intr = raw.get("intrinsics", {})
                orig_w = float(raw.get("image_width", img_w))
                orig_h = float(raw.get("image_height", img_h))
                scale_x = float(img_w) / max(1.0, orig_w)
                scale_y = float(img_h) / max(1.0, orig_h)

                fx = float(intr.get("fx", 0.85 * max(img_w, img_h))) * scale_x
                fy = float(intr.get("fy", 0.85 * max(img_w, img_h))) * scale_y
                cx = float(intr.get("cx", 0.5 * img_w)) * scale_x
                cy = float(intr.get("cy", 0.5 * img_h)) * scale_y
                dist = [float(v) for v in intr.get("dist_coeffs", [0.0] * 4)]
                model = str(raw.get("distortion_model", "fisheye" if len(dist) == 4 else "pinhole"))

                return cls(
                    device_id=str(raw.get("device_id", device_id)),
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    dist_coeffs=dist,
                    calibrated=bool(raw.get("calibrated", False)),
                    scale_source=str(raw.get("scale_source", "assumed")),
                    source=str(raw.get("source", "assumed")),
                    is_metric=bool(raw.get("is_metric", False)),
                    distortion_model=model,
                )
            except Exception:
                pass

        m_dim = float(max(img_w, img_h))
        return cls(
            device_id=device_id,
            fx=0.85 * m_dim,
            fy=0.85 * m_dim,
            cx=0.5 * float(img_w),
            cy=0.5 * float(img_h),
            dist_coeffs=[0.0] * 5,
            calibrated=False,
            scale_source="assumed_palm_85mm",
            source="assumed",
            is_metric=False,
            distortion_model="pinhole",
        )

    def undistort_points(self, pts_2d: np.ndarray) -> np.ndarray:
        """Undistort 2D pixel coordinates using camera calibration.

        pts_2d: shape (N, 2) in pixel space.
        Returns: undistorted pixel coordinates (N, 2).
        """
        if not any(abs(c) > 1e-6 for c in self.dist_coeffs):
            return pts_2d
        import cv2
        k = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)
        d = np.array(self.dist_coeffs, dtype=np.float64)
        pts_in = pts_2d.reshape(-1, 1, 2).astype(np.float64)
        if self.distortion_model == "fisheye":
            undist_norm = cv2.fisheye.undistortPoints(pts_in, k, d)
            u = undist_norm[:, 0, 0] * self.fx + self.cx
            v = undist_norm[:, 0, 1] * self.fy + self.cy
            return np.stack([u, v], axis=-1)
        else:
            undist_norm = cv2.undistortPoints(pts_in, k, d)
            u = undist_norm[:, 0, 0] * self.fx + self.cx
            v = undist_norm[:, 0, 1] * self.fy + self.cy
            return np.stack([u, v], axis=-1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "calibrated": self.calibrated,
            "source": self.source,
            "distortion_model": self.distortion_model,
            "scale_source": self.scale_source,
            "is_metric": self.is_metric,
            "fx": round(self.fx, 2),
            "fy": round(self.fy, 2),
            "cx": round(self.cx, 2),
            "cy": round(self.cy, 2),
            "dist_coeffs": [round(c, 5) for c in self.dist_coeffs],
        }


class MediaPipeBaselineHandBackend:
    capability_status: str = "PARTIAL"

    def estimate_camera_frame_hand_state(
        self,
        hand_landmarks: Any,
        intrinsics: Optional[CameraIntrinsicsProfile] = None,
        img_w: int = 1024,
        img_h: int = 576,
    ) -> np.ndarray:
        if intrinsics is None:
            intrinsics = CameraIntrinsicsProfile.load("generic_smartphone", img_w, img_h)
        st = np.zeros(24, dtype=np.float32)
        if hand_landmarks is None:
            return st

        xn = np.asarray(hand_landmarks.x, dtype=np.float64)
        yn = np.asarray(hand_landmarks.y, dtype=np.float64)
        zn = np.asarray(hand_landmarks.z, dtype=np.float64)

        # Isotropic normalized palm length in focal units (never clips at 1.15m!)
        palm_norm = max(float(np.hypot(xn[9] - xn[0], yn[9] - yn[0])), 0.025)
        z_wrist = float(np.clip((0.085 * 0.85) / palm_norm + zn[0] * 0.25, 0.15, 2.80))

        # Undistort raw 2D pixel coordinates before lifting to 3D metric rays
        pts_2d = np.stack([xn * float(img_w), yn * float(img_h)], axis=-1)
        pts_2d_undist = intrinsics.undistort_points(pts_2d)
        u = pts_2d_undist[:, 0]
        v = pts_2d_undist[:, 1]
        z_kps = z_wrist + zn * 0.085
        x_kps = (u - intrinsics.cx) * z_kps / intrinsics.fx
        y_kps = (v - intrinsics.cy) * z_kps / intrinsics.fy
        pts_3d = np.stack([x_kps, y_kps, z_kps], axis=1)

        st[0:3] = pts_3d[0].astype(np.float32)

        v_fwd = pts_3d[9] - pts_3d[0]
        e_y = v_fwd / max(float(np.linalg.norm(v_fwd)), 1e-6)
        v_lat = pts_3d[5] - pts_3d[17]
        v_lat_orth = v_lat - np.dot(v_lat, e_y) * e_y
        e_x = v_lat_orth / max(float(np.linalg.norm(v_lat_orth)), 1e-6)
        st[3:9] = np.concatenate([e_x, e_y]).astype(np.float32)

        pinch_m = float(np.linalg.norm(pts_3d[4] - pts_3d[8]))
        palm_m = max(float(np.linalg.norm(pts_3d[9] - pts_3d[0])), 0.03)
        st[9] = np.float32(np.clip(pinch_m / (1.35 * palm_m), 0.01, 0.99))

        for idx_t, (ia, ib, ic) in enumerate(FINGER_TRIPLETS):
            b1 = pts_3d[ib] - pts_3d[ia]
            b2 = pts_3d[ic] - pts_3d[ib]
            n1 = max(float(np.linalg.norm(b1)), 1e-6)
            n2 = max(float(np.linalg.norm(b2)), 1e-6)
            cos_ang = float(np.clip(np.dot(b1 / n1, b2 / n2), -0.999, 0.999))
            st[10 + idx_t] = np.float32(np.arccos(cos_ang))
        return st


def compute_24d_camera_frame_state_and_actions(
    frames: List[Any],
    img_w: int = 1024,
    img_h: int = 576,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(frames)
    states = np.zeros((n, 24), dtype=np.float32)
    actions = np.zeros((n, 24), dtype=np.float32)
    hand_valid = np.zeros((n, 2), dtype=np.int64)

    intr = CameraIntrinsicsProfile.load("generic_smartphone", img_w, img_h)
    backend = MediaPipeBaselineHandBackend()

    last_valid_state = np.array([
        0.0, 0.0, 0.55,
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        0.5,
        0.2, 0.2, 0.3, 0.3, 0.2, 0.3, 0.3, 0.2, 0.3, 0.3, 0.2, 0.3, 0.3, 0.2,
    ], dtype=np.float32)

    raw_states = []
    any_valid_1d = np.zeros(n, dtype=bool)

    for i, f in enumerate(frames):
        lh, rh = getattr(f, "left_hand", None), getattr(f, "right_hand", None)
        l_ok = int(lh is not None and not getattr(lh, "is_interpolated", False))
        r_ok = int(rh is not None and not getattr(rh, "is_interpolated", False))
        is_v = 1 if (l_ok or r_ok) else 0
        hand_valid[i] = [is_v, is_v]

        active_h = rh if (rh is not None and r_ok) else (lh if (lh is not None and l_ok) else (rh or lh))
        if active_h is not None and (l_ok or r_ok):
            any_valid_1d[i] = True
            st = backend.estimate_camera_frame_hand_state(active_h, intr, img_w, img_h)
            last_valid_state = st.copy()
            raw_states.append(st)
        else:
            raw_states.append(last_valid_state.copy())

    states = np.stack(raw_states, axis=0) if raw_states else states
    valid_idxs = np.where(any_valid_1d)[0]
    if len(valid_idxs) > 0:
        states[:valid_idxs[0]] = states[valid_idxs[0]]

    if n > 1:
        pair_valid = any_valid_1d[:-1] & any_valid_1d[1:]
        diffs = (states[1:] - states[:-1]).astype(np.float32)
        actions[:-1][pair_valid] = diffs[pair_valid]
        actions[-1] = 0.0

    return states, actions, hand_valid
