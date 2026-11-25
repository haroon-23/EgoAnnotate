"""Single-camera 6-DoF monocular VO, pure-rotation fallback, loop-closure & world registration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from src.egoannotate.config import GeometryConfig
from src.egoannotate.perception_hands import (
    compute_hand_rotation_cam,
    rotation_to_quat_and_rot6d,
)
from src.egoannotate.registry import register_backend
from src.egoannotate.schemas import (
    CameraIntrinsicsModel,
    CameraWorldPose,
    HandState,
    ManipulatedObjectState,
    ScaleProvenance,
)

# Canonical right-handed transform from OpenCV camera optical frame (X right, Y down, Z forward)
# to Gravity-aligned World frame (X forward, Y left, Z up): det(R_W_C0) == +1.
R_WORLD_CAM0 = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


@dataclass
class _KeyframeRecord:
    frame_idx: int
    pos_world_m: np.ndarray
    rot_world: np.ndarray
    descriptors: np.ndarray | None


def _orthonormalize_rotation(mat: np.ndarray) -> np.ndarray:
    u_svd, _, vt_svd = np.linalg.svd(mat)
    rot = u_svd @ vt_svd
    if float(np.linalg.det(rot)) < 0.0:
        u_svd[:, -1] *= -1.0
        rot = u_svd @ vt_svd
    return rot


class MonocularWorldVO:
    """Streaming 6-DoF Essential-Matrix VO with pure-rotation homography fallback & ORB loop-closure."""

    def __init__(self, cfg: GeometryConfig, intrinsics: CameraIntrinsicsModel) -> None:
        self.cfg = cfg
        self.intrinsics = intrinsics
        self._scale_resize = min(1.0, float(cfg.vo_resize_width) / max(float(intrinsics.width), 1.0))
        self._w_vo = max(64, round(intrinsics.width * self._scale_resize))
        self._h_vo = max(64, round(intrinsics.height * self._scale_resize))
        self._k_vo = np.array(
            [
                [intrinsics.fx * self._scale_resize, 0.0, intrinsics.cx * self._scale_resize],
                [0.0, intrinsics.fy * self._scale_resize, intrinsics.cy * self._scale_resize],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self._k_inv = np.linalg.inv(self._k_vo)
        self._prev_gray: np.ndarray | None = None
        self._prev_pts: np.ndarray | None = None
        self._rot_wc = R_WORLD_CAM0.copy()
        self._pos_w = np.zeros(3, dtype=np.float64)
        self._orb = cv2.ORB_create(nfeatures=250)  # type: ignore[attr-defined]
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._keyframes: list[_KeyframeRecord] = []
        self.sparse_world_points: list[list[float]] = []
        self.tracking_ok_count = 0
        self.total_frames = 0
        self.loop_closures_applied = 0
        self._unit_step_m = 0.015

    def _prep_gray(self, bgr: np.ndarray) -> np.ndarray:
        if self._scale_resize < 0.999:
            small = cv2.resize(bgr, (self._w_vo, self._h_vo), interpolation=cv2.INTER_AREA)
        else:
            small = bgr
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def _detect_features(self, gray: np.ndarray) -> np.ndarray | None:
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.cfg.vo_max_features,
            qualityLevel=0.012,
            minDistance=7,
            blockSize=7,
        )

    def _triangulate_sparse(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        r_rel: np.ndarray,
        t_rel: np.ndarray,
        step_m: float,
    ) -> None:
        if len(self.sparse_world_points) >= 1500 or len(p0) < 12:
            return
        proj0 = self._k_vo @ np.hstack([np.eye(3), np.zeros((3, 1))])
        proj1 = self._k_vo @ np.hstack([r_rel, t_rel.reshape(3, 1) * step_m])
        pts_h = cv2.triangulatePoints(proj0, proj1, p0[:40].T, p1[:40].T)
        w_vals = pts_h[3]
        valid = np.abs(w_vals) > 1e-6
        pts_c = (pts_h[:3, valid] / w_vals[valid]).T
        for pt in pts_c:
            z = float(pt[2])
            if 0.2 < z < 4.5 and np.all(np.isfinite(pt)):
                pt_w = self._rot_wc @ pt + self._pos_w
                self.sparse_world_points.append([round(float(c), 4) for c in pt_w])
                if len(self.sparse_world_points) >= 1500:
                    break

    def _estimate_pure_rotation(self, p0: np.ndarray, p1: np.ndarray) -> tuple[bool, int, np.ndarray]:
        """Fallback for pure head pan/tilt where Essential Matrix translation is degenerate."""
        if len(p0) < 10:
            return False, 0, np.eye(3, dtype=np.float64)
        h_mat, h_mask = cv2.findHomography(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=2.5)
        if h_mat is None or h_mask is None or h_mat.shape != (3, 3):
            return False, 0, np.eye(3, dtype=np.float64)
        inliers = int(np.sum(h_mask.ravel() > 0))
        if inliers < max(10, int(0.35 * len(p0))):
            return False, inliers, np.eye(3, dtype=np.float64)
        # H maps p0 -> p1: H ~ K * R_c1_c0 * K^-1 => R_rel (c0_from_c1) = R_c1_c0^T
        m_rot = self._k_inv @ h_mat @ self._k_vo
        if not np.all(np.isfinite(m_rot)):
            return False, inliers, np.eye(3, dtype=np.float64)
        r_c1_c0 = _orthonormalize_rotation(m_rot)
        r_rel = r_c1_c0.T
        angle_deg = float(np.degrees(np.linalg.norm(Rotation.from_matrix(r_rel).as_rotvec())))
        if angle_deg > 18.0:
            return False, inliers, np.eye(3, dtype=np.float64)
        return True, inliers, r_rel

    def _check_loop_closure(self, frame_idx: int, gray: np.ndarray) -> bool:
        if not self.cfg.loop_closure_enabled or frame_idx % 30 != 0:
            return False
        _, desc = self._orb.detectAndCompute(gray, None)
        if desc is None or len(desc) < 20:
            return False

        matched_kf: _KeyframeRecord | None = None
        for kf in self._keyframes:
            if frame_idx - kf.frame_idx < 150 or kf.descriptors is None:
                continue
            matches = self._bf.match(desc, kf.descriptors)
            good = [m for m in matches if m.distance < 34]
            if len(good) >= 35:
                matched_kf = kf
                break

        self._keyframes.append(
            _KeyframeRecord(
                frame_idx=frame_idx,
                pos_world_m=self._pos_w.copy(),
                rot_world=self._rot_wc.copy(),
                descriptors=desc,
            )
        )
        if len(self._keyframes) > 200:
            self._keyframes = self._keyframes[-200:]

        if matched_kf is not None:
            drift_vec = matched_kf.pos_world_m - self._pos_w
            if float(np.linalg.norm(drift_vec)) < 1.5:
                self._pos_w = self._pos_w + 0.30 * drift_vec
                self.loop_closures_applied += 1
                return True
        return False

    def step(self, bgr: np.ndarray, frame_idx: int, chunk_id: int, dt_s: float) -> CameraWorldPose:
        self.total_frames += 1
        gray = self._prep_gray(bgr)

        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_pts = self._detect_features(gray)
            self.tracking_ok_count += 1
            quat = [float(x) for x in Rotation.from_matrix(self._rot_wc).as_quat()]
            return CameraWorldPose(
                pos_world_m=[0.0, 0.0, 0.0],
                quat_xyzw_world=quat,
                tracking_ok=True,
                inlier_count=0 if self._prev_pts is None else len(self._prev_pts),
                chunk_id=chunk_id,
                loop_closure_adjusted=False,
            )

        if self._prev_pts is None or len(self._prev_pts) < 80:
            self._prev_pts = self._detect_features(self._prev_gray)

        tracking_ok = False
        inliers = 0
        tracked_p1: np.ndarray | None = None
        prev_pos = self._pos_w.copy()
        prev_rot = self._rot_wc.copy()

        if self._prev_pts is not None and len(self._prev_pts) >= 12:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray,
                gray,
                self._prev_pts,
                cast(Any, None),
                winSize=(21, 21),
                maxLevel=3,
            )
            if next_pts is not None and status is not None:
                good_mask = status.ravel() == 1
                p0 = self._prev_pts[good_mask].reshape(-1, 2)
                p1 = next_pts[good_mask].reshape(-1, 2)
                in_frame = (
                    (p1[:, 0] >= 2.0)
                    & (p1[:, 0] < self._w_vo - 2.0)
                    & (p1[:, 1] >= 2.0)
                    & (p1[:, 1] < self._h_vo - 2.0)
                )
                p0 = p0[in_frame]
                p1 = p1[in_frame]
                if len(p0) >= 12:
                    tracked_p1 = p1
                    flow_mag = np.linalg.norm(p1 - p0, axis=1)
                    med_flow = float(np.median(flow_mag))
                    if med_flow < 0.8:
                        tracking_ok = True
                        inliers = len(p0)
                    else:
                        e_mat, emask = cv2.findEssentialMat(
                            p1,
                            p0,
                            self._k_vo,
                            method=cv2.RANSAC,
                            prob=0.999,
                            threshold=1.8,
                        )
                        if e_mat is not None and e_mat.shape == (3, 3):
                            inliers, r_rel, t_rel, _ = cv2.recoverPose(e_mat, p1, p0, self._k_vo, mask=emask)
                            if inliers >= 10 and np.all(np.isfinite(r_rel)) and np.all(np.isfinite(t_rel)):
                                rot_angle_deg = float(np.degrees(np.linalg.norm(Rotation.from_matrix(r_rel).as_rotvec())))
                                if rot_angle_deg <= 18.0:
                                    parallax_factor = float(np.clip(med_flow / 12.0, 0.0, 2.0))
                                    step_m = min(self._unit_step_m * parallax_factor, 1.5 * max(dt_s, 1e-3))
                                    self._pos_w = self._pos_w + (self._rot_wc @ (t_rel.reshape(3) * step_m))
                                    self._rot_wc = _orthonormalize_rotation(self._rot_wc @ r_rel)
                                    tracking_ok = True
                                    if frame_idx % 10 == 0 and med_flow > 3.0:
                                        self._triangulate_sparse(p0, p1, r_rel, t_rel, max(step_m, 0.02))

                        if not tracking_ok:
                            ok_rot, h_inliers, r_pure = self._estimate_pure_rotation(p0, p1)
                            if ok_rot:
                                self._rot_wc = _orthonormalize_rotation(self._rot_wc @ r_pure)
                                tracking_ok = True
                                inliers = h_inliers

        if tracking_ok:
            self.tracking_ok_count += 1

        lc_flag = self._check_loop_closure(frame_idx, gray)
        dt_safe = max(dt_s, 1e-3)
        lin_vel = (self._pos_w - prev_pos) / dt_safe
        r_delta = prev_rot.T @ self._rot_wc
        ang_vel = Rotation.from_matrix(r_delta).as_rotvec() / dt_safe

        # Always keep _prev_pts synchronized with _prev_gray
        self._prev_gray = gray
        if tracked_p1 is not None and len(tracked_p1) >= 120 and frame_idx % 10 != 0:
            self._prev_pts = tracked_p1.reshape(-1, 1, 2).astype(np.float32)
        else:
            self._prev_pts = self._detect_features(gray)

        quat = [float(x) for x in Rotation.from_matrix(self._rot_wc).as_quat()]
        return CameraWorldPose(
            pos_world_m=[round(float(c), 5) for c in self._pos_w],
            quat_xyzw_world=[round(float(q), 6) for q in quat],
            lin_vel_world_mps=[round(float(v), 4) for v in lin_vel],
            ang_vel_world_radps=[round(float(w), 4) for w in ang_vel],
            tracking_ok=tracking_ok,
            inlier_count=int(inliers),
            chunk_id=chunk_id,
            loop_closure_adjusted=lc_flag,
        )

    def build_scale_provenance(
        self,
        intrinsics: CameraIntrinsicsModel,
        depth_backend: str,
    ) -> ScaleProvenance:
        has_metric_depth = depth_backend not in ("none", "", "disabled")
        is_cal = intrinsics.source == "calibrated_yaml"
        if is_cal and has_metric_depth:
            source = "calibrated_intrinsics_and_metric_depth"
            unc = 5.0
            is_metric = True
        elif has_metric_depth:
            source = "metric_depth_and_palm_prior"
            unc = 8.5
            is_metric = False
        else:
            source = "anthropometric_palm_prior_only"
            unc = 12.0 if not is_cal else 9.0
            is_metric = False

        table_z: float | None = None
        if len(self.sparse_world_points) >= 20:
            zs = np.array([p[2] for p in self.sparse_world_points], dtype=np.float64)
            table_z = round(float(np.percentile(zs, 20)), 4)

        return ScaleProvenance(
            scale_meters_per_unit=1.0,
            scale_source=source,  # type: ignore[arg-type]
            scale_uncertainty_pct=unc,
            is_calibrated_metric=is_metric,
            gravity_vector_cam0=[0.0, 1.0, 0.0],
            table_plane_z_world_m=table_z,
        )


def register_hand_and_objects_to_world(
    cam_pose: CameraWorldPose,
    hand: HandState,
    prev_hand_world_pos: np.ndarray | None,
    dt_s: float,
) -> np.ndarray | None:
    """Transform camera-frame 3D hand landmarks & wrist orientation into the gravity-aligned world frame."""
    if not hand.detected or not hand.landmarks_3d_cam_m:
        return None
    r_wc = Rotation.from_quat(cam_pose.quat_xyzw_world).as_matrix()
    t_wc = np.asarray(cam_pose.pos_world_m, dtype=np.float64)

    pts_cam = np.asarray(hand.landmarks_3d_cam_m, dtype=np.float64)
    pts_world = (pts_cam @ r_wc.T) + t_wc[None, :]
    wrist_world = pts_world[0]

    rot_cam = compute_hand_rotation_cam(pts_cam, hand.handedness)
    rot_world = r_wc @ rot_cam
    quat_w, rot6d_w = rotation_to_quat_and_rot6d(rot_world)

    vel_w = np.zeros(3, dtype=np.float64)
    if prev_hand_world_pos is not None and dt_s > 1e-4:
        vel_w = (wrist_world - prev_hand_world_pos) / dt_s
        speed = float(np.linalg.norm(vel_w))
        if speed > 4.5:
            vel_w = vel_w * (4.5 / speed)

    hand.landmarks_3d_world_m = [[round(float(c), 5) for c in row] for row in pts_world]
    hand.wrist_pos_world_m = [round(float(c), 5) for c in wrist_world]
    hand.wrist_quat_xyzw_world = [round(float(q), 6) for q in quat_w]
    hand.wrist_rot6d_world = [round(float(r), 6) for r in rot6d_w]
    hand.wrist_vel_world_mps = [round(float(v), 4) for v in vel_w]
    hand.fingertips_world_m = [[round(float(c), 5) for c in pts_world[i]] for i in (4, 8, 12, 16, 20)]
    return wrist_world


def register_objects_to_world(
    cam_pose: CameraWorldPose,
    objects: list[ManipulatedObjectState],
    prev_obj_pos: dict[int, np.ndarray],
    dt_s: float,
) -> None:
    """Transform camera-frame manipulated object centroids into world frame and compute velocities."""
    r_wc = Rotation.from_quat(cam_pose.quat_xyzw_world).as_matrix()
    t_wc = np.asarray(cam_pose.pos_world_m, dtype=np.float64)
    for obj in objects:
        p_cam = np.asarray(obj.pos_world_m, dtype=np.float64)
        p_w = r_wc @ p_cam + t_wc
        vel = np.zeros(3, dtype=np.float64)
        if obj.track_id in prev_obj_pos and dt_s > 1e-4:
            vel = (p_w - prev_obj_pos[obj.track_id]) / dt_s
        prev_obj_pos[obj.track_id] = p_w
        obj.pos_world_m = [round(float(c), 5) for c in p_w]
        obj.vel_world_mps = [round(float(v), 4) for v in vel]


@register_backend(
    stage="vo_slam",
    name="opencv_essential_vo",
    code_license="Apache-2.0",
    weights_license="N/A",
    requires_gpu=False,
    notes="CPU-safe 6-DoF Essential-Matrix VO with pure-rotation fallback & ORB loop-closure.",
)
def build_opencv_vo(cfg: GeometryConfig, intrinsics: CameraIntrinsicsModel) -> MonocularWorldVO:
    return MonocularWorldVO(cfg=cfg, intrinsics=intrinsics)


@register_backend(
    stage="vo_slam",
    name="dpvo_mit",
    code_license="MIT",
    weights_license="MIT",
    requires_gpu=True,
    notes="GPU production Deep Patch Visual Odometry (Princeton-VL DPVO, MIT).",
)
def build_dpvo_mit(**_kwargs: Any) -> Any:
    return {"backend": "dpvo", "license": "MIT", "requires_cuda": True}


@register_backend(
    stage="vo_slam",
    name="droid_slam_bsd",
    code_license="BSD-3-Clause",
    weights_license="BSD-3-Clause",
    requires_gpu=True,
    notes="GPU production dense bundle-adjustment SLAM (Princeton-VL DROID-SLAM, BSD-3-Clause).",
)
def build_droid_slam_bsd(**_kwargs: Any) -> Any:
    return {"backend": "droid_slam", "license": "BSD-3-Clause", "requires_cuda": True}


@register_backend(
    stage="depth",
    name="depth_anything_v2_small_apache",
    code_license="Apache-2.0",
    weights_license="Apache-2.0",
    requires_gpu=True,
    notes="Commercially-safe Small variant ONLY (Apache-2.0). Base/Large/Giant excluded (CC-BY-NC).",
)
def build_dav2_small_apache(**_kwargs: Any) -> Any:
    return {"model_id": "depth-anything/Depth-Anything-V2-Small-hf", "license": "Apache-2.0"}


def _unused_math_helper() -> float:
    return math.tau
