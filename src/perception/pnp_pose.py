"""PnP pose refinement and pinhole back-projection (pure cv2 + numpy).

No heavyweight dependencies: everything here runs on the CPU-only 2017 Intel
Mac with the base requirements.

Failure contract for :func:`solve_pnp_pose`
-------------------------------------------
* ``ValueError`` on bad *input*: fewer than 4 correspondences, non-finite
  values, or degenerate (rank-deficient) 3D point sets.
* ``PnPResult(success=False, ...)`` when RANSAC finds no consensus — never
  silent garbage. A warning is logged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Camera model
# ---------------------------------------------------------------------------


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics (pixels)."""

    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        for name in ("fx", "fy", "cx", "cy"):
            v = float(getattr(self, name))
            if not np.isfinite(v):
                raise ValueError(f"CameraIntrinsics.{name} must be finite, got {v!r}")
            setattr(self, name, v)
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(
                f"CameraIntrinsics focal lengths must be positive, got fx={self.fx}, fy={self.fy}"
            )

    @property
    def matrix(self) -> np.ndarray:
        """3x3 intrinsic matrix K."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> Optional["CameraIntrinsics"]:
        """Build from a ``camera:`` config section; None when any field is null/missing.

        A null section means "intrinsics unknown" — callers must skip
        back-projection/PnP (with a warning), never silently assume a FOV.
        """
        if not d:
            return None
        try:
            vals = [d.get(k) for k in ("fx", "fy", "cx", "cy")]
            if any(v is None for v in vals):
                return None
            return cls(fx=float(vals[0]), fy=float(vals[1]), cx=float(vals[2]), cy=float(vals[3]))
        except (TypeError, ValueError) as e:
            logger.warning("Ignoring invalid camera intrinsics %r: %s", d, e)
            return None


# ---------------------------------------------------------------------------
# Back-projection
# ---------------------------------------------------------------------------


def backproject_pixel(u: float, v: float, z: float, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Back-project pixel ``(u, v)`` at depth ``z`` (meters) to camera frame.

    Returns:
        ``(3,)`` float64 array ``[x, y, z]`` in meters, camera frame.
    """
    z = float(z)
    if not np.isfinite(z) or z <= 0:
        raise ValueError(f"backproject_pixel requires positive finite depth, got {z!r}")
    x = (float(u) - intrinsics.cx) * z / intrinsics.fx
    y = (float(v) - intrinsics.cy) * z / intrinsics.fy
    return np.array([x, y, z], dtype=np.float64)


def backproject_to_3d(
    depth_map: np.ndarray, u: float, v: float, intrinsics: CameraIntrinsics
) -> np.ndarray:
    """Back-project pixel ``(u, v)`` sampling depth from ``depth_map`` (meters).

    Replaces the old 60-degree-FOV approximation in
    ``src/perception/unidepth_estimator.py::project_to_3d`` (removed in Phase C)
    with real intrinsics.
    """
    h, w = depth_map.shape[:2]
    iu = int(np.clip(round(v), 0, h - 1))
    iv = int(np.clip(round(u), 0, w - 1))
    return backproject_pixel(float(u), float(v), float(depth_map[iu, iv]), intrinsics)


# ---------------------------------------------------------------------------
# solvePnP wrapper
# ---------------------------------------------------------------------------


@dataclass
class PnPResult:
    """Outcome of :func:`solve_pnp_pose`."""

    success: bool
    rvec: Optional[np.ndarray]  # (3, 1) Rodrigues rotation, object -> camera
    tvec: Optional[np.ndarray]  # (3, 1) translation (meters), object -> camera
    inliers: Optional[np.ndarray]  # inlier correspondence indices
    mean_reproj_error_px: float  # mean reprojection error over inliers

    @property
    def rotation_matrix(self) -> Optional[np.ndarray]:
        """3x3 rotation matrix, or None when unsuccessful."""
        if not self.success or self.rvec is None:
            return None
        rmat, _ = cv2.Rodrigues(self.rvec)
        return rmat


def solve_pnp_pose(
    object_points_3d: np.ndarray,
    image_points_2d: np.ndarray,
    intrinsics: CameraIntrinsics,
    dist_coeffs: Optional[np.ndarray] = None,
    ransac_reproj_threshold_px: float = 8.0,
    ransac_iterations: int = 200,
) -> PnPResult:
    """Estimate object pose with RANSAC PnP + Levenberg-Marquardt refinement.

    Args:
        object_points_3d: ``(N, 3)`` 3D points in the object frame (meters).
        image_points_2d: ``(N, 2)`` corresponding pixel coordinates.
        intrinsics: Camera intrinsics.
        dist_coeffs: Optional distortion coefficients (4/5/8,); None = zero.
        ransac_reproj_threshold_px: RANSAC inlier threshold in pixels.
        ransac_iterations: RANSAC iteration budget.

    Returns:
        :class:`PnPResult`. ``success=False`` when RANSAC finds no consensus.

    Raises:
        ValueError: fewer than 4 correspondences, non-finite values, or a
            degenerate (collinear/coincident) 3D point set.
    """
    obj = np.asarray(object_points_3d, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(image_points_2d, dtype=np.float64).reshape(-1, 2)
    n = obj.shape[0]
    if img.shape[0] != n:
        raise ValueError(
            f"Correspondence count mismatch: {n} object points vs {img.shape[0]} image points"
        )
    if n < 4:
        raise ValueError(
            f"solve_pnp_pose needs >= 4 correspondences, got {n} "
            "(PnP is underdetermined below 4 points)."
        )
    if not (np.isfinite(obj).all() and np.isfinite(img).all()):
        raise ValueError("solve_pnp_pose requires finite object/image points (got NaN/inf).")
    # Degenerate geometry: object points must span at least a plane.
    if np.linalg.matrix_rank(obj - obj.mean(axis=0), tol=1e-9) < 2:
        raise ValueError(
            "Degenerate 3D point set: object points are coincident or collinear; "
            "PnP cannot determine a unique pose."
        )

    kmat = intrinsics.matrix
    dist = None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)

    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj,
            img,
            kmat,
            dist,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=float(ransac_reproj_threshold_px),
            iterationsCount=int(ransac_iterations),
        )
    except cv2.error as e:
        logger.warning("solvePnPRansac failed on valid input (%s); reporting no consensus.", e)
        return PnPResult(False, None, None, None, float("inf"))

    if not ok or inliers is None or int(inliers.size) < 4:
        logger.warning(
            "PnP RANSAC found no consensus (%s inliers); pose unavailable.",
            0 if inliers is None else int(inliers.size),
        )
        return PnPResult(False, None, None, None, float("inf"))

    # Refine on inliers with Levenberg-Marquardt starting from the RANSAC pose.
    inlier_idx = inliers.ravel()
    obj_in, img_in = obj[inlier_idx], img[inlier_idx]
    try:
        _, rvec, tvec = cv2.solvePnP(
            obj_in, img_in, kmat, dist, rvec, tvec, True, flags=cv2.SOLVEPNP_ITERATIVE
        )
    except cv2.error as e:
        logger.warning("PnP LM refinement failed (%s); keeping RANSAC pose.", e)

    proj, _ = cv2.projectPoints(obj_in, rvec, tvec, kmat, dist)
    err = float(np.linalg.norm(proj.reshape(-1, 2) - img_in, axis=1).mean())
    return PnPResult(True, rvec, tvec, inlier_idx, err)


# ---------------------------------------------------------------------------
# Hand PnP refinement
# ---------------------------------------------------------------------------


def _default_hand_reference_m() -> float:
    """Anthropometric wrist->middle-MCP reference in meters.

    Mirrors ``src.retargeting.metric_calibration.DEFAULT_HAND_REF_WRIST_TO_MCP_M``
    (ANSUR II / NASA STD-3000, 0.090 m). Imported lazily because importing
    ``src.retargeting`` pulls pybullet/MuJoCo/scipy via the package ``__init__``;
    falls back to the documented value when that import is unavailable.
    """
    try:
        from src.retargeting.metric_calibration import DEFAULT_HAND_REF_WRIST_TO_MCP_M

        return float(DEFAULT_HAND_REF_WRIST_TO_MCP_M)
    except Exception:
        return 0.090


_WRIST_IDX = 0
_MIDDLE_MCP_IDX = 9


class HandPnPRefiner:
    """Metric hand pose from MediaPipe landmarks via PnP.

    Builds a canonical 3D hand model from MediaPipe's 21x3 *relative* landmarks
    (wrist at origin, uniformly scaled so wrist->middle-MCP equals the
    anthropometric 0.090 m reference, +/-15% population variation documented in
    ``MetricCalibrator``), then runs :func:`solve_pnp_pose` against the 2D
    pixel landmarks. The recovered ``tvec`` is the metric wrist translation in
    the camera frame — an upgrade over MediaPipe's relative-z depth.

    Caveat: MediaPipe z is only *approximately* the same scale as x/y, so the
    canonical model is approximate; treat the translation as metric within the
    anthropometric error bounds, not as ground truth.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        reference_wrist_to_mcp_m: Optional[float] = None,
    ):
        self.intrinsics = intrinsics
        self.reference_m = (
            float(reference_wrist_to_mcp_m)
            if reference_wrist_to_mcp_m is not None
            else _default_hand_reference_m()
        )
        if self.reference_m <= 0:
            raise ValueError("reference_wrist_to_mcp_m must be positive")

    def canonical_hand_model(self, landmarks_3d_rel: np.ndarray) -> np.ndarray:
        """Scale MediaPipe-relative landmarks to a metric canonical model.

        Args:
            landmarks_3d_rel: ``(21, 3)`` MediaPipe relative landmarks.

        Returns:
            ``(21, 3)`` float64, wrist at origin, wrist->middle-MCP == reference_m.
        """
        pts = np.asarray(landmarks_3d_rel, dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] != 21:
            raise ValueError(f"expected 21 hand landmarks, got {pts.shape[0]}")
        if not np.isfinite(pts).all():
            raise ValueError("hand landmarks contain NaN/inf")
        d = float(np.linalg.norm(pts[_MIDDLE_MCP_IDX] - pts[_WRIST_IDX]))
        if d <= 1e-9:
            raise ValueError("Degenerate hand landmarks: wrist and middle MCP coincide.")
        return (pts - pts[_WRIST_IDX]) * (self.reference_m / d)

    def refine(
        self, landmarks_2d_px: np.ndarray, landmarks_3d_rel: np.ndarray
    ) -> PnPResult:
        """Run PnP for one hand.

        Args:
            landmarks_2d_px: ``(21, 2)`` pixel coordinates.
            landmarks_3d_rel: ``(21, 3)`` MediaPipe relative landmarks.

        Returns:
            :class:`PnPResult` with metric ``tvec`` on success.
        """
        canonical = self.canonical_hand_model(landmarks_3d_rel)
        img = np.asarray(landmarks_2d_px, dtype=np.float64).reshape(-1, 2)
        return solve_pnp_pose(canonical, img, self.intrinsics)


def refine_poses_pnp(
    frames: Sequence,
    intrinsics: CameraIntrinsics,
    image_width: int,
    image_height: int,
    enabled: bool = True,
) -> int:
    """Post-stage: metric wrist translation for each tracked hand via PnP.

    Results are stored additively in ``frame.metadata["hand_pnp"]`` as
    ``{"left": {"tvec_m": [x, y, z], "reproj_err_px": e, "inliers": n}, ...}``
    (JSON-safe). Per-frame failures are logged at debug level and skipped —
    this function never raises, so the pipeline always falls back to the
    existing (MediaPipe-relative) estimates.

    Args:
        frames: Sequence of :class:`AnnotationFrame`.
        intrinsics: Camera intrinsics for the frame resolution.
        image_width/image_height: Resolution the normalized landmarks refer to.
        enabled: Master switch (honors ``pnp_refinement.enabled``).

    Returns:
        Number of hands successfully refined.
    """
    if not enabled:
        return 0
    refiner = HandPnPRefiner(intrinsics)
    scale = np.array([float(image_width), float(image_height)], dtype=np.float64)
    n_ok = 0
    for frame in frames:
        entry: Dict[str, dict] = {}
        for side in ("left", "right"):
            hand = getattr(frame, f"{side}_hand", None)
            if hand is None:
                continue
            try:
                pts2d = np.stack([np.asarray(hand.x), np.asarray(hand.y)], axis=-1) * scale
                pts3d = np.stack(
                    [np.asarray(hand.x), np.asarray(hand.y), np.asarray(hand.z)], axis=-1
                )
                res = refiner.refine(pts2d, pts3d)
                if res.success:
                    entry[side] = {
                        "tvec_m": [float(v) for v in res.tvec.ravel().tolist()],
                        "reproj_err_px": float(res.mean_reproj_error_px),
                        "inliers": int(res.inliers.size),
                    }
                    n_ok += 1
            except Exception as e:
                logger.debug(
                    "PnP refinement skipped for frame %s hand %s: %s",
                    getattr(frame, "frame_idx", "?"),
                    side,
                    e,
                )
        if entry:
            if not isinstance(frame.metadata, dict):
                frame.metadata = {}
            frame.metadata["hand_pnp"] = entry
    logger.info("PnP hand-pose refinement: %d/%d hands refined", n_ok, len(frames) * 2)
    return n_ok


__all__ = [
    "CameraIntrinsics",
    "PnPResult",
    "solve_pnp_pose",
    "backproject_to_3d",
    "backproject_pixel",
    "HandPnPRefiner",
    "refine_poses_pnp",
]
