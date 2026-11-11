"""Metric hand pose from MediaPipe world landmarks via ``cv2.solvePnP``.

Contract: docs/ARCHITECTURE.md section 4. MediaPipe provides 21 image landmarks
and 21 world landmarks (metres, hand-local). PnP between the two, in pixels at
the tracked resolution with the camera matrix ``K``, gives the hand pose in the
OpenCV camera frame ``C`` with metric translation.

Hand frame ``H`` (section 3): origin at the wrist, +x toward the middle-finger
MCP (landmark 9), +z out of the palm, +y = z x x. The rule is identical for
left and right hands, which means the palm normal is computed with a
handedness-dependent sign (the two hands are mirror images).

Depth error is proportional to the focal-length error and to the ratio of the
real hand size to MediaPipe's canonical hand (``hand_scale``); both are
assumptions unless the camera is calibrated and the hand has been measured.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..datatypes import HandLandmarks, MetricHandPose

logger = logging.getLogger(__name__)

WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 0, 5, 9, 17
DEFAULT_ASSUMED_HFOV_DEG = 70.0


# ---------------------------------------------------------------------------
# Camera model
# ---------------------------------------------------------------------------


def _camera_section(cfg: Optional[dict]) -> dict:
    if not cfg:
        return {}
    if isinstance(cfg.get("camera"), dict):
        return cfg["camera"]
    if any(k in cfg for k in ("fx", "fy", "cx", "cy", "assumed_hfov_deg", "calib_size")):
        return cfg
    return {}


def camera_matrix(cfg: dict, image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return ``(K 3x3, dist coeffs, source)`` for an image of ``image_size`` (w, h).

    ``source`` is ``"calibrated"`` when ``camera.fx/fy/cx/cy`` are all set (given
    at ``camera.calib_size`` = ``[w, h]`` and rescaled to ``image_size``), else
    ``"assumed_fov"`` from ``camera.assumed_hfov_deg`` (default 70 degrees,
    square pixels, principal point at the image centre). Distortion
    coefficients come from ``camera.dist_coeffs`` (``[k1, k2, p1, p2(, k3...)]``,
    resolution independent) and are zero when assumed.

    Raises:
        ValueError: calibrated intrinsics without ``calib_size``, a calibration
            aspect ratio that differs from the image, or a non-positive size/FOV.
    """
    w, h = int(image_size[0]), int(image_size[1])
    if w <= 0 or h <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    cam = _camera_section(cfg)
    vals = [cam.get(k) for k in ("fx", "fy", "cx", "cy")]
    dist = np.zeros(5, dtype=np.float64)
    if cam.get("dist_coeffs") is not None:
        dist = np.asarray(cam["dist_coeffs"], dtype=np.float64).reshape(-1)

    if all(v is not None for v in vals):
        calib = cam.get("calib_size")
        if calib is None:
            raise ValueError(
                "camera.calib_size [width, height] is required with camera.fx/fy/cx/cy "
                "so the intrinsics can be rescaled to the tracked resolution."
            )
        cw, ch = float(calib[0]), float(calib[1])
        if cw <= 0 or ch <= 0:
            raise ValueError(f"camera.calib_size must be positive, got {calib}")
        sx, sy = w / cw, h / ch
        if abs(sx - sy) > 0.01 * max(sx, sy):
            raise ValueError(
                f"Calibration size {int(cw)}x{int(ch)} and tracked size {w}x{h} differ in "
                "aspect ratio; intrinsics cannot be rescaled (cropped or squashed frames)."
            )
        fx, fy, cx, cy = (float(v) for v in vals)
        K = np.array([[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]])
        return K, dist, "calibrated"

    hfov = float(cam.get("assumed_hfov_deg") or DEFAULT_ASSUMED_HFOV_DEG)
    if not 1.0 < hfov < 179.0:
        raise ValueError(f"camera.assumed_hfov_deg must be in (1, 179), got {hfov}")
    f = (w / 2.0) / math.tan(math.radians(hfov) / 2.0)
    K = np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])
    return K, np.zeros(5, dtype=np.float64), "assumed_fov"


# ---------------------------------------------------------------------------
# Hand frame
# ---------------------------------------------------------------------------


def hand_frame_rotation(points: np.ndarray, handedness: str) -> Optional[np.ndarray]:
    """Rotation whose columns are the hand-frame axes expressed in ``points``' frame.

    +x wrist -> middle MCP; +z out of the palm; +y = z x x. The palm normal is
    ``cross(index_mcp - wrist, pinky_mcp - wrist)`` for a right hand and its
    negative for a left hand. ``None`` for degenerate (collinear) landmarks.
    """
    p = np.asarray(points, dtype=np.float64)
    x = p[MIDDLE_MCP] - p[WRIST]
    nx = np.linalg.norm(x)
    if nx < 1e-9:
        return None
    x = x / nx
    n = np.cross(p[INDEX_MCP] - p[WRIST], p[PINKY_MCP] - p[WRIST])
    if str(handedness).lower().startswith("l"):
        n = -n
    n = n - np.dot(n, x) * x
    nn = np.linalg.norm(n)
    if nn < 1e-9:
        return None
    z = n / nn
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


# ---------------------------------------------------------------------------
# PnP lift
# ---------------------------------------------------------------------------


def _lift(
    hand: HandLandmarks,
    K: np.ndarray,
    dist: Optional[np.ndarray],
    source: str,
    hand_scale: float,
    max_reproj_px: Optional[float],
) -> Tuple[Optional[MetricHandPose], str]:
    """Core of :func:`lift_hand`; also returns a failure reason (``"ok"`` on success)."""
    if hand.world_xyz is None:
        return None, "no_world_landmarks"
    if hand.image_size is None:
        raise ValueError("HandLandmarks.image_size is required to convert landmarks to pixels")
    world = np.asarray(hand.world_xyz, dtype=np.float64).reshape(21, 3) * float(hand_scale)
    w, h = hand.image_size
    img = np.stack([np.asarray(hand.x, np.float64) * w, np.asarray(hand.y, np.float64) * h], axis=1)
    if not (np.isfinite(world).all() and np.isfinite(img).all()):
        return None, "non_finite"
    if np.linalg.matrix_rank(world - world.mean(axis=0), tol=1e-6) < 3:
        return None, "degenerate_landmarks"

    dist_arr = None if dist is None else np.asarray(dist, dtype=np.float64).reshape(-1, 1)
    try:
        ok, rvec, tvec = cv2.solvePnP(world, img, K, dist_arr, flags=cv2.SOLVEPNP_SQPNP)
        if ok:
            rvec, tvec = cv2.solvePnPRefineLM(world, img, K, dist_arr, rvec, tvec)
    except cv2.error as exc:
        logger.debug("solvePnP failed: %s", exc)
        return None, "pnp_failed"
    if not ok:
        return None, "pnp_failed"

    R_mp, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    cam_pts = world @ R_mp.T + t
    if not (cam_pts[:, 2] > 0).all():
        return None, "behind_camera"
    proj, _ = cv2.projectPoints(world, rvec, tvec, K, dist_arr)
    err = float(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1).mean())
    if max_reproj_px is not None and err > max_reproj_px:
        return None, "reprojection_error"

    R_hand_mp = hand_frame_rotation(world, hand.handedness)
    if R_hand_mp is None:
        return None, "degenerate_landmarks"
    return (
        MetricHandPose(
            wrist_position_m=cam_pts[WRIST].copy(),
            wrist_rotation=R_mp @ R_hand_mp,
            keypoints_m=cam_pts,
            reprojection_error_px=err,
            intrinsics_source=source,
            is_interpolated=bool(hand.is_interpolated),
        ),
        "ok",
    )


def lift_hand(
    hand: HandLandmarks,
    K: np.ndarray,
    dist: Optional[np.ndarray],
    source: str,
    hand_scale: float = 1.0,
    max_reproj_px: Optional[float] = None,
) -> Optional[MetricHandPose]:
    """Metric pose of one hand in the camera frame, or ``None``.

    ``None`` when world landmarks are missing (e.g. interpolated frames), PnP
    fails, a landmark falls behind the camera, or the mean reprojection error
    exceeds ``max_reproj_px``. ``hand_scale`` multiplies the world landmarks
    (real hand size / MediaPipe canonical hand size).

    Raises:
        ValueError: ``hand.image_size`` is unset while world landmarks exist.
    """
    pose, _ = _lift(hand, K, dist, source, hand_scale, max_reproj_px)
    return pose


def lift_frames(frames: Sequence[Any], cfg: Optional[dict]) -> Dict[str, Any]:
    """Fill ``left_hand_metric`` / ``right_hand_metric`` on AnnotationFrames in place.

    Reads ``camera`` (see :func:`camera_matrix`) and ``metric_hand`` from
    ``cfg``: ``max_reproj_px`` (default 12.0, at the tracked resolution) and
    ``hand_scale`` (default 1.0). Existing metric fields are overwritten, so
    stale values never survive a re-run.

    Returns:
        ``{"intrinsics_source", "hand_scale", "max_reproj_px", "left": {...},
        "right": {...}, "median_reprojection_error_px"}`` where each side has
        ``tracked``, ``lifted`` and ``failures`` (reason -> count).
    """
    mh = (cfg or {}).get("metric_hand") or {}
    max_reproj = mh.get("max_reproj_px", 12.0)
    max_reproj = None if max_reproj is None else float(max_reproj)
    hand_scale = float(mh.get("hand_scale", 1.0))

    cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, str]] = {}
    summary: Dict[str, Any] = {
        "intrinsics_source": None,
        "hand_scale": hand_scale,
        "max_reproj_px": max_reproj,
    }
    errors: List[float] = []
    sources = set()
    for side in ("left", "right"):
        tracked = lifted = 0
        failures: Counter = Counter()
        for fr in frames:
            hand = getattr(fr, f"{side}_hand", None)
            setattr(fr, f"{side}_hand_metric", None)
            if hand is None:
                continue
            tracked += 1
            if hand.world_xyz is None:
                failures["no_world_landmarks"] += 1
                continue
            if hand.image_size is None:
                failures["no_image_size"] += 1
                continue
            size = (int(hand.image_size[0]), int(hand.image_size[1]))
            if size not in cache:
                cache[size] = camera_matrix(cfg or {}, size)
            K, dist, source = cache[size]
            pose, reason = _lift(hand, K, dist, source, hand_scale, max_reproj)
            if pose is None:
                failures[reason] += 1
                continue
            setattr(fr, f"{side}_hand_metric", pose)
            lifted += 1
            errors.append(pose.reprojection_error_px)
            sources.add(source)
        summary[side] = {"tracked": tracked, "lifted": lifted, "failures": dict(failures)}
    sources.update(v[2] for v in cache.values())
    summary["intrinsics_source"] = (
        "calibrated" if sources == {"calibrated"} else "assumed_fov" if sources else None
    )
    summary["median_reprojection_error_px"] = float(np.median(errors)) if errors else None
    return summary


__all__ = [
    "camera_matrix",
    "hand_frame_rotation",
    "lift_hand",
    "lift_frames",
    "DEFAULT_ASSUMED_HFOV_DEG",
]
