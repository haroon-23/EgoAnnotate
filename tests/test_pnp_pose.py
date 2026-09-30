"""Tests for src/perception/pnp_pose.py — pure cv2/numpy, synthetic data only."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.perception.pnp_pose import (
    CameraIntrinsics,
    HandPnPRefiner,
    backproject_pixel,
    backproject_to_3d,
    refine_poses_pnp,
    solve_pnp_pose,
)


@pytest.fixture
def intrinsics():
    return CameraIntrinsics(fx=800.0, fy=800.0, cx=320.0, cy=240.0)


@pytest.fixture
def synthetic_scene(intrinsics):
    """Known pose + 3D points -> projected 2D correspondences."""
    rng = np.random.default_rng(7)
    obj_pts = np.column_stack(
        [
            rng.uniform(-0.3, 0.3, 14),
            rng.uniform(-0.3, 0.3, 14),
            rng.uniform(0.8, 1.5, 14),
        ]
    )
    rvec_true = np.array([[0.2], [-0.15], [0.1]])
    tvec_true = np.array([[0.10], [-0.05], [1.20]])
    img_pts, _ = cv2.projectPoints(
        obj_pts, rvec_true, tvec_true, intrinsics.matrix, None
    )
    return obj_pts, img_pts.reshape(-1, 2), rvec_true, tvec_true


def _rotation_angle_deg(rvec_a, rvec_b):
    ra, _ = cv2.Rodrigues(rvec_a)
    rb, _ = cv2.Rodrigues(rvec_b)
    cos_a = float(np.clip((np.trace(ra.T @ rb) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def test_solve_pnp_recovers_known_pose(synthetic_scene, intrinsics):
    obj_pts, img_pts, rvec_true, tvec_true = synthetic_scene
    res = solve_pnp_pose(obj_pts, img_pts, intrinsics)
    assert res.success
    assert np.linalg.norm(res.tvec - tvec_true) < 1e-3
    assert _rotation_angle_deg(res.rvec, rvec_true) < 0.5
    assert res.mean_reproj_error_px < 1.0
    assert res.inliers is not None and len(res.inliers) == len(obj_pts)


def test_solve_pnp_rejects_outliers(synthetic_scene, intrinsics):
    obj_pts, img_pts, rvec_true, tvec_true = synthetic_scene
    noisy = img_pts.copy()
    rng = np.random.default_rng(3)
    n_out = int(0.3 * len(noisy))
    idx = rng.choice(len(noisy), n_out, replace=False)
    noisy[idx] += rng.uniform(60, 120, size=(n_out, 2))  # gross outliers
    res = solve_pnp_pose(obj_pts, noisy, intrinsics)
    assert res.success
    assert np.linalg.norm(res.tvec - tvec_true) < 5e-3
    assert _rotation_angle_deg(res.rvec, rvec_true) < 1.0


def test_solve_pnp_failure_contract(intrinsics):
    obj = np.random.default_rng(1).uniform(-0.2, 0.2, (6, 3)) + [0, 0, 1.0]
    img = np.random.default_rng(2).uniform(0, 640, (6, 2))
    with pytest.raises(ValueError, match=">= 4"):
        solve_pnp_pose(obj[:3], img[:3], intrinsics)
    bad = obj.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        solve_pnp_pose(bad, img, intrinsics)
    line = np.column_stack([np.linspace(0, 1, 6), np.zeros(6), np.ones(6)])
    with pytest.raises(ValueError, match="[Dd]egenerate"):
        solve_pnp_pose(line, img, intrinsics)


def test_solve_pnp_no_consensus_returns_failure(intrinsics):
    # Random 2D points unrelated to the 3D geometry -> RANSAC finds nothing.
    rng = np.random.default_rng(9)
    obj = np.column_stack(
        [rng.uniform(-0.2, 0.2, 10), rng.uniform(-0.2, 0.2, 10), rng.uniform(0.8, 1.2, 10)]
    )
    img = rng.uniform(0, 640, (10, 2))
    res = solve_pnp_pose(obj, img, intrinsics, ransac_reproj_threshold_px=2.0)
    assert res.success is False
    assert res.rvec is None and res.tvec is None


def test_backproject_math(intrinsics):
    depth = np.full((240, 320), 2.0, dtype=np.float32)
    p = backproject_to_3d(depth, 320.0, 240.0, intrinsics)
    np.testing.assert_allclose(p, [0.0, 0.0, 2.0], rtol=1e-6)
    p = backproject_pixel(400.0, 240.0, 2.0, intrinsics)
    np.testing.assert_allclose(p, [0.2, 0.0, 2.0], rtol=1e-6)
    with pytest.raises(ValueError):
        backproject_pixel(10.0, 10.0, -1.0, intrinsics)


def _synthetic_hand():
    """21 plausible 3D hand points (meters), wrist at origin, wrist->MCP9 = 0.090."""
    rng = np.random.default_rng(11)
    pts = rng.uniform(-0.05, 0.05, (21, 3))
    pts[0] = [0.0, 0.0, 0.0]
    pts[9] = [0.02, -0.085, 0.01]
    # Force the reference distance exactly.
    d = float(np.linalg.norm(pts[9] - pts[0]))
    pts = pts * (0.090 / d)
    return pts


def test_hand_pnp_refiner_recovers_metric_translation(intrinsics):
    hand = _synthetic_hand()
    rvec_true = np.array([[0.1], [0.2], [-0.05]])
    tvec_true = np.array([[0.05], [0.02], [0.60]])
    img_pts, _ = cv2.projectPoints(hand, rvec_true, tvec_true, intrinsics.matrix, None)
    # MediaPipe-style relative landmarks: arbitrary uniform scale (unknown to refiner).
    rel = (hand - hand[0]) * 1.7

    refiner = HandPnPRefiner(intrinsics)
    res = refiner.refine(img_pts.reshape(-1, 2), rel)
    assert res.success
    assert np.linalg.norm(res.tvec - tvec_true) < 5e-3
    assert _rotation_angle_deg(res.rvec, rvec_true) < 1.0


def test_hand_pnp_refiner_uses_shared_constant(intrinsics):
    from src.perception.pnp_pose import _default_hand_reference_m

    assert _default_hand_reference_m() == pytest.approx(0.090)
    refiner = HandPnPRefiner(intrinsics)
    assert refiner.reference_m == pytest.approx(0.090)


def test_refine_poses_pnp_post_stage(intrinsics):
    """End-to-end post-stage: runs, stores JSON-safe metadata, plausible depth.

    NOTE: unlike HandPnPRefiner.refine (independent 2D/3D inputs, exact), the
    post-stage builds its 3D model from the same normalized (x, y) + relative-z
    triplets MediaPipe provides, so the canonical model carries the documented
    anisotropic-scale approximation. We assert a sane metric ballpark (±30%),
    not exact recovery.
    """
    import json

    from src.datatypes import AnnotationFrame, HandLandmarks

    hand = _synthetic_hand()
    rvec_true = np.array([[0.0], [0.0], [0.0]])
    tvec_true = np.array([[0.0], [0.0], [0.50]])
    img_pts, _ = cv2.projectPoints(hand, rvec_true, tvec_true, intrinsics.matrix, None)
    img_pts = img_pts.reshape(-1, 2)

    W, H = 640, 480
    lm = HandLandmarks(
        x=(img_pts[:, 0] / W).astype(np.float64),
        y=(img_pts[:, 1] / H).astype(np.float64),
        z=(hand[:, 2] * 2.3).astype(np.float64),  # arbitrary relative-z scale
        confidence=0.9,
        handedness="Right",
    )
    frame = AnnotationFrame(frame_idx=0, timestamp=0.0, image_path="f.png", right_hand=lm)

    intr = CameraIntrinsics(fx=800.0, fy=800.0, cx=320.0, cy=240.0)
    n = refine_poses_pnp([frame], intr, W, H, enabled=True)
    assert n == 1
    entry = frame.metadata["hand_pnp"]["right"]
    assert entry["inliers"] >= 12  # majority consensus under the z-scale approximation
    assert abs(entry["tvec_m"][2] - 0.50) < 0.30 * 0.50  # metric ballpark
    json.dumps(frame.metadata)  # JSON-safe for exporters

    # Disabled -> no-op, never raises.
    frame2 = AnnotationFrame(frame_idx=1, timestamp=0.0, image_path="f.png")
    assert refine_poses_pnp([frame2], intr, W, H, enabled=False) == 0
    assert "hand_pnp" not in frame2.metadata
