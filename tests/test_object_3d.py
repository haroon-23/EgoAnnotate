"""Tests for src/perception/depth.localize_objects_3d — synthetic depth, analytic values."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.perception.depth import localize_objects_3d
from src.perception.detector import Detection
from src.perception.pnp_pose import CameraIntrinsics


@pytest.fixture
def intrinsics():
    return CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0)


def _det(box):
    return Detection(bbox_xyxy_norm=np.array(box, dtype=np.float32), label="mug", score=0.9)


def test_box_center_median_depth(intrinsics):
    # Depth gradient: z = 1.0 + 0.01 * x  (meters).
    yy, xx = np.mgrid[0:100, 0:100]
    depth = (1.0 + 0.01 * xx).astype(np.float32)
    pts = localize_objects_3d([_det([0.2, 0.3, 0.4, 0.5])], depth, intrinsics)
    assert len(pts) == 1 and pts[0] is not None
    # Box -> px [20,30,40,50]; center (30,40); 3x3 median over x in {29,30,31} -> 1.30 m.
    np.testing.assert_allclose(pts[0], [-0.26, -0.13, 1.30], rtol=1e-4)


def test_mask_median_ignores_background(intrinsics):
    depth = np.full((100, 100), 5.0, dtype=np.float32)  # background far
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:50, 20:40] = True
    depth[mask] = 2.0  # object near
    pts = localize_objects_3d([_det([0.2, 0.3, 0.4, 0.5])], depth, intrinsics, masks=[mask])
    assert pts[0] is not None
    assert pts[0][2] == pytest.approx(2.0)  # median over mask, not background
    # Back-projected at the mask centroid (u=29.5, v=39.5).
    np.testing.assert_allclose(
        pts[0], [(29.5 - 50) * 2.0 / 100, (39.5 - 50) * 2.0 / 100, 2.0], rtol=1e-6
    )


def test_invalid_depth_returns_none(intrinsics):
    depth = np.zeros((100, 100), dtype=np.float32)  # no valid samples
    pts = localize_objects_3d([_det([0.2, 0.3, 0.4, 0.5])], depth, intrinsics)
    assert pts == [None]


def test_empty_detections(intrinsics):
    depth = np.ones((10, 10), dtype=np.float32)
    assert localize_objects_3d([], depth, intrinsics) == []
