"""Metric hand lift: synthetic ground truth through a known pinhole camera."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datatypes import AnnotationFrame, HandLandmarks
from src.perception.metric_hand_pose import camera_matrix, hand_frame_rotation, lift_frames, lift_hand

W, H = 1280, 720  # deliberately non-square
K_TRUE = np.array([[900.0, 0, 610.0], [0, 905.0, 355.0], [0, 0, 1.0]])


def hand_model_H(side: str) -> np.ndarray:
    """21 landmarks in the hand frame (x to middle MCP, z out of the palm).

    Right hand: thumb/index on -y. The left hand is the mirror image (y -> -y).
    Fingers curl slightly toward the palm so the point set is not planar.
    """
    p = np.zeros((21, 3))
    p[1] = [0.02, -0.025, 0.012]
    p[2] = [0.045, -0.04, 0.02]
    p[3] = [0.065, -0.05, 0.03]
    p[4] = [0.085, -0.058, 0.04]
    mcp_y = {5: -0.03, 9: 0.0, 13: 0.018, 17: 0.038}
    mcp_x = {5: 0.095, 9: 0.097, 13: 0.09, 17: 0.08}
    seg = {5: 0.04, 9: 0.045, 13: 0.042, 17: 0.032}
    for base, y in mcp_y.items():
        x0 = mcp_x[base]
        for k in range(4):
            reach = x0 + (0.0 if k == 0 else seg[base] * (1.0 + 0.6 * (k - 1)) * 0.55 * k)
            p[base + k] = [reach, y, 0.004 * k * k]
    if side == "left":
        p[:, 1] *= -1
    return p


def look_rotation(rvec_extra) -> np.ndarray:
    """Hand frame -> camera: palm toward camera, fingers up, then a tilt."""
    base = np.array([[0.0, -1, 0], [-1, 0, 0], [0, 0, -1]])
    extra, _ = cv2.Rodrigues(np.asarray(rvec_extra, dtype=np.float64))
    return extra @ base


def make_hand(side, R_true, t_true, K=K_TRUE, dist=None, noise_px=0.0, seed=0, label=None):
    """Hand at a known pose projected through K; world landmarks in an arbitrary frame."""
    pH = hand_model_H(side)
    cam = pH @ R_true.T + t_true
    rvec0 = np.zeros(3)
    px, _ = cv2.projectPoints(cam, rvec0, rvec0, K, dist)
    px = px.reshape(-1, 2)
    if noise_px:
        px = px + np.random.default_rng(seed).normal(0, noise_px, px.shape)
    # MediaPipe-style world landmarks: centred on the hand, in a different orientation.
    Q, _ = cv2.Rodrigues(np.array([0.3, -0.2, 0.5]))
    world = (pH - pH.mean(axis=0)) @ Q.T
    return HandLandmarks(
        x=px[:, 0] / W, y=px[:, 1] / H, z=np.zeros(21), confidence=0.9,
        handedness=label or side.capitalize(), world_xyz=world, image_size=(W, H),
    )


def rot_err_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


POSES = [
    ([0.0, 0.0, 0.0], [0.05, -0.02, 0.45]),
    ([0.4, -0.3, 0.2], [-0.10, 0.04, 0.60]),
    ([-0.5, 0.2, -0.3], [0.12, 0.08, 0.35]),
    ([0.0, 3.0, 0.0], [0.0, 0.0, 0.50]),  # back of the hand toward the camera
]


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("rv,t", POSES)
def test_recovers_known_pose(side, rv, t):
    R_true, t_true = look_rotation(rv), np.array(t)
    hand = make_hand(side, R_true, t_true)
    pose = lift_hand(hand, K_TRUE, np.zeros(5), "calibrated")
    assert pose is not None
    assert np.linalg.norm(pose.wrist_position_m - t_true) < 0.005
    assert rot_err_deg(pose.wrist_rotation, R_true) < 2.0
    assert pose.reprojection_error_px < 0.1
    assert pose.intrinsics_source == "calibrated"
    assert not pose.is_interpolated
    np.testing.assert_allclose(pose.keypoints_m[0], pose.wrist_position_m)


@pytest.mark.parametrize("side", ["left", "right"])
def test_palm_normal_points_out_of_palm_for_both_hands(side):
    """+z of the hand frame is the same anatomical direction for both hands."""
    pH = hand_model_H(side)
    R = hand_frame_rotation(pH, side)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-9)
    # A left hand labelled as right gets the opposite palm normal (mirror rule).
    wrong = hand_frame_rotation(pH, "right" if side == "left" else "left")
    assert wrong[2, 2] < 0


def test_noise_keeps_pose_within_tolerance():
    R_true, t_true = look_rotation([0.2, 0.1, 0.0]), np.array([0.0, 0.0, 0.5])
    hand = make_hand("right", R_true, t_true, noise_px=0.5)
    pose = lift_hand(hand, K_TRUE, np.zeros(5), "calibrated")
    assert np.linalg.norm(pose.wrist_position_m - t_true) < 0.01
    assert rot_err_deg(pose.wrist_rotation, R_true) < 5.0


def test_focal_length_error_scales_depth_proportionally():
    """Documents the sensitivity: +10% focal error gives ~+10% depth."""
    R_true, t_true = look_rotation([0.1, 0.0, 0.0]), np.array([0.02, 0.0, 0.50])
    hand = make_hand("right", R_true, t_true)
    K_wrong = K_TRUE.copy()
    K_wrong[0, 0] *= 1.10
    K_wrong[1, 1] *= 1.10
    z_ok = lift_hand(hand, K_TRUE, np.zeros(5), "calibrated").wrist_position_m[2]
    z_bad = lift_hand(hand, K_wrong, np.zeros(5), "assumed_fov").wrist_position_m[2]
    assert z_ok == pytest.approx(0.50, abs=0.005)
    assert z_bad / z_ok == pytest.approx(1.10, abs=0.03)


def test_hand_scale_scales_depth():
    R_true, t_true = look_rotation([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.50])
    hand = make_hand("left", R_true, t_true)
    z1 = lift_hand(hand, K_TRUE, np.zeros(5), "calibrated").wrist_position_m[2]
    z2 = lift_hand(hand, K_TRUE, np.zeros(5), "calibrated", hand_scale=1.2).wrist_position_m[2]
    assert z2 / z1 == pytest.approx(1.2, abs=0.03)


def test_distortion_is_modelled():
    dist = np.array([-0.25, 0.08, 0.001, -0.001, 0.0])
    R_true, t_true = look_rotation([0.2, -0.1, 0.0]), np.array([0.15, 0.08, 0.40])
    hand = make_hand("right", R_true, t_true, dist=dist)
    with_dist = lift_hand(hand, K_TRUE, dist, "calibrated")
    without = lift_hand(hand, K_TRUE, np.zeros(5), "calibrated")
    assert np.linalg.norm(with_dist.wrist_position_m - t_true) < 0.005
    assert with_dist.reprojection_error_px < 0.1
    assert without.reprojection_error_px > 10 * with_dist.reprojection_error_px


def test_none_when_world_missing_or_error_too_large():
    R_true, t_true = look_rotation([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.5])
    hand = make_hand("right", R_true, t_true)
    no_world = HandLandmarks(x=hand.x, y=hand.y, z=hand.z, confidence=0.9, handedness="Right",
                             is_interpolated=True, world_xyz=None, image_size=(W, H))
    assert lift_hand(no_world, K_TRUE, np.zeros(5), "calibrated") is None

    bad = make_hand("right", R_true, t_true)
    bad.x = bad.x.copy()
    bad.x[[4, 8, 12, 20]] += 0.08  # tips wildly inconsistent with the world model
    assert lift_hand(bad, K_TRUE, np.zeros(5), "calibrated", max_reproj_px=3.0) is None
    assert lift_hand(bad, K_TRUE, np.zeros(5), "calibrated") is not None


def test_interpolated_input_is_flagged():
    R_true, t_true = look_rotation([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.5])
    hand = make_hand("right", R_true, t_true)
    hand.is_interpolated = True
    assert lift_hand(hand, K_TRUE, np.zeros(5), "calibrated").is_interpolated is True


def test_missing_image_size_raises():
    R_true, t_true = look_rotation([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.5])
    hand = make_hand("right", R_true, t_true)
    hand.image_size = None
    with pytest.raises(ValueError):
        lift_hand(hand, K_TRUE, np.zeros(5), "calibrated")


# --- camera_matrix ---------------------------------------------------------


def test_camera_matrix_assumed_fov_uses_width_and_centre():
    K, dist, src = camera_matrix({"camera": {"assumed_hfov_deg": 90}}, (1280, 720))
    assert src == "assumed_fov"
    assert K[0, 0] == pytest.approx(640.0)
    assert K[1, 1] == pytest.approx(640.0)  # square pixels
    assert (K[0, 2], K[1, 2]) == (640.0, 360.0)
    assert not np.any(dist)


def test_camera_matrix_default_fov_when_unconfigured():
    K, _, src = camera_matrix({}, (640, 480))
    assert src == "assumed_fov"
    assert K[0, 0] == pytest.approx(320.0 / np.tan(np.radians(35.0)))


def test_camera_matrix_calibrated_is_rescaled_from_calib_size():
    cfg = {"camera": {"fx": 1000.0, "fy": 1010.0, "cx": 960.0, "cy": 540.0,
                      "calib_size": [1920, 1080], "dist_coeffs": [-0.1, 0.02, 0, 0]}}
    K, dist, src = camera_matrix(cfg, (960, 540))
    assert src == "calibrated"
    np.testing.assert_allclose([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], [500.0, 505.0, 480.0, 270.0])
    np.testing.assert_allclose(dist, [-0.1, 0.02, 0, 0])


def test_camera_matrix_rejects_missing_calib_size_and_aspect_mismatch():
    cam = {"fx": 1000.0, "fy": 1000.0, "cx": 960.0, "cy": 540.0}
    with pytest.raises(ValueError, match="calib_size"):
        camera_matrix({"camera": cam}, (960, 540))
    with pytest.raises(ValueError, match="aspect"):
        camera_matrix({"camera": dict(cam, calib_size=[1920, 1080])}, (224, 224))


# --- lift_frames -----------------------------------------------------------


def test_lift_frames_fills_both_hands_and_summarises():
    R_true, t_true = look_rotation([0.1, 0.0, 0.0]), np.array([0.0, 0.0, 0.5])
    cfg = {"camera": {"fx": K_TRUE[0, 0], "fy": K_TRUE[1, 1], "cx": K_TRUE[0, 2], "cy": K_TRUE[1, 2],
                      "calib_size": [W, H]}}
    frames = []
    for i in range(3):
        fr = AnnotationFrame(frame_idx=i, timestamp=i / 10, image_path="x.png")
        fr.left_hand = make_hand("left", R_true, t_true + [-0.1, 0, 0])
        if i != 1:
            fr.right_hand = make_hand("right", R_true, t_true + [0.1, 0, 0])
        frames.append(fr)
    frames[2].left_hand = HandLandmarks(
        x=frames[2].left_hand.x, y=frames[2].left_hand.y, z=np.zeros(21), confidence=0.5,
        handedness="Left", is_interpolated=True, world_xyz=None, image_size=(W, H))
    frames[0].right_hand_metric = "stale"  # must not survive

    summary = lift_frames(frames, cfg)
    assert summary["intrinsics_source"] == "calibrated"
    assert summary["left"] == {"tracked": 3, "lifted": 2, "failures": {"no_world_landmarks": 1}}
    assert summary["right"]["tracked"] == 2 and summary["right"]["lifted"] == 2
    assert summary["median_reprojection_error_px"] < 0.1
    assert frames[0].left_hand_metric is not None and frames[0].right_hand_metric.intrinsics_source == "calibrated"
    assert frames[1].right_hand_metric is None
    assert frames[2].left_hand_metric is None
    assert np.linalg.norm(frames[0].right_hand_metric.wrist_position_m - (t_true + [0.1, 0, 0])) < 0.005


def test_lift_frames_uses_assumed_fov_and_says_so():
    R_true, t_true = look_rotation([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.5])
    fr = AnnotationFrame(frame_idx=0, timestamp=0.0, image_path="x.png",
                         right_hand=make_hand("right", R_true, t_true))
    summary = lift_frames([fr], {})
    assert summary["intrinsics_source"] == "assumed_fov"
    assert fr.right_hand_metric.intrinsics_source == "assumed_fov"
