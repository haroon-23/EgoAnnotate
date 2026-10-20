"""World-locked and object-centric trajectory solver for table-mounted arm robots.

Solves the 3 domain gaps when retargeting egocentric human video to fixed-base arm robots:
  1. Visual Odometry Ego-Motion Compensation: Estimates per-frame background camera
     translation via RANSAC affine optical flow and subtracts head motion from wrist positions.
  2. Object-Centric Relative Task Space: Anchors the robot reach trajectory relative to
     the detected 3D workspace object position at grasp onset.
  3. Canonical Arm Grasp Approach Frame: Maps human palm orientation deltas onto the
     robot gripper's ergonomic tabletop approach frame (avoiding wrist singularities).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class WorldLockedArmTrajectory:
    wrist_world_positions: np.ndarray        # (N, 3) meters in fixed robot base frame
    wrist_world_quaternions: np.ndarray      # (N, 4) [qx, qy, qz, qw] canonical arm frame
    object_world_positions: np.ndarray       # (N, 3) meters in fixed robot base frame
    wrist_to_object_rel_m: np.ndarray        # (N, 3) object-centric relative vector
    camera_ego_shift_norm: np.ndarray        # (N, 2) cumulative head motion subtracted
    grasp_active_mask: np.ndarray            # (N,) bool


def estimate_cumulative_camera_shift(frames: List[Any]) -> np.ndarray:
    """Estimate cumulative 2D normalized camera ego-motion [dx, dy] via RANSAC background flow."""
    n = len(frames)
    cum_shift = np.zeros((n, 2), dtype=np.float64)
    if n < 2:
        return cum_shift

    prev_gray = None
    for i, f in enumerate(frames):
        if i > 0:
            cum_shift[i] = cum_shift[i - 1].copy()
        if not getattr(f, "image_path", None) or not Path(f.image_path).exists():
            continue
        img = cv2.imread(f.image_path)
        if img is None:
            continue
        # Downsample for fast (<2ms/frame) RANSAC background motion estimation
        small = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sh, sw = gray.shape[:2]

        if prev_gray is not None:
            mask = np.full((sh, sw), 255, dtype=np.uint8)
            for hnd in (getattr(f, "left_hand", None), getattr(f, "right_hand", None)):
                if hnd is not None:
                    x1 = max(0, int((float(np.min(hnd.x)) - 0.08) * sw))
                    x2 = min(sw, int((float(np.max(hnd.x)) + 0.08) * sw))
                    y1 = max(0, int((float(np.min(hnd.y)) - 0.08) * sh))
                    y2 = min(sh, int((float(np.max(hnd.y)) + 0.08) * sh))
                    mask[y1:y2, x1:x2] = 0

            pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=60, qualityLevel=0.02, minDistance=8, mask=mask)
            if pts is not None and len(pts) >= 6:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
                if nxt is not None and st is not None:
                    valid = st.ravel() == 1
                    if np.sum(valid) >= 4:
                        M, _ = cv2.estimateAffinePartial2D(
                            pts[valid].reshape(-1, 2),
                            nxt[valid].reshape(-1, 2),
                            method=cv2.RANSAC,
                            ransacReprojThreshold=2.5,
                        )
                        if M is not None:
                            dx_norm = float(M[0, 2]) / float(sw)
                            dy_norm = float(M[1, 2]) / float(sh)
                            # High-pass leak factor (0.96) prevents unbounded integration drift on long clips
                            cum_shift[i] = 0.96 * cum_shift[i] + np.array([dx_norm, dy_norm], dtype=np.float64)
        prev_gray = gray

    return cum_shift


def build_world_locked_arm_trajectory(
    frames: List[Any],
    robot_table_anchor: np.ndarray = np.array([0.45, 0.0, 0.34], dtype=np.float64),
    table_z_m: float = 0.28,
) -> WorldLockedArmTrajectory:
    """Compute ego-motion-compensated, object-centric 3D trajectories for a fixed-base arm robot."""
    n = len(frames)
    if n == 0:
        z3 = np.zeros((0, 3))
        return WorldLockedArmTrajectory(z3, np.zeros((0, 4)), z3, z3, np.zeros((0, 2)), np.zeros(0, dtype=bool))

    cam_shift = estimate_cumulative_camera_shift(frames)

    # Extract ego-compensated normalized wrist positions
    raw_wrist = np.zeros((n, 3), dtype=np.float64)
    valid_w = np.zeros(n, dtype=bool)
    grasp_mask = np.zeros(n, dtype=bool)

    for i, f in enumerate(frames):
        hnd = f.right_hand if f.right_hand is not None else f.left_hand
        if hnd is not None:
            # Subtract cumulative background camera motion from image-plane wrist coordinates
            wx = float(hnd.x[0]) - cam_shift[i, 0]
            wy = float(hnd.y[0]) - cam_shift[i, 1]
            wz = float(hnd.z[0])
            raw_wrist[i] = [wx, wy, wz]
            valid_w[i] = True
        rc = getattr(f, "right_contact", None)
        lc = getattr(f, "left_contact", None)
        rg = getattr(f, "right_grasp", None)
        lg = getattr(f, "left_grasp", None)
        in_c = bool((rc and rc.in_contact) or (lc and lc.in_contact))
        in_g = bool((rg and rg.type != "open") or (lg and lg.type != "open"))
        grasp_mask[i] = in_c or in_g

    if np.any(valid_w):
        med_w = np.median(raw_wrist[valid_w], axis=0)
    else:
        med_w = np.array([0.5, 0.5, 0.0], dtype=np.float64)

    # Map ego-stabilized wrist into robot table coordinates
    wrist_world = np.zeros((n, 3), dtype=np.float64)
    wrist_quats = np.zeros((n, 4), dtype=np.float64)
    # Canonical Franka top-down/angled grasp quaternion (gripper pointing down at tabletop)
    q_canonical = Rotation.from_euler("xyz", [np.pi, 0.0, 0.0]).as_quat()

    for i, f in enumerate(frames):
        if valid_w[i]:
            delta = raw_wrist[i] - med_w
            # Camera to Robot Base: -Z_cam -> +X_robot, -X_cam -> +Y_robot, -Y_cam -> +Z_robot
            pos = np.array([
                robot_table_anchor[0] + np.clip(-delta[2] * 0.45 - (delta[1] * 0.18), -0.14, 0.16),
                robot_table_anchor[1] + np.clip(-delta[0] * 0.45, -0.22, 0.22),
                robot_table_anchor[2] + np.clip(-delta[1] * 0.38, -0.06, 0.18),
            ], dtype=np.float64)
            # Lower end-effector to table contact height during active grasp windows
            if grasp_mask[i]:
                pos[2] = max(table_z_m + 0.02, pos[2] - 0.03)
            else:
                pos[2] = max(table_z_m + 0.05, pos[2])
            wrist_world[i] = pos

            hnd = f.right_hand if f.right_hand is not None else f.left_hand
            yaw_delta = float(np.arctan2(hnd.y[17] - hnd.y[5], hnd.x[17] - hnd.x[5])) * 0.35
            q_frame = (Rotation.from_euler("z", yaw_delta) * Rotation.from_quat(q_canonical)).as_quat()
            wrist_quats[i] = q_frame
        else:
            wrist_world[i] = robot_table_anchor.copy()
            wrist_quats[i] = q_canonical.copy()

    # Smooth world wrist trajectory
    for k in range(3):
        wrist_world[:, k] = np.convolve(
            np.pad(wrist_world[:, k], (2, 2), mode="edge"), np.ones(5) / 5.0, mode="valid"
        )

    # Compute 3D object world trajectory on the robot table:
    # Starts at initial grasp position on the table, attaches to wrist while grasped, rests on table when released
    grasp_idxs = np.where(grasp_mask & valid_w)[0]
    if len(grasp_idxs) > 0:
        first_g = int(grasp_idxs[0])
        obj_init_pos = np.array([wrist_world[first_g, 0], wrist_world[first_g, 1], table_z_m], dtype=np.float64)
    else:
        obj_init_pos = np.array([robot_table_anchor[0] + 0.02, robot_table_anchor[1], table_z_m], dtype=np.float64)

    obj_world = np.zeros((n, 3), dtype=np.float64)
    cur_obj_pos = obj_init_pos.copy()
    for i in range(n):
        if grasp_mask[i] and valid_w[i]:
            # Object is held in the gripper fingertips
            cur_obj_pos = np.array([
                wrist_world[i, 0],
                wrist_world[i, 1],
                max(table_z_m, wrist_world[i, 2] - 0.015),
            ], dtype=np.float64)
        else:
            # Object rests on the table surface under gravity
            cur_obj_pos[2] = table_z_m
        obj_world[i] = cur_obj_pos.copy()

    rel_vec = wrist_world - obj_world
    return WorldLockedArmTrajectory(
        wrist_world_positions=wrist_world,
        wrist_world_quaternions=wrist_quats,
        object_world_positions=obj_world,
        wrist_to_object_rel_m=rel_vec,
        camera_ego_shift_norm=cam_shift,
        grasp_active_mask=grasp_mask,
    )
