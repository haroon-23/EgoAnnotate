"""Wrist pose extraction and workspace-mapped target pose generation.

Task 2 of the retargeting pipeline: extract the human wrist 3D position and
orientation from HandLandmarks, then map it into the robot's coordinate frame
and workspace scale.

MODELING CAVEATS (stated explicitly per project requirement):
  1. MediaPipe x/y are image-normalized [0,1] pixel projections.
     MediaPipe z is a MONOCULAR relative depth estimate — not metric.
     Absolute Cartesian reconstruction is therefore an approximation.
  2. Workspace normalization is a LINEAR SCALING heuristic that maps the
     observed human wrist bounding box to the robot's reachable workspace.
     It preserves motion shape (not metric accuracy).
  3. Wrist orientation is constructed from three keypoints (wrist, index MCP,
     pinky MCP) following the approach validated in action_computer.py.
     This gives the palm plane orientation only, not full wrist flexion/extension.

These caveats are recorded verbatim in every TargetPose.scaling_metadata dict
so downstream users can never misread the outputs as metric ground truth.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..datatypes import HandLandmarks

logger = logging.getLogger(__name__)


@dataclass
class PoseMapperConfig:
    """Configuration for PoseMapper.

    Attributes:
        preferred_hand: "right", "left", or "auto" (right preferred, left fallback).
        robot_workspace_bounds: Dict with keys "x_min", "x_max", "y_min", "y_max",
            "z_min", "z_max" in metres — the robot workspace the human trajectory
            is mapped into. Defaults target a tabletop task space.
        z_floor_m: Minimum z height for the end-effector target in metres,
            preventing the IK from targeting below-table poses.
        orientation_scale: Weight [0,1] for the wrist rotation component.
            1.0 = use full computed rotation. Reduce if IK becomes unstable.
    """
    preferred_hand: str = "right"
    robot_workspace_bounds: Dict[str, float] = field(default_factory=lambda: {
        "x_min": 0.30,  # in front of robot base, metres
        "x_max": 0.65,
        "y_min": -0.30,
        "y_max": 0.30,
        "z_min": 0.20,  # above table
        "z_max": 0.65,
    })
    z_floor_m: float = 0.10
    orientation_scale: float = 1.0


@dataclass
class TargetPose:
    """End-effector target pose for one video frame.

    Attributes:
        frame_idx: Source frame index.
        timestamp: Source frame timestamp in seconds.
        position: Target position in robot base frame, metres, shape (3,).
        quaternion: Target orientation as [qx, qy, qz, qw], shape (4,).
        hand_detected: Whether a valid hand was found in this frame.
        hand_used: "left", "right", or None.
        is_interpolated: True if hand landmarks were gap-interpolated.
        scaling_metadata: Dict documenting the mapping approximation.
    """
    frame_idx: int
    timestamp: float
    position: np.ndarray          # shape (3,)
    quaternion: np.ndarray        # shape (4,) — [qx, qy, qz, qw]
    hand_detected: bool
    hand_used: Optional[str]
    is_interpolated: bool
    scaling_metadata: Dict        # audit trail for the workspace approximation


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a 3×3 rotation matrix to a quaternion [qx, qy, qz, qw].

    Uses Shepperd's method for numerical stability.
    """
    # Ensure orthonormal
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm > 1e-9:
        q /= norm
    return q


def _safe_normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a 3D vector; return [0,0,1] if near-zero."""
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return v / n


def _wrist_orientation_from_landmarks(hand: HandLandmarks) -> np.ndarray:
    """Construct a 3×3 rotation matrix from wrist keypoints.

    Uses wrist (0), index MCP (5), pinky MCP (17) to define the palm plane —
    same three-point frame used in action_computer.py (validated in this repo).

    The resulting frame is:
        x_axis: wrist → index MCP (finger spread direction)
        z_axis: palm normal (cross product of x and wrist→pinky MCP)
        y_axis: z × x (completes right-handed frame)

    Returns:
        R: np.ndarray of shape (3, 3) — rotation matrix from hand frame to
           camera frame.
    """
    p_wrist = np.array([hand.x[0], hand.y[0], hand.z[0]], dtype=np.float64)
    p_index_mcp = np.array([hand.x[5], hand.y[5], hand.z[5]], dtype=np.float64)
    p_pinky_mcp = np.array([hand.x[17], hand.y[17], hand.z[17]], dtype=np.float64)

    v_index = p_index_mcp - p_wrist
    v_pinky = p_pinky_mcp - p_wrist

    x_axis = _safe_normalize(v_index)
    z_axis = _safe_normalize(np.cross(x_axis, v_pinky))
    y_axis = np.cross(z_axis, x_axis)  # already unit-length if x,z orthonormal

    # Columns: [x_axis, y_axis, z_axis]
    R = np.column_stack([x_axis, y_axis, z_axis])
    return R


class PoseMapper:
    """Converts HandLandmarks to workspace-mapped end-effector target poses.

    Usage::

        mapper = PoseMapper(config)
        targets = mapper.map_frames(frames)
    """

    def __init__(self, config: PoseMapperConfig) -> None:
        self.config = config
        self._wrist_traj: Optional[np.ndarray] = None   # cache for normalization

    def map_frames(
        self,
        frames: List,  # List[AnnotationFrame] — avoid circular import
    ) -> List[TargetPose]:
        """Map a sequence of AnnotationFrames to target end-effector poses.

        Two-pass algorithm:
          1. Collect wrist positions across all frames.
          2. Compute normalization bounds, then map each frame.

        Args:
            frames: Ordered list of AnnotationFrame objects from the pipeline.

        Returns:
            List of TargetPose, one per input frame.
        """
        # --- Pass 1: collect raw wrist positions ----------------------------
        raw_wrists: List[Optional[np.ndarray]] = []
        for frame in frames:
            hand, _ = self._select_hand(frame)
            if hand is not None:
                raw_wrists.append(np.array(
                    [hand.x[0], hand.y[0], hand.z[0]], dtype=np.float64
                ))
            else:
                raw_wrists.append(None)

        # --- Compute normalization bounds from all detected wrists ----------
        detected = [w for w in raw_wrists if w is not None]
        if detected:
            wrist_arr = np.stack(detected, axis=0)  # (N_detected, 3)
            human_min = wrist_arr.min(axis=0)
            human_max = wrist_arr.max(axis=0)
        else:
            # Fallback: unit cube
            human_min = np.zeros(3)
            human_max = np.ones(3)

        # Clamp range to avoid division by zero if wrist barely moves
        human_range = human_max - human_min
        human_range = np.where(human_range > 1e-6, human_range, 1.0)

        wb = self.config.robot_workspace_bounds
        robot_min = np.array([wb["x_min"], wb["y_min"], wb["z_min"]], dtype=np.float64)
        robot_max = np.array([wb["x_max"], wb["y_max"], wb["z_max"]], dtype=np.float64)
        robot_range = robot_max - robot_min

        scaling_metadata_base = {
            "method": "linear_workspace_normalization",
            "caveat": (
                "MediaPipe z is monocular relative depth, not metric. "
                "Workspace mapping is a linear scaling approximation preserving "
                "motion shape, NOT metric accuracy. Treat as retargeted motion "
                "style demonstration only."
            ),
            "human_wrist_bbox_min": human_min.tolist(),
            "human_wrist_bbox_max": human_max.tolist(),
            "robot_workspace_min_m": robot_min.tolist(),
            "robot_workspace_max_m": robot_max.tolist(),
        }

        # --- Pass 2: map each frame -----------------------------------------
        targets: List[TargetPose] = []
        for frame, raw_wrist in zip(frames, raw_wrists):
            hand, side = self._select_hand(frame)
            is_interp = hand.is_interpolated if hand is not None else False

            if raw_wrist is None or hand is None:
                # No hand detected: use a safe default position (robot home)
                pos = (robot_min + robot_max) / 2.0
                q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
                targets.append(TargetPose(
                    frame_idx=frame.frame_idx,
                    timestamp=frame.timestamp,
                    position=pos,
                    quaternion=q,
                    hand_detected=False,
                    hand_used=None,
                    is_interpolated=is_interp,
                    scaling_metadata={
                        **scaling_metadata_base,
                        "frame_idx": frame.frame_idx,
                        "fallback": "no_hand_detected",
                    },
                ))
                continue

            # Normalize wrist position to robot workspace
            alpha = (raw_wrist - human_min) / human_range  # [0,1]^3
            pos = robot_min + alpha * robot_range            # robot metres

            # Enforce z floor
            pos[2] = max(pos[2], self.config.z_floor_m)

            # Construct wrist orientation
            R_hand = _wrist_orientation_from_landmarks(hand)  # (3,3)

            # Apply orientation scale (blend toward identity)
            if self.config.orientation_scale < 1.0:
                s = self.config.orientation_scale
                R_hand = (1.0 - s) * np.eye(3) + s * R_hand
                # Re-orthonormalize via SVD
                U, _, Vt = np.linalg.svd(R_hand)
                R_hand = U @ Vt

            q = _rotation_matrix_to_quaternion(R_hand)

            targets.append(TargetPose(
                frame_idx=frame.frame_idx,
                timestamp=frame.timestamp,
                position=pos,
                quaternion=q,
                hand_detected=True,
                hand_used=side,
                is_interpolated=is_interp,
                scaling_metadata={
                    **scaling_metadata_base,
                    "frame_idx": frame.frame_idx,
                    "raw_wrist_normalized": alpha.tolist(),
                },
            ))

        return targets

    def _select_hand(self, frame) -> Tuple[Optional[HandLandmarks], Optional[str]]:
        """Return (hand, side) according to preferred_hand config."""
        pref = self.config.preferred_hand.lower()
        if pref == "right":
            if frame.right_hand is not None:
                return frame.right_hand, "right"
            if frame.left_hand is not None:
                return frame.left_hand, "left"
        elif pref == "left":
            if frame.left_hand is not None:
                return frame.left_hand, "left"
            if frame.right_hand is not None:
                return frame.right_hand, "right"
        else:  # "auto" — right preferred
            if frame.right_hand is not None:
                return frame.right_hand, "right"
            if frame.left_hand is not None:
                return frame.left_hand, "left"
        return None, None
