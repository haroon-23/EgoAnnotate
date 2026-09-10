"""Trajectory smoothing for wrist target poses before Inverse Kinematics.

Applies Savitzky-Golay filtering to 3D position trajectories and rotation vector
representations of quaternions. Respects tracking loss gaps (is_interpolated / no hand)
so continuous smoothing is never applied across fabricated or missing data.

RATIONALE FOR DEFAULT PARAMETERS (window_length=9, polyorder=2):
  - At 30 FPS, a 9-frame window spans 300 ms (0.3s). This effectively suppresses
    high-frequency MediaPipe keypoint jitter (10-30 Hz noise) while preserving 2nd order
    motion dynamics (smooth accelerations/decelerations) without over-smoothing fast
    human hand maneuvers.
  - Quaternions are converted to continuous 3D rotation vectors (axis-angle representation)
    before filtering, then converted back and normalized. This avoids quaternion sign
    flipping (q == -q ambiguity) and norm degradation associated with naive component filtering.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

from .pose_mapper import TargetPose

logger = logging.getLogger(__name__)


@dataclass
class TrajectorySmootherConfig:
    """Configuration for TrajectorySmoother.

    Attributes:
        enabled: Whether trajectory smoothing is active.
        window_length: Length of the Savitzky-Golay filter window (must be odd).
        polyorder: Order of the polynomial used to fit the samples.
        respect_gaps: If True, smooths independent contiguous segments separated by
            tracking loss or interpolation gaps.
    """
    enabled: bool = True
    window_length: int = 9
    polyorder: int = 2
    respect_gaps: bool = True


class TrajectorySmoother:
    """Smooths TargetPose trajectories prior to IK solving."""

    def __init__(self, config: TrajectorySmootherConfig) -> None:
        self.config = config

    def smooth_poses(self, target_poses: List[TargetPose]) -> List[TargetPose]:
        """Smooth positions and orientations across a sequence of TargetPose objects.

        Args:
            target_poses: List of TargetPose objects from PoseMapper.

        Returns:
            A new list of TargetPose objects with smoothed positions and quaternions.
        """
        if not self.config.enabled or len(target_poses) < 3:
            return target_poses

        # Copy target poses so we do not mutate inputs in-place
        smoothed_poses = [
            TargetPose(
                frame_idx=p.frame_idx,
                timestamp=p.timestamp,
                position=p.position.copy(),
                quaternion=p.quaternion.copy(),
                hand_detected=p.hand_detected,
                hand_used=p.hand_used,
                is_interpolated=p.is_interpolated,
                scaling_metadata=dict(p.scaling_metadata),
            )
            for p in target_poses
        ]

        if not self.config.respect_gaps:
            # Smooth all poses together in one block
            self._smooth_segment(smoothed_poses, 0, len(smoothed_poses))
            return smoothed_poses

        # Partition poses into contiguous segments of valid tracking (hand_detected=True and not is_interpolated)
        segments = self._find_valid_segments(smoothed_poses)
        for start_idx, end_idx in segments:
            if end_idx - start_idx >= 3:
                self._smooth_segment(smoothed_poses, start_idx, end_idx)

        return smoothed_poses

    def _find_valid_segments(self, poses: List[TargetPose]) -> List[Tuple[int, int]]:
        """Identify contiguous runs of valid, non-interpolated hand poses."""
        segments = []
        in_segment = False
        start_idx = 0

        for i, pose in enumerate(poses):
            # Valid detection: hand detected and not gap-interpolated
            is_valid = pose.hand_detected and not pose.is_interpolated
            if is_valid and not in_segment:
                in_segment = True
                start_idx = i
            elif not is_valid and in_segment:
                in_segment = False
                segments.append((start_idx, i))

        if in_segment:
            segments.append((start_idx, len(poses)))

        return segments

    def _smooth_segment(self, poses: List[TargetPose], start_idx: int, end_idx: int) -> None:
        """Smooth a contiguous slice poses[start_idx:end_idx]."""
        n_samples = end_idx - start_idx
        if n_samples < 3:
            return

        # Determine effective window length and polyorder for this segment length
        win = self.config.window_length
        poly = self.config.polyorder

        # Window must be odd and <= n_samples
        if win > n_samples:
            win = n_samples if n_samples % 2 != 0 else n_samples - 1
        if win <= poly:
            poly = max(1, win - 1)

        if win < 3 or poly < 1:
            return

        # --- Position smoothing ---------------------------------------------
        pos_arr = np.array([poses[i].position for i in range(start_idx, end_idx)], dtype=np.float64)  # (N, 3)
        pos_smoothed = savgol_filter(pos_arr, window_length=win, polyorder=poly, axis=0)

        # --- Orientation smoothing via rotation vectors ---------------------
        quat_arr = np.array([poses[i].quaternion for i in range(start_idx, end_idx)], dtype=np.float64)  # (N, 4)

        # Ensure quaternion sign continuity: if q_k dot q_{k-1} < 0, negate q_k
        for k in range(1, len(quat_arr)):
            if np.dot(quat_arr[k], quat_arr[k - 1]) < 0.0:
                quat_arr[k] = -quat_arr[k]

        # Convert quaternions [qx, qy, qz, qw] to Rotation objects and then rotvecs
        rot_seq = R.from_quat(quat_arr)
        rotvecs = rot_seq.as_rotvec()  # (N, 3)

        # Smooth rotation vector sequence
        rotvecs_smoothed = savgol_filter(rotvecs, window_length=win, polyorder=poly, axis=0)

        # Convert back to normalized quaternions
        quat_smoothed = R.from_rotvec(rotvecs_smoothed).as_quat()  # (N, 4)
        norms = np.linalg.norm(quat_smoothed, axis=1, keepdims=True)
        norms = np.where(norms > 1e-9, norms, 1.0)
        quat_smoothed = quat_smoothed / norms

        # Update poses in-place
        for idx_rel, idx_abs in enumerate(range(start_idx, end_idx)):
            poses[idx_abs].position = pos_smoothed[idx_rel]
            poses[idx_abs].quaternion = quat_smoothed[idx_rel]
            poses[idx_abs].scaling_metadata["trajectory_smoothed"] = True
            poses[idx_abs].scaling_metadata["smoothing_window"] = win
            poses[idx_abs].scaling_metadata["smoothing_polyorder"] = poly
