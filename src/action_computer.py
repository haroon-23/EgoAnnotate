"""Action primitive computation from hand trajectories.

Computes robot-agnostic actions (wrist deltas, finger joint angles, gripper openness,
and palm orientation) from tracked hand landmarks across frames.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .datatypes import AnnotationFrame, RobotAgnosticAction

logger = logging.getLogger(__name__)


@dataclass
class ActionComputerConfig:
    """Configuration for the ActionComputer."""
    compute_wrist_delta: bool = True
    compute_finger_angles: bool = True
    compute_gripper_state: bool = True
    normalize_actions: bool = True
    percentile_low: float = 1.0
    percentile_high: float = 99.0


class ActionComputer:
    """Computes robot-agnostic actions from sequential hand landmark frames."""

    def __init__(self, config: ActionComputerConfig):
        """Initialise the ActionComputer."""
        self.config = config

    def compute_actions(self, frames: List[AnnotationFrame]) -> List[AnnotationFrame]:
        """Compute actions for each frame, and optionally normalise wrist deltas.

        Args:
            frames: Sequence of AnnotationFrame objects to be updated in place.

        Returns:
            The list of updated AnnotationFrames.
        """
        if not frames:
            return frames

        # First pass: compute individual frame actions
        for i, frame in enumerate(frames):
            prev_frame = frames[i - 1] if i > 0 else None
            frame.action = self._compute_frame_action(frame, prev_frame)

        # Second pass: normalize wrist deltas across the episode
        if self.config.normalize_actions:
            self._normalize_actions(frames)

        return frames

    def _compute_frame_action(
        self, frame: AnnotationFrame, prev_frame: Optional[AnnotationFrame]
    ) -> Optional[RobotAgnosticAction]:
        """Compute the robot-agnostic action primitive for a single frame."""
        # Use dominant hand (Right hand if present, otherwise Left hand)
        hand = None
        handedness = "Right"
        if frame.right_hand is not None:
            hand = frame.right_hand
            handedness = "Right"
        elif frame.left_hand is not None:
            hand = frame.left_hand
            handedness = "Left"

        if hand is None:
            return None

        pts = hand.to_array()  # (21, 3) representing [x, y, z]

        # 1. Compute Wrist Delta
        wrist_delta = np.zeros(3, dtype=np.float32)
        if self.config.compute_wrist_delta:
            p0 = pts[0]  # Wrist landmark
            prev_hand = None
            if prev_frame is not None:
                prev_hand = (
                    prev_frame.right_hand
                    if handedness == "Right"
                    else prev_frame.left_hand
                )

            if prev_hand is not None:
                prev_pts = prev_hand.to_array()
                prev_p0 = prev_pts[0]
                # Scale delta by 10 and clip to [-1, 1] as requested
                wrist_delta = (p0 - prev_p0) * 10.0
                wrist_delta = np.clip(wrist_delta, -1.0, 1.0)

        # 2. Compute Finger Angles (15 values, 3 per finger)
        finger_angles = np.zeros(15, dtype=np.float32)
        if self.config.compute_finger_angles:
            # Finger joint definitions: Wrist -> MCP -> PIP -> DIP -> Tip
            # Thumb CMC, MCP, IP
            # Others MCP, PIP, DIP
            finger_joints = [
                [0, 1, 2, 3, 4],       # Thumb
                [0, 5, 6, 7, 8],       # Index
                [0, 9, 10, 11, 12],    # Middle
                [0, 13, 14, 15, 16],   # Ring
                [0, 17, 18, 19, 20],   # Pinky
            ]

            angles_list = []
            for joints in finger_joints:
                j0, j1, j2, j3, j4 = pts[joints]

                # Joint 1 angle (between j1-j0 and j2-j1)
                theta1 = self._compute_angle(j1 - j0, j2 - j1)
                angles_list.append(1.0 - 2.0 * (theta1 / np.pi))

                # Joint 2 angle (between j2-j1 and j3-j2)
                theta2 = self._compute_angle(j2 - j1, j3 - j2)
                angles_list.append(1.0 - 2.0 * (theta2 / np.pi))

                # Joint 3 angle (between j3-j2 and j4-j3)
                theta3 = self._compute_angle(j3 - j2, j4 - j3)
                angles_list.append(1.0 - 2.0 * (theta3 / np.pi))

            finger_angles = np.array(angles_list, dtype=np.float32)

        # 3. Compute Gripper Openness
        gripper_openness = 1.0
        if self.config.compute_gripper_state:
            # Distance between thumb tip (4) and pinky tip (20)
            d_thumb_pinky = np.linalg.norm(pts[4] - pts[20])
            # Distance between index tip (8) and pinky tip (20)
            d_index_pinky = np.linalg.norm(pts[8] - pts[20])

            avg_spread = (d_thumb_pinky + d_index_pinky) / 2.0
            # Map from [0.05, 0.5] to [0, 1]
            mapped = (avg_spread - 0.05) / (0.5 - 0.05)
            gripper_openness = float(np.clip(mapped, 0.0, 1.0))

        # 4. Compute Hand Orientation
        # Represent palm plane normal [nx, ny, nz, d] from wrist (0), index_mcp (5), pinky_mcp (17)
        p0 = pts[0]
        p5 = pts[5]
        p17 = pts[17]

        v_index = p5 - p0
        v_pinky = p17 - p0

        normal = np.cross(v_index, v_pinky)
        norm_val = np.linalg.norm(normal)
        if norm_val > 1e-6:
            normal = normal / norm_val
        else:
            normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Compute plane offset d = -normal . p0
        d = -float(np.dot(normal, p0))
        hand_orientation = np.array(
            [normal[0], normal[1], normal[2], d], dtype=np.float32
        )

        return RobotAgnosticAction(
            wrist_delta=wrist_delta,
            finger_angles=finger_angles,
            gripper_openness=gripper_openness,
            hand_orientation=hand_orientation,
        )

    def _compute_angle(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute the angle in radians between two 3D vectors."""
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        cos_theta = np.dot(v1, v2) / (n1 * n2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return float(np.arccos(cos_theta))

    def _normalize_actions(self, frames: List[AnnotationFrame]) -> None:
        """Normalise wrist_deltas in place across all frames."""
        deltas = []
        for f in frames:
            if f.action is not None:
                deltas.append(f.action.wrist_delta)

        if not deltas:
            return

        deltas_arr = np.array(deltas)  # (N, 3)

        # Compute low and high percentiles for each of x, y, z dimensions
        low = np.percentile(deltas_arr, self.config.percentile_low, axis=0)
        high = np.percentile(deltas_arr, self.config.percentile_high, axis=0)

        center = (high + low) / 2.0
        scale = (high - low) / 2.0

        for f in frames:
            if f.action is not None:
                val = f.action.wrist_delta
                # Avoid divide by zero if scale is too small
                safe_scale = np.where(scale > 1e-6, scale, 1.0)
                norm_val = (val - center) / safe_scale
                f.action.wrist_delta = np.clip(norm_val, -1.0, 1.0)
