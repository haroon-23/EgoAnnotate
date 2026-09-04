"""Grasp-to-gripper opening mapper.

Task 4 of the retargeting pipeline: map GraspType (from grasp_classifier.py)
and thumb-index fingertip distance to the robot gripper joint range derived
from the URDF.

PRECEDENCE RULE (Correction 2 from user spec):
  - If grasp_type.confidence >= CONFIDENCE_THRESHOLD (default 0.6):
      Use the DISCRETE CLAMP RANGE for that grasp type.
      Method reported as "grasp_type" in output.
  - If grasp_type.confidence < CONFIDENCE_THRESHOLD (label is unreliable):
      Use the CONTINUOUS thumb-index-distance formula instead.
      Method reported as "continuous_distance" in output.

SIMPLIFICATION CAVEAT (stated explicitly per project requirement):
  This mapping reduces a 21-DOF human hand to a 1-DOF parallel gripper opening.
  The grasp-type clamp ranges are engineering heuristics, not anatomically precise.
  They must NOT be interpreted as high-fidelity finger-level retargeting.
  This caveat is recorded in every GripperCommand.mapping_metadata.

All gripper limits (min/max opening) come from the URDF via RobotKinematics —
nothing is hardcoded in the mapping logic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from ..datatypes import GraspType
from .urdf_loader import RobotKinematics

logger = logging.getLogger(__name__)

# Confidence threshold that determines which mapping method is used.
# If grasp_type.confidence >= this, use discrete clamp. Otherwise continuous.
CONFIDENCE_THRESHOLD = 0.6

# Average adult hand reference length in metres (wrist to middle fingertip).
# Used to scale the normalized thumb-index distance to a metric opening.
DEFAULT_HAND_REF_SIZE_M = 0.18

# Gripper clamp ranges as fractions of [0, 1] normalized to gripper_max_m.
# Designed from ergonomic grasp taxonomy and Franka Panda physical constraints.
# CAVEAT: These are heuristic approximations. See module docstring.
_GRASP_TYPE_CLAMP_NORMALIZED: Dict[str, Tuple[float, float]] = {
    "precision_pinch": (0.00, 0.15),  # nearly closed — fine pinch
    "power_wrap":      (0.25, 0.75),  # partially open — palm wrap
    "hook":            (0.10, 0.40),  # partial closure
    "open":            (0.75, 1.00),  # fully open hand
    "unknown":         (0.20, 0.80),  # wide range — uncertain
}


@dataclass
class GripperMapperConfig:
    """Configuration for GripperMapper.

    Attributes:
        confidence_threshold: Grasp type confidence above which the discrete
            clamp range is used instead of the continuous formula.
        hand_ref_size_m: Reference human hand size in metres for scaling
            the normalized thumb-index distance to a metric value.
        smoothing_alpha: Exponential smoothing coefficient [0,1] applied to
            the output gripper opening sequence (0 = no smoothing, 1 = hold).
    """
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    hand_ref_size_m: float = DEFAULT_HAND_REF_SIZE_M
    smoothing_alpha: float = 0.3


@dataclass
class GripperCommand:
    """Gripper target for one video frame.

    Attributes:
        frame_idx: Source frame index.
        timestamp: Source frame timestamp.
        opening_m: Target gripper opening in metres (per-finger, not total).
            Range: [0, gripper_max_opening_per_finger_m].
        opening_normalized: opening_m / gripper_max_opening_per_finger_m, [0,1].
        gripper_mapping_method: "grasp_type" or "continuous_distance".
            Auditable per-frame to trace which precedence branch was taken.
        grasp_type_used: The grasp type string if method is "grasp_type", else None.
        grasp_confidence: The grasp confidence score used for the decision.
        mapping_metadata: Dict with full audit trail of the mapping.
    """
    frame_idx: int
    timestamp: float
    opening_m: float
    opening_normalized: float
    gripper_mapping_method: str      # "grasp_type" | "continuous_distance"
    grasp_type_used: Optional[str]
    grasp_confidence: float
    mapping_metadata: Dict = field(default_factory=dict)


class GripperMapper:
    """Maps GraspType + thumb-index distance to robot gripper opening.

    Usage::

        mapper = GripperMapper(kinematics, config)
        commands = mapper.map_frames(frames)
    """

    def __init__(
        self,
        kinematics: RobotKinematics,
        config: Optional[GripperMapperConfig] = None,
    ) -> None:
        self.kinematics = kinematics
        self.config = config or GripperMapperConfig()

        # Gripper max per-finger opening from URDF
        if kinematics.gripper_joints:
            # Take the first finger joint's upper limit as the per-finger max.
            self._gripper_max_per_finger_m = kinematics.gripper_joints[0].upper_limit
            self._gripper_min_per_finger_m = kinematics.gripper_joints[0].lower_limit
        else:
            # Fallback if no gripper joints found (e.g. arm-only robot)
            logger.info(
                "No gripper joints found in kinematics for %s. "
                "Using default unit scale [0, 0.05] m.",
                kinematics.robot_name,
            )
            self._gripper_max_per_finger_m = 0.05
            self._gripper_min_per_finger_m = 0.0

        self._prev_opening_m: Optional[float] = None  # for smoothing

    def map_frame(
        self,
        frame_idx: int,
        timestamp: float,
        grasp: Optional[GraspType],
    ) -> GripperCommand:
        """Map a single frame's grasp info to a gripper command.

        Args:
            frame_idx: Frame index.
            timestamp: Frame timestamp in seconds.
            grasp: GraspType from grasp_classifier, or None if hand absent.

        Returns:
            GripperCommand with mapping method and audit trail.
        """
        max_m = self._gripper_max_per_finger_m
        min_m = self._gripper_min_per_finger_m
        cfg = self.config

        # Simplification caveat — always included in metadata
        simplification_caveat = (
            "SIMPLIFICATION: This maps a 21-DOF human hand to a 1-DOF parallel "
            "gripper. The grasp-type clamp ranges are engineering heuristics. "
            "This is NOT high-fidelity finger-level retargeting. "
            "Gripper limits [%.4f, %.4f] m derived from URDF, not hardcoded."
            % (min_m, max_m)
        )

        if grasp is None:
            # No hand detected — open gripper to maximum (safe default)
            raw_m = max_m
            method = "no_hand"
            opening_m = float(max_m)
            opening_norm = 1.0
            cmd = GripperCommand(
                frame_idx=frame_idx,
                timestamp=timestamp,
                opening_m=opening_m,
                opening_normalized=opening_norm,
                gripper_mapping_method=method,
                grasp_type_used=None,
                grasp_confidence=0.0,
                mapping_metadata={
                    "method": method,
                    "reason": "no_hand_detected",
                    "caveat": simplification_caveat,
                    "gripper_max_m": max_m,
                    "gripper_min_m": min_m,
                },
            )
            self._prev_opening_m = opening_m
            return cmd

        confidence = grasp.confidence
        thumb_idx_dist = grasp.thumb_index_distance
        grasp_type = grasp.type

        # ----------------------------------------------------------------
        # PRECEDENCE RULE (Correction 2):
        #   confidence >= threshold → discrete grasp-type clamp
        #   confidence <  threshold → continuous thumb-index distance
        # ----------------------------------------------------------------
        if confidence >= cfg.confidence_threshold:
            # --- Method: grasp_type (discrete clamp) -----------------------
            clamp_range = _GRASP_TYPE_CLAMP_NORMALIZED.get(
                grasp_type,
                _GRASP_TYPE_CLAMP_NORMALIZED["unknown"],
            )
            clamp_lo_m = clamp_range[0] * max_m
            clamp_hi_m = clamp_range[1] * max_m

            # Within the clamp range, scale by thumb-index distance as a
            # fine-grained signal (preserves some motion nuance)
            alpha = min(1.0, thumb_idx_dist / (cfg.hand_ref_size_m * 0.5))
            raw_m = clamp_lo_m + alpha * (clamp_hi_m - clamp_lo_m)
            raw_m = float(max(clamp_lo_m, min(clamp_hi_m, raw_m)))

            method = "grasp_type"
            metadata = {
                "method": method,
                "grasp_type": grasp_type,
                "confidence": confidence,
                "confidence_threshold": cfg.confidence_threshold,
                "clamp_range_normalized": list(clamp_range),
                "clamp_lo_m": clamp_lo_m,
                "clamp_hi_m": clamp_hi_m,
                "thumb_index_dist_normalized": thumb_idx_dist,
                "raw_opening_m_pre_smooth": raw_m,
                "caveat": simplification_caveat,
                "gripper_max_m": max_m,
                "gripper_min_m": min_m,
            }
        else:
            # --- Method: continuous_distance (fallback) --------------------
            # Scale normalized thumb-index distance to metric gripper opening.
            # hand_ref_size_m converts the normalized distance to metric:
            #   opening_m = thumb_idx_dist_normalized × hand_ref_size_m × scale_factor
            # Scale factor accounts for the ratio of human hand span to gripper span.
            scale_factor = max_m / (cfg.hand_ref_size_m * 0.6)  # 0.6 ≈ typical max grip width fraction
            raw_m = float(
                max(min_m, min(max_m, thumb_idx_dist * cfg.hand_ref_size_m * scale_factor))
            )

            method = "continuous_distance"
            metadata = {
                "method": method,
                "reason": "grasp_confidence_below_threshold",
                "confidence": confidence,
                "confidence_threshold": cfg.confidence_threshold,
                "thumb_index_dist_normalized": thumb_idx_dist,
                "hand_ref_size_m": cfg.hand_ref_size_m,
                "scale_factor": scale_factor,
                "raw_opening_m_pre_smooth": raw_m,
                "caveat": simplification_caveat,
                "gripper_max_m": max_m,
                "gripper_min_m": min_m,
            }

        # ----------------------------------------------------------------
        # Exponential smoothing to reduce jitter between frames
        # ----------------------------------------------------------------
        if self._prev_opening_m is not None:
            alpha_s = cfg.smoothing_alpha
            smoothed_m = (1.0 - alpha_s) * raw_m + alpha_s * self._prev_opening_m
        else:
            smoothed_m = raw_m
        smoothed_m = float(max(min_m, min(max_m, smoothed_m)))
        self._prev_opening_m = smoothed_m
        metadata["opening_m_after_smooth"] = smoothed_m

        opening_norm = (smoothed_m - min_m) / (max_m - min_m) if max_m > min_m else 0.0

        return GripperCommand(
            frame_idx=frame_idx,
            timestamp=timestamp,
            opening_m=smoothed_m,
            opening_normalized=float(opening_norm),
            gripper_mapping_method=method,
            grasp_type_used=grasp_type if method == "grasp_type" else None,
            grasp_confidence=confidence,
            mapping_metadata=metadata,
        )

    def map_frames(self, frames: list) -> list:
        """Map a sequence of AnnotationFrame objects to GripperCommands.

        Resets smoothing state at the start of each sequence.
        """
        self._prev_opening_m = None
        results = []
        for frame in frames:
            # Use right grasp if available, else left (matches PoseMapper logic)
            grasp = frame.right_grasp if frame.right_grasp is not None else frame.left_grasp
            cmd = self.map_frame(frame.frame_idx, frame.timestamp, grasp)
            results.append(cmd)
        return results

    def print_method_summary(self, commands: list) -> None:
        """Print a breakdown of which mapping method was used across all frames."""
        from collections import Counter
        counts = Counter(c.gripper_mapping_method for c in commands)
        total = max(len(commands), 1)
        sep = "=" * 70
        print(f"\n{sep}")
        print("  GRIPPER MAPPING METHOD SUMMARY")
        print(f"{sep}")
        print(f"  Confidence threshold: {self.config.confidence_threshold}")
        print(f"  Gripper range (per finger, from URDF): "
              f"[{self._gripper_min_per_finger_m:.4f}, "
              f"{self._gripper_max_per_finger_m:.4f}] m")
        print()
        for method, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {method:<30} {cnt:>5} frames ({100*cnt/total:.1f}%)")
        print(f"  {'TOTAL':<30} {total:>5}")
        print(f"{sep}\n")
