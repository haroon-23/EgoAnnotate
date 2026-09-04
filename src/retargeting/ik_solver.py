"""IK solver using PyBullet's calculateInverseKinematics with strict joint-limit enforcement.

Task 3 of the retargeting pipeline.

DESIGN PRINCIPLES (per project spec):
  - Joint limits come EXCLUSIVELY from the URDF (via RobotKinematics from urdf_loader.py).
    Nothing is hardcoded.
  - If an IK solution violates any joint limit by more than VIOLATION_THRESHOLD_RAD,
    the frame is marked reachable=False and the deviation is recorded per-joint.
    The solution is NOT silently clipped.
  - On an unreachable frame, the previous valid joint configuration is used as
    the output, with the actual deviation recorded for audit.
  - IK convergence failure (residual > residual_threshold_m) also triggers
    reachable=False.
  - All these decisions are logged and surfaced in the IKResult, never hidden.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .urdf_loader import RobotKinematics
from .pose_mapper import TargetPose

logger = logging.getLogger(__name__)

# Frames with joint limit violation > this threshold are marked unreachable.
# Smaller than LIMIT_WARN_THRESHOLD_RAD in urdf_loader — this is a strict check.
VIOLATION_THRESHOLD_RAD = 0.005  # 0.3 degrees — very tight, intentional


@dataclass
class IKSolverConfig:
    """Configuration for the IK solver.

    Attributes:
        max_iterations: PyBullet IK iteration count per frame.
        residual_threshold_m: If IK end-effector residual exceeds this (metres),
            mark reachable=False.
        joint_damping: Per-joint damping coefficient for IK stability.
        num_attempts: Number of random restarts per frame to escape local minima.
        seed: Random seed for PRNG restarts to guarantee 100% deterministic IK solves.
        verbose_violations: If True, log every joint limit violation to DEBUG.
    """
    max_iterations: int = 200
    residual_threshold_m: float = 0.005   # 5 mm
    joint_damping: float = 0.005
    num_attempts: int = 3
    seed: Optional[int] = 42
    verbose_violations: bool = False


@dataclass
class IKResult:
    """IK solve result for a single frame.

    Attributes:
        frame_idx: Source frame index.
        timestamp: Source frame timestamp in seconds.
        joint_angles: Arm joint angles in radians, shape (n_dof,).
            If unreachable, contains the nearest fallback configuration.
        reachable: True iff IK converged and solution is within all joint limits.
        ik_residual_m: End-effector position error of the IK solution in metres.
        solve_time_ms: Wall-clock time for the IK solve in milliseconds.
        joint_limit_violations: Dict mapping joint_name → violation_rad for any
            joint that exceeded its URDF limit in the raw IK solution.
            Empty if reachable=True.
        fallback_used: True if the previous valid config was substituted.
        fallback_deviation_rad: Max per-joint deviation from the raw IK solution
            to the fallback config (if fallback_used=True).
        has_self_collision: True if PyBullet detected self-collision at this pose.
    """
    frame_idx: int
    timestamp: float
    joint_angles: np.ndarray         # shape (7,) in radians
    reachable: bool
    ik_residual_m: float
    solve_time_ms: float
    joint_limit_violations: Dict[str, float] = field(default_factory=dict)
    fallback_used: bool = False
    fallback_deviation_rad: float = 0.0
    has_self_collision: bool = False


class IKSolver:
    """Solves IK for a sequence of target poses using PyBullet.

    Opens a single persistent PyBullet DIRECT client for the entire sequence,
    reusing the loaded robot model across frames for efficiency.

    Usage::

        solver = IKSolver(kinematics, config)
        results = solver.solve_sequence(target_poses)
        solver.close()

    Or use as a context manager::

        with IKSolver(kinematics, config) as solver:
            results = solver.solve_sequence(target_poses)
    """

    def __init__(
        self,
        kinematics: RobotKinematics,
        config: Optional[IKSolverConfig] = None,
    ) -> None:
        self.kinematics = kinematics
        self.config = config or IKSolverConfig()
        self._client_id: Optional[int] = None
        self._robot_id: Optional[int] = None
        self._pb = None
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

    def _connect(self) -> None:
        """Open PyBullet DIRECT client and load robot."""
        try:
            import pybullet as pb
            import pybullet_data
        except ImportError as exc:
            raise ImportError(
                "pybullet is required. Install with: pip install pybullet>=3.2.5"
            ) from exc
        self._pb = pb
        self._client_id = pb.connect(pb.DIRECT)
        pb.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self._client_id
        )
        self._robot_id = pb.loadURDF(
            self.kinematics.urdf_path,
            basePosition=[0, 0, 0],
            useFixedBase=True,
            physicsClientId=self._client_id,
        )
        # Set robot to rest pose initially
        for idx, angle in zip(
            self.kinematics.arm_joint_indices,
            self.kinematics.rest_poses.tolist(),
        ):
            pb.resetJointState(
                self._robot_id, idx, angle, physicsClientId=self._client_id
            )

    def close(self) -> None:
        """Disconnect PyBullet client."""
        if self._pb is not None and self._client_id is not None:
            try:
                self._pb.disconnect(self._client_id)
            except Exception:
                pass
        self._client_id = None
        self._robot_id = None
        self._pb = None

    def __enter__(self) -> "IKSolver":
        self._connect()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _set_joint_state(self, angles: np.ndarray) -> None:
        """Set arm joint states in the PyBullet simulation."""
        pb = self._pb
        for idx, angle in zip(self.kinematics.arm_joint_indices, angles.tolist()):
            pb.resetJointState(
                self._robot_id, idx, angle, physicsClientId=self._client_id
            )

    def _get_ee_position(self) -> np.ndarray:
        """Return current end-effector position in world frame."""
        state = self._pb.getLinkState(
            self._robot_id,
            self.kinematics.end_effector_index,
            computeForwardKinematics=True,
            physicsClientId=self._client_id,
        )
        return np.array(state[4], dtype=np.float64)  # worldLinkFramePosition

    def _solve_ik_single(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Run IK for a single target pose and return (joint_angles, residual_m).

        Returns the best result over num_attempts random restarts.
        """
        pb = self._pb
        kin = self.kinematics
        cfg = self.config

        # Build per-joint damping list (PyBullet IK requires one per DOF)
        n_active = len(kin.active_joints)  # includes gripper
        damping = [cfg.joint_damping] * n_active

        best_angles = None
        best_residual = float("inf")

        for attempt in range(cfg.num_attempts):
            if attempt > 0:
                # Random restart: perturb current joint state
                noise = np.random.uniform(-0.3, 0.3, len(kin.arm_joint_indices))
                current = np.clip(
                    kin.rest_poses + noise,
                    kin.lower_limits,
                    kin.upper_limits,
                )
                self._set_joint_state(current)

            raw = pb.calculateInverseKinematics(
                self._robot_id,
                kin.end_effector_index,
                target_pos.tolist(),
                target_quat.tolist(),
                lowerLimits=kin.lower_limits.tolist(),
                upperLimits=kin.upper_limits.tolist(),
                jointRanges=(kin.upper_limits - kin.lower_limits).tolist(),
                restPoses=kin.rest_poses.tolist(),
                jointDamping=damping,
                maxNumIterations=cfg.max_iterations,
                residualThreshold=1e-6,
                physicsClientId=self._client_id,
            )

            # raw contains values for ALL active joints (arm + gripper fingers)
            # We take only the arm DOFs (first n_arm values)
            n_arm = len(kin.arm_joint_indices)
            arm_angles = np.array(raw[:n_arm], dtype=np.float64)

            # Apply to simulation and measure actual residual
            self._set_joint_state(arm_angles)
            ee_actual = self._get_ee_position()
            residual = float(np.linalg.norm(ee_actual - target_pos))

            if residual < best_residual:
                best_residual = residual
                best_angles = arm_angles.copy()

        return best_angles, best_residual

    def _check_joint_limits(
        self, angles: np.ndarray
    ) -> Dict[str, float]:
        """Check angles against URDF limits.

        Returns a dict of joint_name → violation_rad for any joint that violates.
        Violations are how far outside the limit boundary the angle is.
        """
        violations: Dict[str, float] = {}
        for i, joint in enumerate(self.kinematics.arm_joints):
            lo = self.kinematics.lower_limits[i]
            hi = self.kinematics.upper_limits[i]
            angle = angles[i]
            if angle < lo - VIOLATION_THRESHOLD_RAD:
                violations[joint.name] = float(lo - angle)
            elif angle > hi + VIOLATION_THRESHOLD_RAD:
                violations[joint.name] = float(angle - hi)
        return violations

    def _check_self_collision(self) -> bool:
        """Return True if the current robot state has any self-collision."""
        contacts = self._pb.getContactPoints(
            self._robot_id, self._robot_id, physicsClientId=self._client_id
        )
        return contacts is not None and len(contacts) > 0

    def solve_sequence(
        self,
        target_poses: List[TargetPose],
    ) -> List[IKResult]:
        """Solve IK for a sequence of target poses.

        Opens the PyBullet client if not already open (supports calling directly
        without context manager for convenience).

        Args:
            target_poses: List of TargetPose from PoseMapper.

        Returns:
            List of IKResult, one per input pose.
        """
        if self._pb is None:
            self._connect()

        results: List[IKResult] = []
        prev_valid_angles: Optional[np.ndarray] = self.kinematics.rest_poses.copy()
        n_reachable = 0
        n_fallback = 0
        n_no_hand = 0
        total_solve_ms = 0.0

        for pose in target_poses:
            t0 = time.perf_counter()

            if not pose.hand_detected:
                # No hand — use previous valid config, mark reachable=False
                n_no_hand += 1
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                results.append(IKResult(
                    frame_idx=pose.frame_idx,
                    timestamp=pose.timestamp,
                    joint_angles=prev_valid_angles.copy(),
                    reachable=False,
                    ik_residual_m=float("nan"),
                    solve_time_ms=elapsed_ms,
                    joint_limit_violations={},
                    fallback_used=True,
                    fallback_deviation_rad=0.0,
                    has_self_collision=False,
                ))
                continue

            # Solve IK
            raw_angles, residual = self._solve_ik_single(
                pose.position, pose.quaternion
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            total_solve_ms += elapsed_ms

            # --- STRICT joint limit check (never silent clip) ---------------
            violations = self._check_joint_limits(raw_angles)
            convergence_failed = residual > self.config.residual_threshold_m

            reachable = (len(violations) == 0) and (not convergence_failed)

            if reachable:
                output_angles = raw_angles.copy()
                self._set_joint_state(output_angles)
                has_collision = self._check_self_collision()
                n_reachable += 1
                fallback_used = False
                fallback_dev = 0.0
                prev_valid_angles = output_angles
            else:
                # NOT silently clipping — use previous valid config as fallback
                output_angles = prev_valid_angles.copy()
                self._set_joint_state(output_angles)
                has_collision = self._check_self_collision()
                fallback_used = True
                fallback_dev = float(np.max(np.abs(raw_angles - prev_valid_angles)))
                n_fallback += 1

                if self.config.verbose_violations and violations:
                    for jname, viol in violations.items():
                        logger.debug(
                            "Frame %d: joint '%s' violated by %.4f rad",
                            pose.frame_idx, jname, viol,
                        )
                if convergence_failed:
                    logger.debug(
                        "Frame %d: IK residual %.4f m > threshold %.4f m",
                        pose.frame_idx, residual, self.config.residual_threshold_m,
                    )

            results.append(IKResult(
                frame_idx=pose.frame_idx,
                timestamp=pose.timestamp,
                joint_angles=output_angles,
                reachable=reachable,
                ik_residual_m=residual,
                solve_time_ms=elapsed_ms,
                joint_limit_violations=violations,
                fallback_used=fallback_used,
                fallback_deviation_rad=fallback_dev,
                has_self_collision=has_collision,
            ))

        n_hand_frames = len(target_poses) - n_no_hand
        avg_solve_ms = total_solve_ms / max(n_hand_frames, 1)

        logger.info(
            "IK sequence complete: %d/%d reachable (%.1f%%), %d fallback, "
            "%d no-hand, avg solve %.2f ms/frame",
            n_reachable, len(target_poses),
            100.0 * n_reachable / max(len(target_poses), 1),
            n_fallback, n_no_hand, avg_solve_ms,
        )

        return results

    def print_summary(self, results: List[IKResult]) -> None:
        """Print a human-readable summary of IK results."""
        n = len(results)
        n_reach = sum(1 for r in results if r.reachable)
        n_fallback = sum(1 for r in results if r.fallback_used)
        n_collision = sum(1 for r in results if r.has_self_collision)
        solved = [r for r in results if not np.isnan(r.ik_residual_m)]
        avg_res = np.mean([r.ik_residual_m for r in solved]) if solved else float("nan")
        avg_solve = np.mean([r.solve_time_ms for r in solved]) if solved else float("nan")

        # Collect all unique violation joints
        violation_counts: Dict[str, int] = {}
        for r in results:
            for jname in r.joint_limit_violations:
                violation_counts[jname] = violation_counts.get(jname, 0) + 1

        sep = "=" * 70
        print(f"\n{sep}")
        print("  IK SOLVE SUMMARY")
        print(f"{sep}")
        print(f"  Total frames          : {n}")
        print(f"  Reachable (valid IK)  : {n_reach} ({100*n_reach/max(n,1):.1f}%)")
        print(f"  Unreachable/fallback  : {n_fallback} ({100*n_fallback/max(n,1):.1f}%)")
        print(f"  Self-collision events : {n_collision} ({100*n_collision/max(n,1):.1f}%)")
        print(f"  Avg IK residual       : {avg_res*1000:.2f} mm")
        print(f"  Avg solve time        : {avg_solve:.2f} ms/frame")
        if violation_counts:
            print()
            print("  Joint limit violations (frames per joint):")
            for jname, cnt in sorted(violation_counts.items(), key=lambda x: -x[1]):
                print(f"    {jname:<32} {cnt} frames")
        else:
            print("  ✓  No joint limit violations in any reachable frame.")
        print(f"{sep}\n")
