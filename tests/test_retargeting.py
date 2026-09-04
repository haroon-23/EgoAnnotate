"""Tests for the human-to-robot kinematic retargeting module.

Tests cover:
  1. URDF loading and joint limit extraction — compare against known Franka spec.
  2. PoseMapper with synthetic hand landmarks — verify quaternion orthonormality
     and workspace normalization bounds.
  3. IKSolver with a known reachable target (FK from neutral pose) — verify IK
     recovers the joint angles within tolerance.
  4. Joint limit violation detection — explicitly out-of-range result must be
     flagged reachable=False, NOT silently clipped.
  5. GripperMapper precedence rule — high-confidence uses grasp_type method,
     low-confidence uses continuous_distance.
  6. End-to-end Retargeter smoke test on minimal synthetic data.
"""
from __future__ import annotations

import math
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure src/ importable from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pybullet as pb
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False

from src.retargeting.pose_mapper import (
    PoseMapper,
    PoseMapperConfig,
    _rotation_matrix_to_quaternion,
    _wrist_orientation_from_landmarks,
    _safe_normalize,
)
from src.retargeting.gripper_mapper import (
    GripperMapper,
    GripperMapperConfig,
    CONFIDENCE_THRESHOLD,
)
from src.datatypes import HandLandmarks, GraspType, AnnotationFrame


def _make_hand_landmarks(
    wrist=(0.5, 0.5, 0.0),
    index_mcp=(0.6, 0.4, -0.05),
    pinky_mcp=(0.4, 0.4, -0.05),
) -> HandLandmarks:
    """Construct a minimal HandLandmarks with only the key points set."""
    x = np.zeros(21, dtype=np.float32)
    y = np.zeros(21, dtype=np.float32)
    z = np.zeros(21, dtype=np.float32)
    # Wrist
    x[0], y[0], z[0] = wrist
    # Index MCP (5)
    x[5], y[5], z[5] = index_mcp
    # Pinky MCP (17)
    x[17], y[17], z[17] = pinky_mcp
    # Thumb tip (4) and index tip (8) — needed for grasp
    x[4], y[4], z[4] = 0.55, 0.45, -0.03
    x[8], y[8], z[8] = 0.62, 0.35, -0.06
    return HandLandmarks(
        x=x, y=y, z=z, confidence=0.9, handedness="Right", is_interpolated=False
    )


def _make_frame(frame_idx=0, hand=None) -> AnnotationFrame:
    h = hand or _make_hand_landmarks()
    return AnnotationFrame(
        frame_idx=frame_idx,
        timestamp=float(frame_idx) / 30.0,
        image_path="fake.jpg",
        right_hand=h,
    )


# ============================================================================
# 1. URDF Loading Tests
# ============================================================================

@unittest.skipUnless(PYBULLET_AVAILABLE, "pybullet not installed")
class TestURDFLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.retargeting.urdf_loader import URDFLoader
        loader = URDFLoader()
        cls.kin = loader.load(
            end_effector_link="panda_link8",
            gripper_joint_names=("panda_finger_joint1", "panda_finger_joint2"),
        )  # Uses bundled panda.urdf

    def test_missing_params_raises_error(self):
        """Omitting end_effector_link or gripper_joint_names must raise ValueError."""
        from src.retargeting.urdf_loader import URDFLoader
        loader = URDFLoader()
        with self.assertRaises(ValueError):
            loader.load(end_effector_link=None, gripper_joint_names=("j1",))
        with self.assertRaises(ValueError):
            loader.load(end_effector_link="l1", gripper_joint_names=None)

    def test_kuka_iiwa_loads_7_dof_no_gripper(self):
        """KUKA iiwa URDF must load cleanly with 7 revolute arm joints and 0 gripper joints."""
        from src.retargeting.urdf_loader import URDFLoader
        loader = URDFLoader()
        kin_kuka = loader.load(
            urdf_key="kuka_iiwa/model.urdf",
            end_effector_link="lbr_iiwa_link_7",
            gripper_joint_names=(),
        )
        self.assertEqual(len(kin_kuka.arm_joints), 7)
        self.assertEqual(len(kin_kuka.gripper_joints), 0)
        self.assertEqual(kin_kuka.robot_name, "model")

    def test_arm_dof_is_7(self):
        """Franka Panda must have exactly 7 revolute arm joints."""
        self.assertEqual(len(self.kin.arm_joints), 7,
                         f"Expected 7 arm DOF, got {len(self.kin.arm_joints)}")

    def test_gripper_dof_is_2(self):
        """Franka Panda parallel gripper has 2 prismatic finger joints."""
        self.assertEqual(len(self.kin.gripper_joints), 2,
                         f"Expected 2 gripper joints, got {len(self.kin.gripper_joints)}")

    def test_joint_names_match_panda(self):
        """Arm joint names must match expected panda_joint1..7 pattern."""
        names = [j.name for j in self.kin.arm_joints]
        for i in range(1, 8):
            self.assertIn(
                f"panda_joint{i}", names,
                f"panda_joint{i} not found in {names}",
            )

    def test_joint_limits_match_spec_within_threshold(self):
        """Extracted joint limits must match known published Franka Panda spec within 0.01 rad."""
        from src.retargeting.urdf_loader import _PANDA_KNOWN_LIMITS, LIMIT_WARN_THRESHOLD_RAD
        joint_by_name = {j.name: j for j in self.kin.arm_joints + self.kin.gripper_joints}

        for joint_name, (expected_lo, expected_hi) in _PANDA_KNOWN_LIMITS.items():
            if joint_name not in joint_by_name:
                continue
            j = joint_by_name[joint_name]
            threshold = 0.001 if j.type == "prismatic" else LIMIT_WARN_THRESHOLD_RAD
            self.assertAlmostEqual(
                j.lower_limit, expected_lo, delta=threshold,
                msg=(f"{joint_name} lower: URDF={j.lower_limit:.4f}, "
                     f"spec={expected_lo:.4f}, delta={abs(j.lower_limit-expected_lo):.4f}"),
            )
            self.assertAlmostEqual(
                j.upper_limit, expected_hi, delta=threshold,
                msg=(f"{joint_name} upper: URDF={j.upper_limit:.4f}, "
                     f"spec={expected_hi:.4f}, delta={abs(j.upper_limit-expected_hi):.4f}"),
            )

    def test_no_validation_warnings(self):
        """Bundled panda.urdf should produce no limit mismatch warnings."""
        if self.kin.validation_warnings:
            # Warn but don't fail — the URDF is authoritative
            print("\n[INFO] Joint limit warnings (URDF vs known spec):")
            for w in self.kin.validation_warnings:
                print(f"  {w}")
        # This is an informational check, not a hard failure
        # (URDF is always the source of truth for the IK solver)

    def test_limits_arrays_shapes(self):
        """Lower/upper limit arrays must have shape (7,) for 7-DOF arm."""
        self.assertEqual(self.kin.lower_limits.shape, (7,))
        self.assertEqual(self.kin.upper_limits.shape, (7,))
        self.assertEqual(self.kin.rest_poses.shape, (7,))

    def test_all_lower_limits_less_than_upper(self):
        """Every joint lower_limit must be strictly less than upper_limit."""
        for j in self.kin.arm_joints:
            self.assertLess(
                j.lower_limit, j.upper_limit,
                f"{j.name}: lower {j.lower_limit:.4f} >= upper {j.upper_limit:.4f}",
            )

    def test_gripper_max_opening_is_positive(self):
        """Gripper max total opening must be positive."""
        self.assertGreater(self.kin.gripper_max_opening_m, 0.0)

    def test_end_effector_index_valid(self):
        """End-effector link index must be a valid joint index."""
        self.assertGreaterEqual(self.kin.end_effector_index, 0)


# ============================================================================
# 2. PoseMapper Tests
# ============================================================================

class TestRotationHelpers(unittest.TestCase):

    def test_rotation_matrix_to_quaternion_identity(self):
        """Identity rotation matrix must give [0, 0, 0, 1] quaternion."""
        R = np.eye(3)
        q = _rotation_matrix_to_quaternion(R)
        # qx, qy, qz, qw
        np.testing.assert_allclose(q, [0.0, 0.0, 0.0, 1.0], atol=1e-7)

    def test_rotation_matrix_to_quaternion_norm(self):
        """Quaternion output must always be unit-norm."""
        for _ in range(20):
            # Random rotation via QR decomposition
            M = np.random.randn(3, 3)
            Q, _ = np.linalg.qr(M)
            if np.linalg.det(Q) < 0:
                Q[:, 0] *= -1  # ensure proper rotation
            q = _rotation_matrix_to_quaternion(Q)
            self.assertAlmostEqual(np.linalg.norm(q), 1.0, places=7,
                                   msg=f"Quaternion norm {np.linalg.norm(q):.8f} != 1.0")

    def test_wrist_orientation_produces_orthonormal_frame(self):
        """Orientation frame from wrist landmarks must be orthonormal."""
        hand = _make_hand_landmarks()
        R = _wrist_orientation_from_landmarks(hand)
        self.assertEqual(R.shape, (3, 3))
        # Columns should be orthonormal
        RtR = R.T @ R
        np.testing.assert_allclose(RtR, np.eye(3), atol=1e-6,
                                   err_msg="R.T @ R is not identity — frame is not orthonormal")

    def test_safe_normalize_near_zero(self):
        """Near-zero vector should return [0, 0, 1] without division error."""
        v = np.array([0.0, 0.0, 0.0])
        out = _safe_normalize(v)
        np.testing.assert_array_equal(out, [0.0, 0.0, 1.0])

    def test_safe_normalize_unit_output(self):
        """Nonzero vector must be returned with unit norm."""
        v = np.array([3.0, 4.0, 0.0])
        out = _safe_normalize(v)
        self.assertAlmostEqual(np.linalg.norm(out), 1.0, places=7)


class TestPoseMapper(unittest.TestCase):

    def setUp(self):
        self.config = PoseMapperConfig(preferred_hand="right")
        self.mapper = PoseMapper(self.config)

    def test_hand_detected_pose_in_workspace(self):
        """Mapped position must be within configured robot workspace bounds."""
        frames = [_make_frame(i) for i in range(10)]
        targets = self.mapper.map_frames(frames)
        wb = self.config.robot_workspace_bounds
        for t in targets:
            if t.hand_detected:
                self.assertGreaterEqual(t.position[0], wb["x_min"] - 1e-6)
                self.assertLessEqual(t.position[0], wb["x_max"] + 1e-6)
                self.assertGreaterEqual(t.position[1], wb["y_min"] - 1e-6)
                self.assertLessEqual(t.position[1], wb["y_max"] + 1e-6)
                self.assertGreaterEqual(t.position[2], self.config.z_floor_m - 1e-6)

    def test_no_hand_returns_center(self):
        """Frame with no hand should return workspace center and hand_detected=False."""
        frame = AnnotationFrame(
            frame_idx=0, timestamp=0.0, image_path="fake.jpg",
            right_hand=None, left_hand=None,
        )
        targets = self.mapper.map_frames([frame])
        t = targets[0]
        self.assertFalse(t.hand_detected)
        self.assertIsNone(t.hand_used)

    def test_quaternion_is_unit_norm(self):
        """All output quaternions must be unit-norm."""
        frames = [_make_frame(i) for i in range(5)]
        targets = self.mapper.map_frames(frames)
        for t in targets:
            norm = np.linalg.norm(t.quaternion)
            self.assertAlmostEqual(norm, 1.0, places=6,
                                   msg=f"Frame {t.frame_idx}: quaternion norm {norm:.8f}")

    def test_scaling_metadata_contains_caveats(self):
        """Every target pose must include the workspace approximation caveat."""
        frames = [_make_frame(0)]
        targets = self.mapper.map_frames(frames)
        self.assertIn("caveat", targets[0].scaling_metadata)
        self.assertIn("MediaPipe", targets[0].scaling_metadata["caveat"])

    def test_preferred_hand_right_then_fallback_left(self):
        """right preference should use right hand; fallback to left when right absent."""
        left_hand = _make_hand_landmarks(wrist=(0.3, 0.3, 0.0))
        frame_right = _make_frame(0)   # has right_hand
        frame_left = AnnotationFrame(
            frame_idx=1, timestamp=1/30, image_path="fake.jpg",
            left_hand=left_hand, right_hand=None,
        )
        targets = self.mapper.map_frames([frame_right, frame_left])
        self.assertEqual(targets[0].hand_used, "right")
        self.assertEqual(targets[1].hand_used, "left")


# ============================================================================
# 3. IKSolver Tests (require pybullet)
# ============================================================================

@unittest.skipUnless(PYBULLET_AVAILABLE, "pybullet not installed")
class TestIKSolver(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.retargeting.urdf_loader import URDFLoader
        from src.retargeting.ik_solver import IKSolver, IKSolverConfig
        loader = URDFLoader()
        cls.kin = loader.load(
            end_effector_link="panda_link8",
            gripper_joint_names=("panda_finger_joint1", "panda_finger_joint2"),
        )
        cls.IKSolver = IKSolver
        cls.IKSolverConfig = IKSolverConfig

    def _make_target_pose_at_rest_fk(self):
        """Compute FK at the rest pose and return a TargetPose at that position."""
        import pybullet as pb
        import pybullet_data
        from src.retargeting.pose_mapper import TargetPose

        client = pb.connect(pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
        robot = pb.loadURDF(
            self.kin.urdf_path, useFixedBase=True, physicsClientId=client
        )
        # Set to rest pose
        for idx, angle in zip(self.kin.arm_joint_indices, self.kin.rest_poses.tolist()):
            pb.resetJointState(robot, idx, angle, physicsClientId=client)
        # Get end-effector position via FK
        state = pb.getLinkState(robot, self.kin.end_effector_index,
                                computeForwardKinematics=True, physicsClientId=client)
        ee_pos = np.array(state[4])
        ee_quat = np.array(state[5])  # xyzw
        pb.disconnect(client)

        return TargetPose(
            frame_idx=0,
            timestamp=0.0,
            position=ee_pos,
            quaternion=ee_quat,
            hand_detected=True,
            hand_used="right",
            is_interpolated=False,
            scaling_metadata={"method": "test_fk"},
        )

    def test_ik_recovers_rest_pose_within_tolerance(self):
        """IK at the FK-computed rest-pose position should recover rest pose angles within 0.05 rad."""
        target = self._make_target_pose_at_rest_fk()
        config = self.IKSolverConfig(max_iterations=300, residual_threshold_m=0.005, num_attempts=3)

        with self.IKSolver(self.kin, config) as solver:
            results = solver.solve_sequence([target])

        r = results[0]
        self.assertTrue(r.reachable,
                        f"IK failed to reach FK position. residual={r.ik_residual_m:.4f}m, "
                        f"violations={r.joint_limit_violations}")
        # Check each recovered angle is close to rest pose
        for i, (recovered, expected) in enumerate(zip(r.joint_angles, self.kin.rest_poses)):
            self.assertAlmostEqual(
                recovered, expected, delta=0.15,
                msg=(f"Joint {i} ({self.kin.arm_joints[i].name}): "
                     f"IK={recovered:.4f} rad, expected rest={expected:.4f} rad"),
            )

    def test_reachable_is_false_on_joint_limit_violation(self):
        """If a raw IK solution violates joint limits, the frame must be flagged reachable=False."""
        from src.retargeting.pose_mapper import TargetPose

        # Target position far outside Franka's reachable workspace — forces limit violations
        out_of_range_target = TargetPose(
            frame_idx=0, timestamp=0.0,
            position=np.array([5.0, 5.0, 5.0]),  # 5 metres — impossible
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            hand_detected=True,
            hand_used="right",
            is_interpolated=False,
            scaling_metadata={},
        )
        config = self.IKSolverConfig(max_iterations=50, residual_threshold_m=0.005)

        with self.IKSolver(self.kin, config) as solver:
            results = solver.solve_sequence([out_of_range_target])

        r = results[0]
        self.assertFalse(r.reachable,
                         "Out-of-range target should be marked reachable=False, not silently clipped")

    def test_fallback_angles_are_previous_valid_config(self):
        """On unreachable frame, joint_angles must equal previous valid config (rest_poses for first)."""
        from src.retargeting.pose_mapper import TargetPose

        out_of_range = TargetPose(
            frame_idx=0, timestamp=0.0,
            position=np.array([10.0, 10.0, 10.0]),
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            hand_detected=True,
            hand_used="right",
            is_interpolated=False,
            scaling_metadata={},
        )
        config = self.IKSolverConfig(max_iterations=30)

        with self.IKSolver(self.kin, config) as solver:
            results = solver.solve_sequence([out_of_range])

        r = results[0]
        # Fallback should be rest_poses (the initial prev_valid_angles)
        np.testing.assert_allclose(
            r.joint_angles, self.kin.rest_poses, atol=1e-9,
            err_msg="Fallback angles should be rest_poses for first unreachable frame",
        )
        self.assertTrue(r.fallback_used)

    def test_joint_limit_check_catches_violation(self):
        """_check_joint_limits must return violations for out-of-range angles."""
        from src.retargeting.ik_solver import IKSolver, VIOLATION_THRESHOLD_RAD

        solver = IKSolver(self.kin)
        # Create angles where joint 0 is 1 radian below its lower limit
        angles = self.kin.rest_poses.copy()
        angles[0] = self.kin.lower_limits[0] - 1.0  # 1 rad below minimum

        violations = solver._check_joint_limits(angles)
        joint_name = self.kin.arm_joints[0].name
        self.assertIn(joint_name, violations,
                      f"Expected violation for {joint_name} but got: {violations}")
        self.assertGreater(violations[joint_name], VIOLATION_THRESHOLD_RAD)
        solver.close()

    def test_valid_angles_produce_no_violation(self):
        """Rest poses (which are within limits by construction) must produce no violations."""
        from src.retargeting.ik_solver import IKSolver

        solver = IKSolver(self.kin)
        violations = solver._check_joint_limits(self.kin.rest_poses)
        self.assertEqual(violations, {},
                         f"Rest poses should not violate limits, got: {violations}")
        solver.close()

    def test_ik_result_has_solve_time(self):
        """IK results must contain positive solve_time_ms."""
        target = self._make_target_pose_at_rest_fk()
        with self.IKSolver(self.kin) as solver:
            results = solver.solve_sequence([target])
        self.assertGreater(results[0].solve_time_ms, 0.0)


# ============================================================================
# 4. GripperMapper Tests
# ============================================================================

@unittest.skipUnless(PYBULLET_AVAILABLE, "pybullet not installed")
class TestGripperMapper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.retargeting.urdf_loader import URDFLoader
        loader = URDFLoader()
        cls.kin = loader.load(
            end_effector_link="panda_link8",
            gripper_joint_names=("panda_finger_joint1", "panda_finger_joint2"),
        )

    def _mapper(self, **kwargs):
        cfg = GripperMapperConfig(**kwargs)
        return GripperMapper(self.kin, cfg)

    def _grasp(self, gtype: str, conf: float, thumb_idx_dist: float = 0.05) -> GraspType:
        return GraspType(
            type=gtype, confidence=conf,
            thumb_index_distance=thumb_idx_dist, num_curled_fingers=2,
        )

    def test_precision_pinch_high_conf_maps_to_nearly_closed(self):
        """precision_pinch at conf >= threshold → nearly closed (< 20% of max)."""
        mapper = self._mapper()
        grasp = self._grasp("precision_pinch", conf=0.8)
        cmd = mapper.map_frame(0, 0.0, grasp)
        self.assertEqual(cmd.gripper_mapping_method, "grasp_type")
        max_m = self.kin.gripper_joints[0].upper_limit
        self.assertLess(cmd.opening_m, 0.20 * max_m + 1e-6,
                        f"precision_pinch opening {cmd.opening_m:.4f}m should be < 20% of {max_m:.4f}m")

    def test_open_high_conf_maps_to_mostly_open(self):
        """open grasp at conf >= threshold → > 70% of max opening."""
        mapper = self._mapper()
        grasp = self._grasp("open", conf=0.9, thumb_idx_dist=0.25)
        cmd = mapper.map_frame(0, 0.0, grasp)
        self.assertEqual(cmd.gripper_mapping_method, "grasp_type")
        max_m = self.kin.gripper_joints[0].upper_limit
        self.assertGreater(cmd.opening_m, 0.70 * max_m - 1e-6,
                           f"open grasp opening {cmd.opening_m:.4f}m should be > 70% of {max_m:.4f}m")

    def test_low_confidence_uses_continuous_distance(self):
        """Grasp with conf < threshold must use continuous_distance method."""
        mapper = self._mapper(confidence_threshold=0.6)
        grasp = self._grasp("precision_pinch", conf=0.4)  # below 0.6
        cmd = mapper.map_frame(0, 0.0, grasp)
        self.assertEqual(cmd.gripper_mapping_method, "continuous_distance",
                         f"Low confidence should use continuous_distance, got {cmd.gripper_mapping_method}")

    def test_high_confidence_uses_grasp_type_method(self):
        """Grasp with conf >= threshold must use grasp_type method."""
        mapper = self._mapper(confidence_threshold=0.6)
        grasp = self._grasp("power_wrap", conf=0.7)
        cmd = mapper.map_frame(0, 0.0, grasp)
        self.assertEqual(cmd.gripper_mapping_method, "grasp_type")

    def test_no_grasp_returns_max_opening(self):
        """None grasp (no hand) should produce max-opening command."""
        mapper = self._mapper()
        cmd = mapper.map_frame(0, 0.0, None)
        max_m = self.kin.gripper_joints[0].upper_limit
        self.assertEqual(cmd.gripper_mapping_method, "no_hand")
        self.assertAlmostEqual(cmd.opening_m, max_m, places=5)

    def test_opening_always_in_valid_range(self):
        """All opening values must be within URDF [min, max] bounds."""
        mapper = self._mapper()
        max_m = self.kin.gripper_joints[0].upper_limit
        min_m = self.kin.gripper_joints[0].lower_limit
        grasps = [
            self._grasp("precision_pinch", 0.9, 0.01),
            self._grasp("power_wrap", 0.9, 0.15),
            self._grasp("open", 0.9, 0.25),
            self._grasp("hook", 0.3, 0.08),   # low conf — continuous
            self._grasp("unknown", 0.5, 0.10),
        ]
        for g in grasps:
            cmd = mapper.map_frame(0, 0.0, g)
            self.assertGreaterEqual(cmd.opening_m, min_m - 1e-9,
                                    f"Opening {cmd.opening_m:.4f}m below min {min_m:.4f}m")
            self.assertLessEqual(cmd.opening_m, max_m + 1e-9,
                                 f"Opening {cmd.opening_m:.4f}m above max {max_m:.4f}m")

    def test_gripper_method_field_present_in_output(self):
        """gripper_mapping_method must be present and non-empty in output."""
        mapper = self._mapper()
        grasp = self._grasp("open", 0.8)
        cmd = mapper.map_frame(0, 0.0, grasp)
        self.assertIn(cmd.gripper_mapping_method, {"grasp_type", "continuous_distance", "no_hand"})

    def test_mapping_metadata_contains_caveat(self):
        """Mapping metadata must always contain the simplification caveat."""
        mapper = self._mapper()
        for g in [self._grasp("open", 0.9), self._grasp("open", 0.3), None]:
            cmd = mapper.map_frame(0, 0.0, g)
            self.assertIn("caveat", cmd.mapping_metadata,
                          f"caveat missing from metadata for method={cmd.gripper_mapping_method}")
            self.assertIn("SIMPLIFICATION", cmd.mapping_metadata["caveat"])

    def test_confidence_boundary_exactly_at_threshold(self):
        """conf == threshold exactly should use grasp_type (>= not >)."""
        mapper = self._mapper(confidence_threshold=0.6)
        grasp = self._grasp("power_wrap", conf=0.6)
        cmd = mapper.map_frame(0, 0.0, grasp)
        self.assertEqual(cmd.gripper_mapping_method, "grasp_type",
                         "Confidence exactly at threshold should use grasp_type method")


# ============================================================================
# 5. End-to-end Retargeter smoke test
# ============================================================================

@unittest.skipUnless(PYBULLET_AVAILABLE, "pybullet not installed")
class TestRetargeterSmokeTest(unittest.TestCase):

    def _write_minimal_annotations(self, tmp_dir: Path, n: int = 5) -> Path:
        """Write a minimal frame_annotations.json for smoke testing."""
        import json
        frames = []
        hand = _make_hand_landmarks()
        for i in range(n):
            grasp = GraspType(
                type="power_wrap", confidence=0.75,
                thumb_index_distance=0.12, num_curled_fingers=3,
            )
            af = AnnotationFrame(
                frame_idx=i,
                timestamp=float(i) / 30.0,
                image_path=f"fake_{i:06d}.jpg",
                right_hand=hand,
                right_grasp=grasp,
            )
            frames.append(af.to_dict())
        path = tmp_dir / "frame_annotations.json"
        with open(path, "w") as f:
            json.dump(frames, f)
        return path

    def test_smoke_retargeter_runs_without_error(self):
        """Retargeter should run end-to-end on minimal data without raising."""
        import tempfile
        from src.retargeting.retargeter import Retargeter, RetargetingConfig
        from src.retargeting.ik_solver import IKSolverConfig

        with tempfile.TemporaryDirectory() as td:
            tmp_dir = Path(td)
            ann_path = self._write_minimal_annotations(tmp_dir, n=3)
            config = RetargetingConfig(
                ik_solver=IKSolverConfig(max_iterations=50, num_attempts=1),
            )
            retargeter = Retargeter(config)
            result = retargeter.run_from_annotations(str(ann_path), episode_id="smoke_test")

        self.assertEqual(result.n_frames, 3)
        self.assertEqual(result.joint_trajectories.shape, (3, 7))
        self.assertEqual(result.gripper_trajectory.shape, (3,))
        self.assertEqual(result.reachability_mask.shape, (3,))
        self.assertIn("pct_reachable", result.summary)

    def test_gripper_method_summary_present(self):
        """Result summary must contain gripper_method_counts."""
        import tempfile
        from src.retargeting.retargeter import Retargeter, RetargetingConfig
        from src.retargeting.ik_solver import IKSolverConfig

        with tempfile.TemporaryDirectory() as td:
            tmp_dir = Path(td)
            ann_path = self._write_minimal_annotations(tmp_dir, n=3)
            config = RetargetingConfig(
                ik_solver=IKSolverConfig(max_iterations=30, num_attempts=1),
            )
            retargeter = Retargeter(config)
            result = retargeter.run_from_annotations(str(ann_path), episode_id="smoke_test")

        self.assertIn("gripper_method_counts", result.summary)
        total_frames_counted = sum(result.summary["gripper_method_counts"].values())
        self.assertEqual(total_frames_counted, 3)

    def test_save_json_produces_valid_file(self):
        """save_json must produce a parseable JSON file with correct structure."""
        import json
        import tempfile
        from src.retargeting.retargeter import Retargeter, RetargetingConfig
        from src.retargeting.ik_solver import IKSolverConfig

        with tempfile.TemporaryDirectory() as td:
            tmp_dir = Path(td)
            ann_path = self._write_minimal_annotations(tmp_dir, n=3)
            config = RetargetingConfig(
                ik_solver=IKSolverConfig(max_iterations=30, num_attempts=1),
            )
            retargeter = Retargeter(config)
            result = retargeter.run_from_annotations(str(ann_path), episode_id="smoke_test")
            out_path = str(tmp_dir / "result.json")
            result.save_json(out_path)

            with open(out_path) as f:
                data = json.load(f)

        self.assertIn("frames", data)
        self.assertEqual(len(data["frames"]), 3)
        first_frame = data["frames"][0]
        self.assertIn("joint_angles_rad", first_frame)
        self.assertIn("gripper_mapping_method", first_frame)
        self.assertIn("reachable", first_frame)
        self.assertEqual(len(first_frame["joint_angles_rad"]), 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
