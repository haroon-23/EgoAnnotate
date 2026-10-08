"""Regression tests for the 2026-10-08 real-verification failure (Run #2).

Episode ``whatsapp_video`` (729 frames) failed the physics gate after Fix G:
  - M1 tracking 0.5213 rad (threshold 0.15) — down from 1.94, but still failing
  - 189/729 frames "reachable" (25.9%)

Root cause found by instrumented M1 replay on the real trajectory:
  - The pose mapper clamped ALL wrist targets to z_min=0.20 m.
  - The table top is at z=0.25 m — so every target was 5 cm INSIDE the table.
  - 189/189 "reachable" IK solutions had the arm buried in the tabletop
    (23 contacts at the commanded pose on the worst frame).
  - The M1 "tracking error" was MuJoCo's contact solver fighting the position
    servos, not a tracking failure. The config comment even said
    ``z_min: 0.20  # above table`` — false.

Fix H:
  1. ``z_min`` 0.20 → 0.35 (10 cm above the table), ``z_floor_m`` 0.10 → 0.30.
  2. The IK solver loads the static table as a PyBullet collision body and
     flags penetrating solutions (``IKResult.has_table_collision``).
  3. ``apply_retargeting_gates`` gains a table-collision gate.
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hermetic_stubs import (  # noqa: E402
    scipy_stub_for_import,
    fake_module,
)

with scipy_stub_for_import():
    from src.retargeting.ik_solver import (  # noqa: E402
        IKSolver,
        IKSolverConfig,
        IKResult,
    )
    from src.retargeting.pose_mapper import PoseMapperConfig  # noqa: E402
    from src.retargeting.retargeter import (  # noqa: E402
        RetargetingConfig,
        apply_retargeting_gates,
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_ik_result(idx, reachable=True, table_collided=False):
    return IKResult(
        frame_idx=idx,
        timestamp=idx / 30.0,
        joint_angles=np.zeros(7),
        reachable=reachable,
        ik_residual_m=0.002,
        solve_time_ms=1.0,
        has_table_collision=table_collided,
    )


def _make_target_pose(idx):
    from src.retargeting.pose_mapper import TargetPose
    return TargetPose(
        frame_idx=idx,
        timestamp=idx / 30.0,
        position=np.array([0.5, 0.0, 0.4]),
        quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        hand_detected=True,
        hand_used="right",
        is_interpolated=False,
        scaling_metadata={},
    )


def _make_gripper_cmd(idx):
    from src.retargeting.gripper_mapper import GripperCommand
    return GripperCommand(
        frame_idx=idx,
        timestamp=idx / 30.0,
        opening_m=0.05,
        opening_normalized=1.0,
        gripper_mapping_method="continuous_distance",
        grasp_type_used=None,
        grasp_confidence=0.0,
        mapping_metadata={},
    )


# ---------------------------------------------------------------------------
# 1. Table-collision gate (pure numpy — runs everywhere)
# ---------------------------------------------------------------------------
class TestTableCollisionGate(unittest.TestCase):
    def _run_gates(self, table_flags):
        n = len(table_flags)
        ik_results = [_make_ik_result(i, table_collided=f) for i, f in enumerate(table_flags)]
        poses = [_make_target_pose(i) for i in range(n)]
        grips = [_make_gripper_cmd(i) for i in range(n)]
        q = np.zeros((n, 7))
        out = apply_retargeting_gates(ik_results, poses, grips, q)
        return ik_results, out

    def test_table_colliding_frame_gated(self):
        """A frame with has_table_collision=True is marked unreachable."""
        ik_results, out = self._run_gates([False, True, False])
        self.assertTrue(ik_results[0].reachable)
        self.assertFalse(ik_results[1].reachable)
        self.assertTrue(ik_results[2].reachable)
        self.assertEqual(out["n_table_gated"], 1)

    def test_no_table_collision_no_gate(self):
        ik_results, out = self._run_gates([False, False, False])
        self.assertTrue(all(r.reachable for r in ik_results))
        self.assertEqual(out["n_table_gated"], 0)

    def test_table_gate_only_downgrades(self):
        """The table gate never flips reachable False -> True."""
        ik_results, out = self._run_gates([True, True])
        # Make one unreachable for another reason first
        ik_results[0].reachable = False
        # Re-run: gate must not resurrect it
        poses = [_make_target_pose(i) for i in range(2)]
        grips = [_make_gripper_cmd(i) for i in range(2)]
        out = apply_retargeting_gates(ik_results, poses, grips, np.zeros((2, 7)))
        self.assertFalse(ik_results[0].reachable)
        self.assertFalse(ik_results[1].reachable)


# ---------------------------------------------------------------------------
# 2. Config defaults and YAML loading
# ---------------------------------------------------------------------------
class TestTableConfig(unittest.TestCase):
    def test_ik_solver_table_defaults(self):
        cfg = IKSolverConfig()
        self.assertTrue(cfg.table_check_enabled)
        self.assertEqual(cfg.table_center, (0.5, 0.0, 0.125))
        self.assertEqual(cfg.table_half_extents, (0.35, 0.30, 0.125))
        # Table top must be below the workspace z_min (else targets are inside it)
        table_top = cfg.table_center[2] + cfg.table_half_extents[2]
        self.assertAlmostEqual(table_top, 0.25)

    def test_pose_mapper_z_min_clears_table(self):
        cfg = PoseMapperConfig()
        table_top = 0.25
        self.assertGreater(cfg.robot_workspace_bounds["z_min"], table_top,
                           "z_min must clear the table top (2026-10-08 failure)")
        self.assertGreaterEqual(cfg.z_floor_m, table_top)

    def test_yaml_loads_table_config(self):
        import yaml
        yaml_path = Path(__file__).resolve().parent.parent / "configs" / "retargeting_franka.yaml"
        self.assertTrue(yaml_path.exists(), "retargeting_franka.yaml must exist")
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        # Workspace z_min clears the table
        z_min = data["robot"]["workspace_bounds"]["z_min"]
        table_top = (data["robot"]["table"]["center"][2]
                     + data["robot"]["table"]["half_extents"][2])
        self.assertGreater(z_min, table_top)
        self.assertGreaterEqual(data["pose_mapper"]["z_floor_m"], table_top)
        # RetargetingConfig.from_yaml picks up the table fields
        rc = RetargetingConfig.from_yaml(str(yaml_path))
        self.assertTrue(rc.ik_solver.table_check_enabled)
        self.assertEqual(tuple(rc.ik_solver.table_center),
                         tuple(data["robot"]["table"]["center"]))
        self.assertEqual(tuple(rc.ik_solver.table_half_extents),
                         tuple(data["robot"]["table"]["half_extents"]))


# ---------------------------------------------------------------------------
# 3. _check_table_collision with mocked PyBullet
# ---------------------------------------------------------------------------
class TestCheckTableCollision(unittest.TestCase):
    def _make_solver_with_mock_pb(self, contacts):
        """Build an IKSolver whose PyBullet returns `contacts` for table queries."""
        import src.retargeting.ik_solver as ik_mod

        class FakePB:
            GEOM_BOX = 3
            DIRECT = 1

            def __init__(self):
                self.created_table = False

            def connect(self, *a, **k):
                return 7

            def setAdditionalSearchPath(self, *a, **k):
                pass

            def loadURDF(self, *a, **k):
                return 3

            def createCollisionShape(self, *a, **k):
                return 11

            def createMultiBody(self, *a, **k):
                self.created_table = True
                return 12

            def resetJointState(self, *a, **k):
                pass

            def getContactPoints(self, bodyA, bodyB, **k):
                # Table body id is 12; return the scripted contacts for it.
                if bodyB == 12:
                    return contacts
                return []

            def disconnect(self, *a, **k):
                pass

        fake_pb = FakePB()
        fake_pb_data = types.SimpleNamespace()
        fake_pb_data.getDataPath = lambda: "/tmp"

        # Patch the imports inside ik_solver._connect
        orig_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "pybullet":
                return fake_pb
            if name == "pybullet_data":
                return fake_pb_data
            return orig_import(name, *args, **kwargs)

        import builtins
        real_import = builtins.__import__
        builtins.__import__ = fake_import
        try:
            kin = types.SimpleNamespace(
                urdf_path="/tmp/fake.urdf",
                arm_joint_indices=[0, 1, 2, 3, 4, 5, 6],
                rest_poses=np.zeros(7),
            )
            solver = IKSolver(kin, IKSolverConfig(table_check_enabled=True))
            solver._connect()
        finally:
            builtins.__import__ = real_import
        # Fix up: _connect stored the FakePB *instance* as self._pb via the
        # fake import; make sure the instance methods are used.
        solver._pb = fake_pb
        return solver

    @staticmethod
    def _contact(dist):
        # getContactPoints tuple; index 8 is contactDistance.
        return (0,) * 8 + (dist,) + (0,) * 4

    def test_penetration_detected(self):
        solver = self._make_solver_with_mock_pb([self._contact(-0.05)])
        self.assertTrue(solver._check_table_collision())

    def test_no_contact_clean(self):
        solver = self._make_solver_with_mock_pb([])
        self.assertFalse(solver._check_table_collision())

    def test_light_touch_below_tolerance_ignored(self):
        # 1 mm penetration < 2 mm tolerance → not a collision.
        solver = self._make_solver_with_mock_pb([self._contact(-0.001)])
        self.assertFalse(solver._check_table_collision())

    def test_disabled_check_always_false(self):
        solver = self._make_solver_with_mock_pb([self._contact(-0.10)])
        solver.config.table_check_enabled = False
        self.assertFalse(solver._check_table_collision())


if __name__ == "__main__":
    unittest.main()
