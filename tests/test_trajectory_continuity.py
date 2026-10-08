"""Regression tests for the 2026-10-03 real-verification failure.

Episode ``whatsapp_video`` (729 frames) failed the physics gate:
  - M1 tracking 1.9396 rad (threshold 0.15)
  - 15 self-collision frames counted as reachable
  - 0 grasp windows found (a cascade of the tracking failure)

Root causes fixed here:
  1. ``IKSolver._solve_ik_single`` picked the best of ``num_attempts`` by
     residual alone, so random restarts flipped IK branches frame-to-frame
     (up to 5.677 rad jumps in the exported trajectory). Attempts are now
     scored ``residual + continuity_weight * max|dq|`` against the previous
     exported solution.
  2. The pipeline's Stage 9 inlined retargeting WITHOUT the R3 source gate,
     R4 velocity gate, or self-collision gate (those only lived in
     ``Retargeter.run_from_annotations``). Both paths now share
     ``apply_retargeting_gates``.
  3. Verifier M1 counted teleportation error across unreachable gaps: after a
     gap the sim resumed from a stale pose. The sim is now placed at the
     commanded pose at each reachable segment start.
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
    import h5py  # noqa: E402 (real; installed in the sandbox)

    from src.retargeting.episode_verifier import (  # noqa: E402
        EpisodeVerifier,
        VerifierConfig,
    )
    from src.retargeting.ik_solver import IKSolverConfig, IKResult  # noqa: E402
    from src.retargeting.retargeter import (  # noqa: E402
        RetargetingConfig,
        apply_retargeting_gates,
        enforce_velocity_limits,
    )

try:
    import pybullet  # noqa: F401
    import pybullet_data  # noqa: F401
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False

# Reuse the cache-XML + writer helpers from the hermetic verifier tests.
from test_episode_verifier import _write_cache, CACHE_XML  # noqa: E402

EE_ID, OBJ_ID, TABLE_ID, WELD_ID = 10, 11, 12, 0
EE_POS = np.array([0.5, 0.0, 0.31])


# ---------------------------------------------------------------------------
# Helpers: minimal fakes for the gate tests (no pybullet needed)
# ---------------------------------------------------------------------------
def _make_ik_result(idx, angles, reachable=True, collided=False):
    return IKResult(
        frame_idx=idx,
        timestamp=idx / 30.0,
        joint_angles=np.asarray(angles, dtype=float),
        reachable=reachable,
        ik_residual_m=0.002,
        solve_time_ms=1.0,
        has_self_collision=collided,
    )


def _make_target_pose(idx, hand_detected=True, interpolated=False):
    from src.retargeting.pose_mapper import TargetPose
    return TargetPose(
        frame_idx=idx,
        timestamp=idx / 30.0,
        position=np.array([0.5, 0.0, 0.4]),
        quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        hand_detected=hand_detected,
        hand_used="right",
        is_interpolated=interpolated,
        scaling_metadata={},
    )


def _make_gripper_cmd(idx, opening=0.05, method="continuous_distance"):
    from src.retargeting.gripper_mapper import GripperCommand
    return GripperCommand(
        frame_idx=idx,
        timestamp=idx / 30.0,
        opening_m=opening,
        opening_normalized=opening / 0.04,
        gripper_mapping_method=method,
        grasp_type_used=None,
        grasp_confidence=0.0,
        mapping_metadata={},
    )


# ---------------------------------------------------------------------------
# 1. Gate unit tests (pure numpy — run everywhere)
# ---------------------------------------------------------------------------
class TestRetargetingGates(unittest.TestCase):
    def test_velocity_gate_flags_branch_flip_jump(self):
        """A 2.5 rad single-frame jump (like the real branch flips) is gated."""
        q = np.zeros((4, 7))
        q[2, 4] = 2.5
        reach = np.array([True, True, True, True])
        out = enforce_velocity_limits(q, reach, dt=1 / 30.0)
        self.assertTrue(out[0] and out[1] and out[3])
        self.assertFalse(out[2])

    def test_gates_flag_self_collision_frame_unreachable(self):
        n = 5
        ik = [_make_ik_result(i, np.full(7, 0.02 * i), collided=(i == 2)) for i in range(n)]
        poses = [_make_target_pose(i) for i in range(n)]
        cmds = [_make_gripper_cmd(i) for i in range(n)]
        joint_traj = np.stack([r.joint_angles for r in ik])
        gates = apply_retargeting_gates(ik, poses, cmds, joint_traj)
        reach = gates["reachability"]
        self.assertTrue(all(reach[[0, 1, 3, 4]]))
        self.assertFalse(reach[2])
        self.assertEqual(gates["n_collision_gated"], 1)
        self.assertFalse(ik[2].reachable)

    def test_gates_keep_clean_trajectory_reachable(self):
        n = 8
        ik = [_make_ik_result(i, np.full(7, 0.02 * i)) for i in range(n)]
        poses = [_make_target_pose(i) for i in range(n)]
        cmds = [_make_gripper_cmd(i) for i in range(n)]
        joint_traj = np.stack([r.joint_angles for r in ik])
        gates = apply_retargeting_gates(ik, poses, cmds, joint_traj)
        self.assertTrue(gates["reachability"].all())
        self.assertEqual(gates["n_velocity_infeasible"], 0)
        self.assertEqual(gates["n_collision_gated"], 0)

    def test_r3_interpolated_frame_unreachable_and_gripper_holds(self):
        n = 4
        ik = [_make_ik_result(i, np.full(7, 0.02 * i)) for i in range(n)]
        poses = [_make_target_pose(i, interpolated=(i == 2)) for i in range(n)]
        cmds = [_make_gripper_cmd(i, opening=0.01 * (i + 1)) for i in range(n)]
        joint_traj = np.stack([r.joint_angles for r in ik])
        gates = apply_retargeting_gates(ik, poses, cmds, joint_traj)
        self.assertFalse(gates["reachability"][2])
        # Gripper holds previous opening on interpolated frames.
        self.assertAlmostEqual(gates["gripper_trajectory"][2], 0.02)

    def test_continuity_weight_config_default_and_yaml_plumbing(self):
        self.assertAlmostEqual(IKSolverConfig().continuity_weight, 0.1)
        self.assertAlmostEqual(
            IKSolverConfig(continuity_weight=0.5).continuity_weight, 0.5
        )

    def test_from_yaml_reads_continuity_weight(self):
        import tempfile, os
        d = tempfile.mkdtemp()
        p = os.path.join(d, "rt.yaml")
        with open(p, "w") as f:
            f.write("robot:\n  ik_backend: pybullet\nik_solver:\n  continuity_weight: 0.25\n")
        cfg = RetargetingConfig.from_yaml(p)
        self.assertAlmostEqual(cfg.ik_solver.continuity_weight, 0.25)


# ---------------------------------------------------------------------------
# 2. Verifier M1 segment-reset test (fake mujoco, partial tracking)
# ---------------------------------------------------------------------------
def _make_partial_tracking_mujoco():
    """Fake mujoco whose position servos track only 50% per step.

    Without the segment-start reset, the first frame of a reachable segment
    after an unreachable gap measures the teleport error (stale sim pose vs
    new command). With the reset it measures ~0. The 1.7 rad gap jump below
    fails the 0.15 rad gate without the fix and passes with it.
    """
    mj = types.ModuleType("mujoco")

    class _Obj:
        mjOBJ_JOINT = 1
        mjOBJ_BODY = 2
        mjOBJ_EQUALITY = 3
        mjOBJ_ACTUATOR = 4

    class _Joint:
        mjJNT_FREE = 0

    mj.mjtObj = _Obj()
    mj.mjtJoint = _Joint()
    _kind = {1: "joint", 2: "body", 3: "equality", 4: "actuator"}

    class FakeModel:
        def __init__(self):
            self.nq = 16
            self.nv = 15
            self.nu = 9
            self.njnt = 10
            self.nbody = 13
            self.neq = 1
            self.jnt_type = np.array([1] * 9 + [0])
            self.jnt_qposadr = np.arange(10)
            self.jnt_dofadr = np.arange(10)
            self.opt = types.SimpleNamespace(timestep=0.00416667)
            self.eq_data = np.zeros((1, 7))
            self.geom_bodyid = np.zeros(30, dtype=int)
            self._ids = {}
            for i in range(7):
                self._ids[("joint", f"panda_joint{i+1}")] = i
                self._ids[("actuator", f"ver_q{i+1}")] = i
            for i in range(2):
                self._ids[("joint", f"panda_finger_joint{i+1}")] = 7 + i
                self._ids[("actuator", f"ver_f{i+1}")] = 7 + i
            self._ids[("body", "panda_link8")] = EE_ID
            self._ids[("body", "obj")] = OBJ_ID
            self._ids[("body", "table")] = TABLE_ID
            self._ids[("equality", "ver_grasp_weld")] = WELD_ID

        @classmethod
        def from_xml_path(cls, path):
            return cls()

    class FakeData:
        def __init__(self, model):
            self.qpos = np.zeros(model.nq)
            self.qvel = np.zeros(model.nv)
            self.ctrl = np.zeros(model.nu)
            self.xpos = np.zeros((model.nbody, 3))
            self.xquat = np.zeros((model.nbody, 4))
            self.xquat[:, 0] = 1.0
            self.eq_active = np.zeros(model.neq, dtype=int)
            self.contact = np.zeros(0, dtype=[("geom1", "i4"), ("geom2", "i4")])
            self.ncon = 0
            self.xpos[EE_ID] = EE_POS.copy()
            self.xpos[OBJ_ID] = EE_POS.copy()

    def mj_name2id(model, objtype, name):
        return model._ids.get((_kind[objtype], name), -1)

    def mj_saveLastXML(path, model):
        pass

    def mj_resetData(model, data):
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        data.eq_active[:] = 0
        data.xpos[EE_ID] = EE_POS.copy()
        data.xpos[OBJ_ID] = EE_POS.copy()

    def mj_forward(model, data):
        pass

    def mj_step(model, data):
        # Partial (50%) position-servo tracking per step.
        data.qpos[:7] += 0.5 * (data.ctrl[:7] - data.qpos[:7])
        data.qpos[7:9] += 0.5 * (data.ctrl[7:9] - data.qpos[7:9])

    mj.MjModel = FakeModel
    mj.MjData = FakeData
    mj.mj_name2id = mj_name2id
    mj.mj_saveLastXML = mj_saveLastXML
    mj.mj_resetData = mj_resetData
    mj.mj_forward = mj_forward
    mj.mj_step = mj_step
    return mj


def _write_gap_hdf5(path):
    n = 15
    q = np.zeros((n, 7))
    q[0:5] = 0.3
    q[10:15] = 2.0  # 1.7 rad teleport across the unreachable gap
    g = np.full((n, 1), 0.08)  # gripper open -> no grasp windows
    r = np.zeros(n, dtype=bool)
    r[0:5] = True
    r[10:15] = True
    with h5py.File(path, "w") as f:
        ep = f.create_group("ep001")
        obs = ep.create_group("steps").create_group("observation")
        obs.create_dataset("robot_joint_angles", data=q)
        obs.create_dataset("robot_gripper_opening_m", data=g)
        obs.create_dataset("robot_reachable", data=r)


class TestVerifierSegmentReset(unittest.TestCase):
    def test_m1_ignores_teleport_across_unreachable_gap(self):
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        urdf, _ = _write_cache(tmp)
        hdf5 = tmp / "ep.hdf5"
        _write_gap_hdf5(str(hdf5))
        with fake_module("mujoco", _make_partial_tracking_mujoco()):
            import tempfile as _tf
            default = Path(_tf.gettempdir()) / "egoannotate_physics"
            default.mkdir(parents=True, exist_ok=True)
            (default / "panda_verifier_cache.xml").write_text(CACHE_XML)
            v = EpisodeVerifier(VerifierConfig(urdf_path=urdf))
            rep = v.verify_episode(str(hdf5), episode_id="ep001")
        by_name = {c.name: c for c in rep.checks}
        m1 = by_name["m1_tracking_err_rad"]
        # Without the segment-start reset this is ~0.165 (fails the 0.15 gate);
        # with it the in-segment tracking is ~0.
        self.assertTrue(m1.passed, f"M1 should pass; value={m1.value}")
        self.assertLess(m1.value, 0.15)


# ---------------------------------------------------------------------------
# 3. PyBullet continuity test (runs on the Mac; skipped in minimal envs)
# ---------------------------------------------------------------------------
@unittest.skipUnless(PYBULLET_AVAILABLE, "pybullet not installed")
class TestIKContinuity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.retargeting.urdf_loader import URDFLoader
        from src.retargeting.ik_solver import IKSolver, IKSolverConfig
        from src.retargeting.pose_mapper import TargetPose
        loader = URDFLoader()
        cls.kin = loader.load(
            end_effector_link="panda_link8",
            gripper_joint_names=("panda_finger_joint1", "panda_finger_joint2"),
        )
        cls.IKSolver = IKSolver
        cls.IKSolverConfig = IKSolverConfig
        cls.TargetPose = TargetPose
        # FK at rest pose -> known-reachable target.
        client = pybullet.connect(pybullet.DIRECT)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                         physicsClientId=client)
        robot = pybullet.loadURDF(cls.kin.urdf_path, useFixedBase=True,
                                  physicsClientId=client)
        for idx, angle in zip(cls.kin.arm_joint_indices,
                              cls.kin.rest_poses.tolist()):
            pybullet.resetJointState(robot, idx, angle, physicsClientId=client)
        state = pybullet.getLinkState(robot, cls.kin.end_effector_index,
                                      computeForwardKinematics=True,
                                      physicsClientId=client)
        cls.ee_pos = np.array(state[4])
        cls.ee_quat = np.array(state[5])
        pybullet.disconnect(client)

    def test_smooth_targets_yield_continuous_trajectory(self):
        """12 near-identical targets must not produce branch-flip jumps."""
        poses = []
        for i in range(12):
            poses.append(self.TargetPose(
                frame_idx=i,
                timestamp=i / 30.0,
                position=self.ee_pos + np.array([0.002 * i, 0.0, 0.0]),
                quaternion=self.ee_quat,
                hand_detected=True,
                hand_used="right",
                is_interpolated=False,
                scaling_metadata={},
            ))
        cfg = self.IKSolverConfig(num_attempts=3, seed=42)
        with self.IKSolver(self.kin, cfg) as solver:
            results = solver.solve_sequence(poses)
        self.assertTrue(all(r.reachable for r in results),
                        "all near-identical targets should be reachable")
        qs = np.stack([r.joint_angles for r in results])
        max_jump = float(np.max(np.abs(np.diff(qs, axis=0))))
        self.assertLess(max_jump, 0.3,
                        f"branch flip detected: max frame jump {max_jump:.3f} rad")


if __name__ == "__main__":
    unittest.main()
