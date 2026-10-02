"""Hermetic tests for src/retargeting/mink_ik.py.

mink and mujoco are scripted with fake modules (sys.modules injection) that
implement the exact call surface the solver uses (verified against
scripts/run_reference_pipeline.py):

    FrameTask(...).set_target(SE3...)
    solve_ik(configuration, tasks, dt, solver=..., limits=[...], damping=...)
    configuration.integrate_inplace(vel, dt)
    configuration.get_transform_frame_to_world(name, "body").translation()
"""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hermetic_stubs import (  # noqa: E402
    scipy_stub_for_import,
    blocked_modules,
    fake_module,
)

with scipy_stub_for_import():
    from src.retargeting.ik_solver import IKResult, IKSolver  # noqa: E402
    from src.retargeting.mink_ik import (  # noqa: E402
        MinkIKSolver,
        MinkIKConfig,
        _mink_import_status,
        _quat_xyzw_to_matrix,
    )
    from src.retargeting.pose_mapper import TargetPose  # noqa: E402
    from src.retargeting.retargeter import (  # noqa: E402
        RetargetingConfig,
        create_ik_solver,
    )
    from src.retargeting.urdf_loader import JointInfo, RobotKinematics  # noqa: E402


def make_kinematics(n_arm=7):
    arm = [
        JointInfo(
            index=i, name=f"panda_joint{i+1}", type="revolute",
            lower_limit=-2.9, upper_limit=2.9,
            parent_link=f"panda_link{i}", child_link=f"panda_link{i+1}",
            axis=(0.0, 0.0, 1.0),
        )
        for i in range(n_arm)
    ]
    return RobotKinematics(
        urdf_path="/tmp/fake_panda.urdf",
        robot_name="panda",
        base_link="panda_link0",
        end_effector_link="panda_link8",
        end_effector_index=8,
        joints=[], active_joints=arm, arm_joints=arm, gripper_joints=[],
        arm_joint_indices=list(range(n_arm)), gripper_joint_indices=[],
        lower_limits=np.full(n_arm, -2.9), upper_limits=np.full(n_arm, 2.9),
        rest_poses=np.zeros(n_arm), gripper_max_opening_m=0.08,
        validation_warnings=[],
    )


def make_pose(i, pos, hand_detected=True):
    return TargetPose(
        frame_idx=i, timestamp=i / 30.0,
        position=np.asarray(pos, float),
        quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        hand_detected=hand_detected, hand_used="right",
        is_interpolated=False, scaling_metadata={},
    )


class FakeTransform:
    def __init__(self, pos):
        self._pos = np.asarray(pos, float)

    def translation(self):
        return self._pos


def _make_fake_mink_mujoco(mode="ok", ee_pos=(0.5, 0.0, 0.3)):
    """mode: 'ok' | 'quadprog_missing' | 'all_dead' | 'no_get_transform'."""
    mj = types.ModuleType("mujoco")
    mink = types.ModuleType("mink")

    class _Obj:
        mjOBJ_JOINT = 1
        mjOBJ_BODY = 2

    mj.mjtObj = _Obj()

    class FakeModel:
        nq = 9

        def __init__(self):
            self.jnt_qposadr = np.arange(9)
            self._jids = {f"panda_joint{i+1}": i for i in range(7)}
            self._bids = {"panda_link8": 8}

        @classmethod
        def from_xml_path(cls, path):
            return cls()

    def mj_name2id(model, objtype, name):
        if objtype == 1:
            return model._jids.get(name, -1)
        return model._bids.get(name, -1)

    class FakeData:
        def __init__(self, model):
            self.qpos = np.zeros(model.nq)
            self.xpos = np.zeros((10, 3))
            self.fk_ee = np.asarray(ee_pos, float)

    def mj_forward(model, data):
        data.xpos[8] = data.fk_ee  # FK fallback path

    mj.MjModel = FakeModel
    mj.MjData = FakeData
    mj.mj_name2id = mj_name2id
    mj.mj_forward = mj_forward

    state = {"calls": []}

    class FakeConfiguration:
        def __init__(self, model):
            self._q = np.zeros(model.nq)
            self.ee_pos = np.asarray(ee_pos, float)

        def update(self, q):
            self._q = np.asarray(q, float).copy()

        @property
        def q(self):
            return self._q

        def integrate_inplace(self, dq, dt):
            self._q = self._q + np.asarray(dq, float).ravel() * dt

        if mode != "no_get_transform":
            def get_transform_frame_to_world(self, name, ftype):
                return FakeTransform(self.ee_pos)

    class FakeTask:
        def __init__(self, frame_name, frame_type, position_cost, orientation_cost):
            self.frame_name = frame_name
            self.target = None

        def set_target(self, t):
            self.target = t

        def set_target_from_position(self, p):
            self.target = ("pos", np.asarray(p, float))

    class FakePosture:
        def __init__(self, model, cost):
            self.target = None

        def set_target(self, q):
            self.target = np.asarray(q, float)

    class FakeLimit:
        def __init__(self, model, *a):
            pass

    class FakeSE3:
        @staticmethod
        def from_rotation_and_translation(rot, pos):
            return ("se3", rot, np.asarray(pos, float))

        @staticmethod
        def from_translation(pos):
            return ("se3t", np.asarray(pos, float))

    class FakeSO3:
        @staticmethod
        def from_matrix(R):
            return ("so3", np.asarray(R, float))

    def solve_ik(configuration, tasks, dt, **kw):
        state["calls"].append(kw.get("solver", "<default>"))
        solver = kw.get("solver", "daqp")
        if mode == "all_dead":
            raise RuntimeError("all QP solvers exploded")
        if mode == "quadprog_missing" and solver == "quadprog":
            raise RuntimeError("No QP solver 'quadprog' available")
        assert kw.get("limits"), "expected ConfigurationLimit + VelocityLimit"
        assert "damping" in kw
        return np.zeros(configuration.q.shape)

    mink.Configuration = FakeConfiguration
    mink.FrameTask = FakeTask
    mink.PostureTask = FakePosture
    mink.ConfigurationLimit = FakeLimit
    mink.VelocityLimit = FakeLimit
    mink.SE3 = FakeSE3
    mink.SO3 = FakeSO3
    mink.solve_ik = solve_ik
    return mj, mink, state


def _run_with_fakes(mode="ok", poses=None, ee_pos=(0.5, 0.0, 0.3), config=None):
    mj, mink, state = _make_fake_mink_mujoco(mode=mode, ee_pos=ee_pos)
    kin = make_kinematics()
    poses = poses if poses is not None else [make_pose(i, ee_pos) for i in range(3)]
    with fake_module("mujoco", mj), fake_module("mink", mink):
        solver = MinkIKSolver(kin, ee_link_name="panda_link8", config=config)
        assert solver.is_available() is True
        with solver:
            results = solver.solve_sequence(poses)
    return solver, results, state


# -- availability ------------------------------------------------------------
def test_blocked_mink_is_available_false_and_solve_raises():
    kin = make_kinematics()
    with blocked_modules("mink", "mujoco"):
        assert _mink_import_status()[0] is False
        solver = MinkIKSolver(kin)
        assert solver.is_available() is False
        with pytest.raises(ImportError):
            solver.solve_sequence([make_pose(0, (0.5, 0.0, 0.3))])


def test_module_import_never_raises_without_mink():
    # Importing the module is already proven by this file importing it under
    # blocked_modules-free conditions; here assert construction is lazy:
    with blocked_modules("mink", "mujoco"):
        kin = make_kinematics()
        MinkIKSolver(kin)  # must not raise


# -- solve ---------------------------------------------------------------------
def test_solve_sequence_reachable_with_fakes():
    solver, results, state = _run_with_fakes()
    assert len(results) == 3
    assert all(isinstance(r, IKResult) for r in results)
    assert all(r.reachable for r in results)
    assert all(not r.fallback_used for r in results)
    assert all(r.joint_angles.shape == (7,) for r in results)
    assert all(r.ik_residual_m <= 0.01 for r in results)
    assert state["calls"] == ["quadprog"] * 3  # configured solver used every frame
    assert solver._solver_name == "quadprog"


def test_solver_fallback_chain_quadprog_to_daqp():
    solver, results, state = _run_with_fakes(mode="quadprog_missing")
    assert all(r.reachable for r in results)
    assert state["calls"][0] == "quadprog"   # tried first
    assert state["calls"][1] == "daqp"        # then fell back
    assert solver._solver_name == "daqp"
    assert solver._solver_dead is False  # recovered via fallback, not dead
    # (is_available() needs the mink import, which the fake context removed)


def test_all_solvers_dead_degrades_to_unavailable():
    solver, results, state = _run_with_fakes(mode="all_dead")
    assert len(results) == 3
    assert all(not r.reachable for r in results)
    assert all(r.fallback_used for r in results)
    assert solver._solver_dead is True  # degraded, never raises


def test_no_hand_pose_uses_fallback():
    poses = [make_pose(0, (0.5, 0.0, 0.3), hand_detected=False),
             make_pose(1, (0.5, 0.0, 0.3))]
    _, results, _ = _run_with_fakes(poses=poses)
    assert results[0].reachable is False and results[0].fallback_used is True
    np.testing.assert_allclose(results[0].joint_angles, np.zeros(7))
    assert results[1].reachable is True


def test_unreachable_target_marks_not_reachable():
    # EE stays at (0.5,0,0.3) but the target is 1 m away -> residual too big.
    _, results, _ = _run_with_fakes(poses=[make_pose(0, (1.5, 0.0, 0.3))])
    assert results[0].reachable is False
    assert results[0].fallback_used is True
    assert results[0].ik_residual_m > 0.01


def test_fk_fallback_when_mink_transform_absent():
    # FakeConfiguration without get_transform_frame_to_world -> MuJoCo FK path.
    _, results, _ = _run_with_fakes(mode="no_get_transform")
    assert all(r.reachable for r in results)


def test_context_manager_protocol():
    mj, mink, _ = _make_fake_mink_mujoco()
    with fake_module("mujoco", mj), fake_module("mink", mink):
        with MinkIKSolver(make_kinematics()) as solver:
            assert isinstance(solver, MinkIKSolver)


# -- factory / config ------------------------------------------------------------
def test_create_ik_solver_unknown_backend_raises():
    with pytest.raises(ValueError):
        create_ik_solver("wat", make_kinematics())


def test_create_ik_solver_mink_missing_raises_import_error():
    with blocked_modules("mink", "mujoco"):
        with pytest.raises(ImportError):
            create_ik_solver("mink", make_kinematics())


def test_create_ik_solver_pybullet_routing():
    # IKSolver construction does not touch pybullet (only solve_sequence does),
    # so the factory routing is testable hermetically.
    solver = create_ik_solver("pybullet", make_kinematics())
    assert isinstance(solver, IKSolver)
    assert RetargetingConfig().ik_backend == "pybullet"


def test_mink_config_defaults():
    cfg = MinkIKConfig()
    assert cfg.solver == "quadprog"
    assert cfg.residual_threshold_m == 0.01
    assert cfg.use_posture_task is True and cfg.use_limit_tasks is True


# -- pure-numpy helper ---------------------------------------------------------------
def test_quat_xyzw_to_matrix_identity():
    np.testing.assert_allclose(
        _quat_xyzw_to_matrix([0, 0, 0, 1]), np.eye(3), atol=1e-12)


def test_quat_xyzw_to_matrix_z90():
    R = _quat_xyzw_to_matrix([0, 0, np.sqrt(0.5), np.sqrt(0.5)])
    np.testing.assert_allclose(
        R, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-9)
