"""mink-based IK backend for the retargeter (Phase E, opt-in).

Wraps the ``mink`` library (MuJoCo-native differential IK, pip-installable,
Apache-2.0) behind the :class:`IKSolver` interface so the retargeter can
select it via ``RetargetingConfig.ik_backend = "mink"``. The PyBullet
``IKSolver`` stays the default; mink is opt-in and needs ``mujoco`` + ``mink``
installed.

Call pattern (verified against the official mink docs on 2026-10-02, and the
repo's own proven usage in ``scripts/run_reference_pipeline.py``)::

    from mink import Configuration, FrameTask, SE3, solve_ik
    task = FrameTask(frame_name=..., frame_type="body",
                     position_cost=1.0, orientation_cost=1.0)
    task.set_target(SE3.from_rotation_and_translation(SO3.from_matrix(R), pos))
    vel = solve_ik(configuration, [task], dt=0.01, solver="daqp")
    configuration.integrate_inplace(vel, dt)

MuJoCo parses URDF directly (``MjModel.from_xml_path`` detects the format
from the top-level XML element, not the file extension — verified against the
official MuJoCo docs on 2026-10-02). Caveat: MuJoCo's URDF importer discards
``<limit effort/velocity>``, so joint velocity limits for mink's
``VelocityLimit`` task come from the URDF-kinematics when available and fall
back to 0.9 x the Franka Panda rated velocities (same 0.9 safety factor as
``scripts/run_reference_pipeline.py``).

QP solver fallback chain: the configured solver is tried first; on failure a
loud warning is logged and ``"daqp"`` is retried, then the library default
(the ``solver`` kwarg is omitted). The backend never raises for a missing
solver — if every candidate fails it degrades to unavailable (subsequent
``is_available()`` calls return False and sequences yield unreachable
fallbacks).

All mink/mujoco imports are lazy: importing this module never raises.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ik_solver import IKResult
from .pose_mapper import TargetPose
from .urdf_loader import RobotKinematics

logger = logging.getLogger(__name__)

# Franka Panda rated joint velocities (rad/s), copied from
# scripts/run_reference_pipeline.py. MuJoCo's URDF importer discards
# <limit velocity/>, so these are the fallback for mink's VelocityLimit task.
_PANDA_VEL = np.array([2.17, 2.17, 2.17, 2.17, 2.61, 2.61, 2.61])
_EE_FALLBACKS = ("panda_link8", "panda_hand", "panda_link7", "link8", "hand", "link7")


def _mink_import_status() -> Tuple[bool, str]:
    """Return (available, reason). Never raises."""
    try:
        import mujoco  # noqa: F401
        import mink  # noqa: F401
    except ImportError as exc:
        return False, (
            f"mink/mujoco not installed ({exc}). "
            "Install with: pip install mujoco mink"
        )
    return True, ""


def _quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    """Quaternion [qx, qy, qz, qw] -> 3x3 rotation matrix. Pure numpy."""
    x, y, z, w = (float(v) for v in np.asarray(q, dtype=float).ravel()[:4])
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


@dataclass
class MinkIKConfig:
    """Configuration for the mink IK backend.

    Attributes:
        residual_threshold_m: EE position error above which a frame is marked
            unreachable. (mink is more accurate than PyBullet IK, so the
            default is tighter than IKSolverConfig's 5 mm... actually 10 mm
            here to stay comparable; tune per robot.)
        dt: Integration timestep per frame (1/30 s video frame time).
        position_cost: FrameTask position cost.
        orientation_cost: FrameTask orientation cost. 0.0 = position-only.
        lm_damping: Levenberg-Marquardt damping passed to solve_ik.
        solver: QP solver name tried first ("quadprog", "daqp", ...).
        use_posture_task: Add a low-cost PostureTask biasing toward rest pose.
        use_limit_tasks: Add ConfigurationLimit + VelocityLimit tasks.
    """

    residual_threshold_m: float = 0.01
    dt: float = 1 / 30
    position_cost: float = 1.0
    orientation_cost: float = 1.0
    lm_damping: float = 1e-3
    solver: str = "quadprog"
    use_posture_task: bool = True
    use_limit_tasks: bool = True


class MinkIKSolver:
    """Differential IK via mink, implementing the IKSolver interface.

    Args:
        kinematics: RobotKinematics from URDFLoader (arm joint names, limits,
            rest poses, and the resolved URDF path).
        ee_link_name: End-effector body name for the FrameTask.
        config: MinkIKConfig (defaults if None).
    """

    def __init__(
        self,
        kinematics: RobotKinematics,
        ee_link_name: str = "panda_link8",
        config: Optional[MinkIKConfig] = None,
    ) -> None:
        self.kinematics = kinematics
        self.ee_link_name = ee_link_name
        self.config = config or MinkIKConfig()
        self._ready = False
        self._solver_dead = False
        self._solver_name: Optional[str] = None
        self._fk_warned = False
        self._ori_warned = False
        # Lazily built by _ensure_solver(); checked before use, never at import.
        self._mj = None
        self._mink = None
        self._model = None
        self._configuration = None
        self._data = None
        self._task = None
        self._tasks: List = []
        self._limits: List = []
        self._arm_qpos: Optional[np.ndarray] = None
        self._arm_names: List[str] = []
        self._ee_id: int = -1

    # -- availability -------------------------------------------------------
    def is_available(self) -> bool:
        """False when mink/mujoco are missing, or every QP solver failed."""
        ok, _ = _mink_import_status()
        return bool(ok) and not self._solver_dead

    # -- lazy construction ----------------------------------------------------
    def _ensure_solver(self) -> None:
        """Build the MuJoCo model + mink tasks. Raises loudly on real problems."""
        if self._ready:
            return
        ok, reason = _mink_import_status()
        if not ok:
            raise ImportError(f"MinkIKSolver unavailable: {reason}")
        import mujoco as mj
        import mink

        urdf = self.kinematics.urdf_path
        if not urdf:
            raise ValueError("MinkIKSolver needs kinematics.urdf_path (a URDF file).")
        # MuJoCo parses URDF directly (format from the top-level XML element).
        model = mj.MjModel.from_xml_path(urdf)
        self._mj, self._mink, self._model = mj, mink, model

        J = mj.mjtObj.mjOBJ_JOINT
        arm_qpos = []
        for j in self.kinematics.arm_joints:
            jid = mj.mj_name2id(model, J, j.name)
            if jid < 0:
                raise RuntimeError(
                    f"MinkIKSolver: arm joint {j.name!r} not found in {urdf}"
                )
            arm_qpos.append(int(model.jnt_qposadr[jid]))
        self._arm_qpos = np.array(arm_qpos, dtype=int)
        self._arm_names = [j.name for j in self.kinematics.arm_joints]

        B = mj.mjtObj.mjOBJ_BODY
        ee_id = mj.mj_name2id(model, B, self.ee_link_name)
        if ee_id < 0:
            for fb in _EE_FALLBACKS:
                ee_id = mj.mj_name2id(model, B, fb)
                if ee_id >= 0:
                    logger.warning(
                        "MinkIKSolver: EE body %r not found; using fallback %r",
                        self.ee_link_name, fb,
                    )
                    self.ee_link_name = fb
                    break
        if ee_id < 0:
            raise RuntimeError(
                f"MinkIKSolver: EE body {self.ee_link_name!r} not found in {urdf}"
            )
        self._ee_id = int(ee_id)

        cfg = self.config
        configuration = mink.Configuration(model)
        nq = int(model.nq)
        q0 = np.zeros(nq)
        rest = np.asarray(self.kinematics.rest_poses, dtype=float).ravel()
        if rest.shape[0] == len(self._arm_qpos):
            q0[self._arm_qpos] = rest
        configuration.update(q0)
        self._configuration = configuration
        self._data = mj.MjData(model)  # FK fallback for EE pose

        self._task = mink.FrameTask(
            frame_name=self.ee_link_name,
            frame_type="body",
            position_cost=cfg.position_cost,
            orientation_cost=cfg.orientation_cost,
        )
        tasks: List = [self._task]
        if cfg.use_posture_task:
            posture = mink.PostureTask(model, cost=1e-3)
            posture.set_target(q0)
            tasks.append(posture)
        if cfg.use_limit_tasks:
            self._limits = [
                mink.ConfigurationLimit(model),
                mink.VelocityLimit(model, self._velocity_limit_dict()),
            ]
        self._tasks = tasks
        self._ready = True
        logger.info(
            "MinkIKSolver ready: %d arm joints, EE=%r, solver=%r",
            len(self._arm_qpos), self.ee_link_name, cfg.solver,
        )

    def _velocity_limit_dict(self) -> Dict[str, float]:
        """Joint-name -> velocity limit (rad/s) for mink's VelocityLimit."""
        names = self._arm_names
        kin_lims = getattr(self.kinematics, "joint_velocity_limits", None)
        if kin_lims is not None:
            arr = np.asarray(kin_lims, dtype=float).ravel()
            if arr.shape[0] == len(names):
                return {n: float(0.9 * v) for n, v in zip(names, arr)}
            logger.warning(
                "MinkIKSolver: joint_velocity_limits shape %s != %d joints; "
                "using fallback", arr.shape, len(names),
            )
        if len(names) == len(_PANDA_VEL):
            return {n: float(0.9 * v) for n, v in zip(names, _PANDA_VEL)}
        logger.warning(
            "MinkIKSolver: no velocity limits for %d joints; using 2.0 rad/s",
            len(names),
        )
        return {n: 2.0 for n in names}

    # -- solve ------------------------------------------------------------------
    def _call_solve_ik(self, solver_name: Optional[str]):
        """One solve_ik call. solver_name=None -> omit the kwarg (library default)."""
        kw: Dict = {"damping": self.config.lm_damping}
        if self._limits:
            kw["limits"] = self._limits
        if solver_name is not None:
            kw["solver"] = solver_name
        return self._mink.solve_ik(
            self._configuration, self._tasks, self.config.dt, **kw
        )

    def _solve_with_fallback_chain(self):
        """Try configured solver -> 'daqp' -> library default. Returns (dq, name)."""
        cfg = self.config
        candidates: List[Optional[str]] = []
        for c in (cfg.solver, "daqp", None):
            if c not in candidates:
                candidates.append(c)
        last_exc: Optional[Exception] = None
        for name in candidates:
            label = name if name is not None else "library default"
            try:
                dq = self._call_solve_ik(name)
                if name != cfg.solver:
                    logger.warning(
                        "MinkIKSolver: solver %r failed (%s); fell back to %r",
                        cfg.solver, last_exc, label,
                    )
                return dq, name
            except Exception as exc:  # noqa: BLE001 - chain must survive any solver error
                last_exc = exc
                logger.warning("MinkIKSolver: QP solver %r failed: %s", label, exc)
        return None, None

    def _set_task_target(self, pos: np.ndarray, quat_xyzw: Optional[np.ndarray]) -> None:
        """Set the FrameTask target, probing the mink SE3/SO3 API at runtime."""
        mink, cfg = self._mink, self.config
        if quat_xyzw is not None and cfg.orientation_cost > 0:
            so3 = getattr(mink, "SO3", None)
            from_matrix = getattr(so3, "from_matrix", None) if so3 is not None else None
            se3 = getattr(mink, "SE3", None)
            from_rt = getattr(se3, "from_rotation_and_translation", None) if se3 is not None else None
            if from_matrix is not None and from_rt is not None:
                R = _quat_xyzw_to_matrix(quat_xyzw)
                self._task.set_target(from_rt(from_matrix(R), np.asarray(pos, float)))
                return
            if not self._ori_warned:
                logger.warning(
                    "MinkIKSolver: mink.SO3.from_matrix unavailable; "
                    "degrading to position-only targets."
                )
                self._ori_warned = True
        # Position-only target.
        if hasattr(self._task, "set_target_from_position"):
            self._task.set_target_from_position(np.asarray(pos, float))
        elif hasattr(mink.SE3, "from_translation"):
            self._task.set_target(mink.SE3.from_translation(np.asarray(pos, float)))
        else:
            raise RuntimeError(
                "MinkIKSolver: cannot set a FrameTask target with this mink "
                "version (no set_target_from_position / SE3.from_translation)."
            )

    def _ee_position(self) -> np.ndarray:
        """Current EE world position. Prefers mink's API; falls back to MuJoCo FK."""
        get_t = getattr(self._configuration, "get_transform_frame_to_world", None)
        if get_t is not None:
            try:
                tf = get_t(self.ee_link_name, "body")
                return np.asarray(tf.translation(), dtype=float).ravel()[:3]
            except Exception as exc:  # noqa: BLE001 - FK fallback covers it
                logger.debug("MinkIKSolver: get_transform_frame_to_world failed (%s)", exc)
        if not self._fk_warned:
            logger.warning(
                "MinkIKSolver: Configuration.get_transform_frame_to_world absent; "
                "EE pose via MuJoCo forward kinematics."
            )
            self._fk_warned = True
        mj, model, data = self._mj, self._model, self._data
        data.qpos[:] = np.asarray(self._configuration.q, dtype=float).ravel()[: int(model.nq)]
        mj.mj_forward(model, data)
        return np.array(data.xpos[self._ee_id], dtype=float).copy()

    def _check_joint_limits(self, angles: np.ndarray) -> Dict[str, float]:
        violations: Dict[str, float] = {}
        for j, a in zip(self.kinematics.arm_joints, np.asarray(angles, float).ravel()):
            lo, hi = float(j.lower_limit), float(j.upper_limit)
            if a < lo - 1e-9:
                violations[j.name] = lo - a
            elif a > hi + 1e-9:
                violations[j.name] = a - hi
        return violations

    def _fallback_result(
        self, pose: TargetPose, reason: str, prev_valid: Optional[np.ndarray] = None
    ) -> IKResult:
        rest = np.asarray(self.kinematics.rest_poses, dtype=float).ravel()
        angles = prev_valid.copy() if prev_valid is not None else rest.copy()
        logger.debug("MinkIKSolver frame %d: fallback (%s)", pose.frame_idx, reason)
        return IKResult(
            frame_idx=pose.frame_idx,
            timestamp=pose.timestamp,
            joint_angles=angles,
            reachable=False,
            ik_residual_m=float("nan"),
            solve_time_ms=0.0,
            fallback_used=True,
        )

    def solve_sequence(self, target_poses: List[TargetPose]) -> List[IKResult]:
        """Solve IK for a sequence of target poses. Same contract as IKSolver."""
        self._ensure_solver()
        cfg = self.config
        if self._solver_dead:
            rest = np.asarray(self.kinematics.rest_poses, dtype=float).ravel()
            return [self._fallback_result(p, "mink solver unavailable", rest)
                    for p in target_poses]

        integrate = getattr(self._configuration, "integrate_inplace", None)
        if integrate is None:
            integrate = getattr(self._configuration, "integrate", None)
        if integrate is None:
            raise RuntimeError(
                "MinkIKSolver: mink Configuration has neither integrate_inplace "
                "nor integrate."
            )

        results: List[IKResult] = []
        rest = np.asarray(self.kinematics.rest_poses, dtype=float).ravel()
        prev_valid = rest.copy()
        solver_name: Optional[str] = None
        first_frame = True

        for pose in target_poses:
            t0 = time.perf_counter()
            if self._solver_dead:
                results.append(self._fallback_result(
                    pose, "mink solver unavailable", prev_valid))
                continue
            if not pose.hand_detected:
                results.append(self._fallback_result(pose, "no hand detected", prev_valid))
                continue
            pos = np.asarray(pose.position, dtype=float).ravel()[:3]
            quat = getattr(pose, "quaternion", None)
            quat = np.asarray(quat, dtype=float).ravel()[:4] if quat is not None else None
            try:
                self._set_task_target(pos, quat)
            except Exception as exc:  # noqa: BLE001 - target-setting must not kill the run
                logger.warning("MinkIKSolver frame %d: bad target (%s)", pose.frame_idx, exc)
                results.append(self._fallback_result(pose, f"bad target: {exc}", prev_valid))
                continue

            if first_frame:
                dq, solver_name = self._solve_with_fallback_chain()
                first_frame = False
                if dq is None:
                    self._solver_dead = True
                    logger.error(
                        "MinkIKSolver: all QP solvers failed; backend unavailable.")
                    results.append(self._fallback_result(pose, "qp solver failure", prev_valid))
                    # Mark the rest of the sequence unreachable too.
                    continue
                self._solver_name = solver_name
            else:
                try:
                    dq = self._call_solve_ik(solver_name)
                except Exception as exc:  # noqa: BLE001 - per-frame failure -> fallback
                    logger.warning(
                        "MinkIKSolver frame %d: solve failed (%s); using fallback",
                        pose.frame_idx, exc)
                    results.append(self._fallback_result(pose, f"solve failed: {exc}", prev_valid))
                    continue

            integrate(np.asarray(dq, dtype=float).ravel(), cfg.dt)
            angles = np.array(self._configuration.q, dtype=float).ravel()[self._arm_qpos]
            residual = float(np.linalg.norm(self._ee_position() - pos))
            violations = self._check_joint_limits(angles)
            reachable = (residual <= cfg.residual_threshold_m) and not violations
            solve_ms = (time.perf_counter() - t0) * 1000.0
            if reachable:
                prev_valid = angles.copy()
                results.append(IKResult(
                    frame_idx=pose.frame_idx, timestamp=pose.timestamp,
                    joint_angles=angles, reachable=True,
                    ik_residual_m=residual, solve_time_ms=solve_ms))
            else:
                fb_dev = float(np.max(np.abs(angles - prev_valid))) \
                    if prev_valid.shape == angles.shape else 0.0
                results.append(IKResult(
                    frame_idx=pose.frame_idx, timestamp=pose.timestamp,
                    joint_angles=prev_valid.copy(), reachable=False,
                    ik_residual_m=residual, solve_time_ms=solve_ms,
                    joint_limit_violations=violations,
                    fallback_used=True, fallback_deviation_rad=fb_dev))
        return results

    # -- context manager (drop-in for `with IKSolver(...) as solver`) ------------
    def close(self) -> None:
        self._ready = False

    def __enter__(self) -> "MinkIKSolver":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # -- reporting -----------------------------------------------------------------
    def print_summary(self, results: List[IKResult]) -> None:
        """Print a human-readable summary of IK results (IKSolver parity)."""
        n = len(results)
        n_reach = sum(1 for r in results if r.reachable)
        n_fallback = sum(1 for r in results if r.fallback_used)
        solved = [r for r in results if not np.isnan(r.ik_residual_m)]
        avg_res = np.mean([r.ik_residual_m for r in solved]) if solved else float("nan")
        avg_solve = np.mean([r.solve_time_ms for r in solved]) if solved else float("nan")
        sep = "=" * 70
        print(f"\n{sep}")
        print("  MINK IK SOLVE SUMMARY")
        print(f"{sep}")
        print(f"  Total frames          : {n}")
        print(f"  Reachable             : {n_reach} ({100.0 * n_reach / n:.1f}%)" if n else "  Reachable             : 0")
        print(f"  Fallbacks used        : {n_fallback}")
        print(f"  Avg residual (m)      : {avg_res:.4f}")
        print(f"  Avg solve time (ms)   : {avg_solve:.2f}")
        print(f"  QP solver             : {self._solver_name}")
        print(f"{sep}\n")
