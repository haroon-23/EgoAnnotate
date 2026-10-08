"""Gated physics verifier for exported episodes (Phase E, Stage 9b).

Library version of the ``scripts/physics_replay.py`` protocol. Where the
reference CLI relies on *friction* to lift (and measured 0.017 m < 0.02 m —
friction luck, not a grasp), this verifier uses **weld grasps**
(:mod:`src.retargeting.weld_grasp`): during each validated grasp window the
object is rigidly attached to the end-effector via a MuJoCo weld equality
constraint. Verification then tests *trajectory dynamic feasibility +
attachment stability* rather than friction luck.

Protocol:
  M1 (tracking gate) — replay the exported joint trajectory with position
  actuators. MuJoCo's URDF importer discards ``<limit effort/velocity>``, so
  actuator gains are set explicitly (same KP/KV as ``physics_replay.py``).
  ``mean(max|q - q_cmd|)`` over reachable frames must be strictly below
  ``tracking_err_rad_max``.
  Per-window weld trials — for each grasp window, place the object at the
  demonstrated grasp point, replay to the grasp frame, activate the weld
  equality (capturing the current EE-relative pose into ``eq_data``), and
  measure lift + slip through ``window_end + 15`` frames.

``passed`` = AND of all checks. The verifier NEVER drops episodes — it only
sets the ``physics_verified`` label; the pipeline keeps all data either way.

``scripts/physics_replay.py`` is untouched and remains the reference CLI.
``scripts/hero_clip.py`` reuses :func:`build_verification_scene`,
:func:`replay_frame`, :func:`activate_weld` and :func:`deactivate_weld`.
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .weld_grasp import WeldGraspConfig, GraspWindow, find_grasp_windows

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Actuator gains — MuJoCo's URDF importer discards <limit effort/velocity>,
# so gains are set explicitly. Same values as scripts/physics_replay.py.
# ---------------------------------------------------------------------------
_VERIFIER_KP = [1000, 1000, 750, 750, 300, 150, 100]
_VERIFIER_KV = [100, 100, 75, 75, 30, 15, 10]
_VERIFIER_KF = 400.0
_VERIFIER_KVF = 20.0
_VERIFIER_TABLE_TOP = 0.25  # matches scripts/physics_replay.py table geometry
_VERIFIER_TIMESTEP = 0.00416667  # 1/240 s, matches scripts/physics_replay.py
_VERIFIER_FPS = 30.0  # substeps/frame assume 30 fps HDF5, like the reference CLI
_WELD_NAME = "ver_grasp_weld"
_EE_FALLBACKS = ("panda_link8", "panda_hand", "panda_link7", "link8", "hand", "link7")


def _mujoco_import_status() -> Tuple[bool, str]:
    """Return (available, reason). Never raises."""
    try:
        import mujoco  # noqa: F401
    except ImportError as exc:
        return False, f"mujoco not installed ({exc}). Install it to enable physics verification."
    return True, ""


# ---------------------------------------------------------------------------
# Pure-numpy quaternion helpers (MuJoCo wxyz convention). scipy is a lazy
# dependency of the retargeting stack, so the verifier does not require it.
# ---------------------------------------------------------------------------
def _quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).ravel()
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_rotate_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by unit quaternion q (wxyz)."""
    q = np.asarray(q, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    qv = np.array([0.0, v[0], v[1], v[2]])
    out = _quat_mul_wxyz(_quat_mul_wxyz(q, qv), _quat_conj_wxyz(q))
    return out[1:]


# ---------------------------------------------------------------------------
# Config / report dataclasses
# ---------------------------------------------------------------------------
@dataclass
class VerifierConfig:
    """Configuration for the physics verifier.

    Attributes:
        tracking_err_rad_max: M1 gate — strict ``<`` comparison.
        lift_min_m: Minimum object lift per weld trial (>= passes).
        grasp_proximity_m: 2D EE-to-object distance required at the grasp frame.
        min_reachable_pct: Minimum % of reachable frames (>= passes).
        max_self_collision_frames: Allowed robot self-collision frames (<=).
        timestep_s: MuJoCo option timestep (1/240 like the reference CLI).
        weld: WeldGraspConfig for window detection + slip tolerance.
        urdf_path: Robot URDF. Falls back to the retargeting config URDF when
            the pipeline wires it; ``verify_episode`` also accepts an override.
        ee_link_name: End-effector body name; falls back through
            ``_EE_FALLBACKS`` when not found in the model.
    """

    tracking_err_rad_max: float = 0.15
    lift_min_m: float = 0.02
    grasp_proximity_m: float = 0.09
    min_reachable_pct: float = 50.0
    max_self_collision_frames: int = 0
    timestep_s: float = _VERIFIER_TIMESTEP
    weld: WeldGraspConfig = field(default_factory=WeldGraspConfig)
    urdf_path: Optional[str] = None
    ee_link_name: str = "panda_link8"


@dataclass
class CheckResult:
    name: str
    passed: bool
    value: float
    threshold: float
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "value": round(float(self.value), 6),
            "threshold": float(self.threshold),
            "detail": self.detail,
        }


@dataclass
class VerificationReport:
    episode_id: str
    passed: bool
    checks: List[CheckResult] = field(default_factory=list)
    windows: List[GraspWindow] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    report_path: Optional[str] = None

    def save_json(self, path: str | os.PathLike) -> str:
        payload = {
            "episode_id": self.episode_id,
            "passed": bool(self.passed),
            "checks": [c.to_dict() for c in self.checks],
            "windows": [w.to_dict() for w in self.windows],
            "metrics": {k: round(float(v), 6) for k, v in self.metrics.items()},
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2))
        self.report_path = str(p)
        return str(p)


@dataclass
class SceneInfo:
    """Resolved indices into a MuJoCo verification scene."""

    mj: object  # the lazily imported mujoco module
    arm_q: np.ndarray  # qpos indices of the 7 arm joints
    arm_dof: np.ndarray  # qvel/dof indices of the 7 arm joints
    fing_q: np.ndarray  # qpos indices of the 2 finger joints
    arm_ctrl: np.ndarray  # ctrl indices of the arm actuators
    fing_ctrl: np.ndarray  # ctrl indices of the finger actuators
    ee_body_id: int
    obj_body_id: int
    table_body_id: int
    obj_qpos_adr: int
    weld_eq_id: int
    substeps: int


# ---------------------------------------------------------------------------
# Scene construction (shared with scripts/hero_clip.py)
# ---------------------------------------------------------------------------
def build_verification_scene(
    urdf_path: str,
    cache_dir: Optional[str | os.PathLike] = None,
    ee_link_name: str = "panda_link8",
) -> Tuple[object, SceneInfo]:
    """Build the MuJoCo verification scene for ``urdf_path``.

    Mirrors ``scripts/physics_replay.py::build_scene`` (same actuator gains,
    table/object geometry, timestep) and additionally splices a weld equality
    ``ver_grasp_weld`` between the end-effector body and the object. The weld
    starts INACTIVE (``eq_active = 0``); trials toggle it per grasp window.

    Unlike the reference CLI (which writes next to the URDF), cache/scene XML
    go under ``cache_dir`` (default: the system temp dir) so library use never
    writes next to user files.

    Returns:
        (model, SceneInfo).

    Raises:
        ImportError: mujoco is not installed.
        FileNotFoundError: ``urdf_path`` does not exist.
        RuntimeError: the spliced scene XML fails sanity checks, or a
            required body/joint cannot be resolved.
    """
    ok, reason = _mujoco_import_status()
    if not ok:
        raise ImportError(reason)
    import mujoco as mj

    urdf = Path(urdf_path)
    if not urdf.exists():
        raise FileNotFoundError(f"Verifier URDF not found: {urdf_path}")
    cdir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "egoannotate_physics"
    cdir.mkdir(parents=True, exist_ok=True)
    m0 = mj.MjModel.from_xml_path(str(urdf))
    cache = cdir / f"{urdf.stem}_verifier_cache.xml"
    if not cache.exists():
        mj.mj_saveLastXML(str(cache), m0)
    txt = cache.read_text()

    body_names = []
    if hasattr(mj, "mj_id2name") and hasattr(mj, "mjtObj"):
        try:
            body_names = [mj.mj_id2name(m0, mj.mjtObj.mjOBJ_BODY, i) for i in range(getattr(m0, "nbody", 0))]
        except Exception:
            pass
    elif hasattr(m0, "body"):
        try:
            body_names = [m0.body(i).name for i in range(getattr(m0, "nbody", 0))]
        except Exception:
            pass
    body_names = [b for b in body_names if b]

    if ee_link_name not in body_names and body_names:
        for fb in list(_EE_FALLBACKS) + ["panda_link7", "panda_hand", "link7", "hand", "ee_link"]:
            if fb in body_names:
                logger.warning("Verifier: EE body %r not found; using fallback %r", ee_link_name, fb)
                ee_link_name = fb
                break

    names = re.findall(r'joint name="([^"]+)"', txt)
    arm = [j for j in names if "finger" not in j][:7]
    fing = [j for j in names if "finger" in j][:2]
    if len(arm) < 7:
        raise RuntimeError(
            f"Verifier needs 7 non-finger arm joints in {urdf_path}; found {len(arm)}: {arm}"
        )

    acts = "\n".join(
        [f'    <position name="ver_q{i+1}" joint="{j}" kp="{k}" kv="{v}"/>'
         for i, (j, k, v) in enumerate(zip(arm, _VERIFIER_KP, _VERIFIER_KV))]
        + [f'    <position name="ver_f{i+1}" joint="{j}" kp="{_VERIFIER_KF:.0f}" kv="{_VERIFIER_KVF}"/>'
           for i, j in enumerate(fing)]
    )
    # Table + free object — identical geometry to scripts/physics_replay.py.
    body = (
        f'\n    <body name="table" pos="0.5 0 {_VERIFIER_TABLE_TOP / 2}"><geom type="box" '
        f'size="0.35 0.30 {_VERIFIER_TABLE_TOP / 2}" rgba="0.45 0.3 0.15 1"/></body>'
        '\n    <body name="obj" pos="0.5 0 0.31"><freejoint name="obj_joint"/>'
        '<geom type="cylinder" size="0.030 0.060" mass="0.12" '
        'friction="1.2 0.005 0.0001" rgba="0.8 0.1 0.1 1"/></body>\n  '
    )
    weld = (
        f'\n  <equality><weld name="{_WELD_NAME}" body1="{ee_link_name}" '
        f'body2="obj"/></equality>'
    )
    # Strip any pre-existing equality block (e.g. from a stale cache), then
    # splice option timestep, bodies, actuators and the weld equality.
    txt = re.sub(r"<equality>.*?</equality>", "", txt, flags=re.S)
    txt = re.sub(
        r"<mujoco[^>]*>",
        lambda m: m.group(0) + f'\n  <option timestep="{_VERIFIER_TIMESTEP}"/>\n  <compiler meshdir="{urdf.parent.resolve()}"/>',
        txt,
        count=1,
    )
    txt = txt.replace("</worldbody>", body + "</worldbody>")
    txt = txt.replace("</mujoco>", f"<actuator>\n{acts}\n</actuator>{weld}\n</mujoco>")
    if txt.count("<equality>") != 1:
        raise RuntimeError("Verifier scene splice failed: expected exactly one <equality> block")
    for token in ('name="obj"', 'name="table"', "</mujoco>", _WELD_NAME):
        if token not in txt:
            raise RuntimeError(f"Verifier scene splice failed: missing {token!r}")
    scene = cdir / f"{urdf.stem}_verifier_scene.xml"
    scene.write_text(txt)
    model = mj.MjModel.from_xml_path(str(scene))

    J, B, E, A = mj.mjtObj.mjOBJ_JOINT, mj.mjtObj.mjOBJ_BODY, mj.mjtObj.mjOBJ_EQUALITY, mj.mjtObj.mjOBJ_ACTUATOR
    arm_q = np.array([model.jnt_qposadr[mj.mj_name2id(model, J, j)] for j in arm], dtype=int)
    arm_dof = np.array([model.jnt_dofadr[mj.mj_name2id(model, J, j)] for j in arm], dtype=int)
    fing_q = np.array([model.jnt_qposadr[mj.mj_name2id(model, J, j)] for j in fing], dtype=int)
    arm_ctrl = np.array(
        [mj.mj_name2id(model, A, f"ver_q{i+1}") for i in range(len(arm))], dtype=int
    )
    fing_ctrl = np.array(
        [mj.mj_name2id(model, A, f"ver_f{i+1}") for i in range(len(fing))], dtype=int
    )
    if np.any(arm_q < 0) or np.any(arm_ctrl < 0):
        raise RuntimeError("Verifier could not resolve arm joints/actuators in the scene")
    ee_id = mj.mj_name2id(model, B, ee_link_name)
    if ee_id < 0:
        for fb in _EE_FALLBACKS:
            ee_id = mj.mj_name2id(model, B, fb)
            if ee_id >= 0:
                logger.warning("Verifier: EE body %r not found; using fallback %r", ee_link_name, fb)
                break
    obj_id = mj.mj_name2id(model, B, "obj")
    table_id = mj.mj_name2id(model, B, "table")
    free_jnts = [i for i in range(model.njnt) if model.jnt_type[i] == mj.mjtJoint.mjJNT_FREE]
    if ee_id < 0 or obj_id < 0 or not free_jnts:
        raise RuntimeError(
            f"Verifier could not resolve scene bodies (ee={ee_id}, obj={obj_id}, "
            f"free joints={len(free_jnts)})"
        )
    weld_id = mj.mj_name2id(model, E, _WELD_NAME)
    if weld_id < 0:
        raise RuntimeError("Verifier could not resolve the weld equality in the scene")
    substeps = max(1, int(round((_VERIFIER_FPS ** -1) / model.opt.timestep)))

    info = SceneInfo(
        mj=mj,
        arm_q=arm_q,
        arm_dof=arm_dof,
        fing_q=fing_q,
        arm_ctrl=arm_ctrl,
        fing_ctrl=fing_ctrl,
        ee_body_id=int(ee_id),
        obj_body_id=int(obj_id),
        table_body_id=int(table_id),
        obj_qpos_adr=int(model.jnt_qposadr[free_jnts[0]]),
        weld_eq_id=int(weld_id),
        substeps=int(substeps),
    )
    return model, info


def replay_frame(mj, model, data, info: SceneInfo, q_i: np.ndarray, g_scalar: float) -> None:
    """Replay one trajectory frame: set position-servo targets, step substeps."""
    data.ctrl[info.arm_ctrl] = np.asarray(q_i, dtype=float).ravel()
    if len(info.fing_ctrl):
        data.ctrl[info.fing_ctrl] = float(g_scalar) / 2.0
    for _ in range(info.substeps):
        mj.mj_step(model, data)


def activate_weld(mj, model, data, info: SceneInfo) -> Tuple[np.ndarray, np.ndarray]:
    """Activate the weld equality at the CURRENT poses.

    Captures the object pose relative to the end-effector into
    ``model.eq_data`` and sets ``eq_active = 1``. Call ``mj_forward`` before
    this so xpos/xquat are current. Returns (rel_pos, rel_quat_wxyz).
    """
    ee_pos = np.array(data.xpos[info.ee_body_id], dtype=float)
    ee_quat = np.array(data.xquat[info.ee_body_id], dtype=float)
    obj_pos = np.array(data.xpos[info.obj_body_id], dtype=float)
    obj_quat = np.array(data.xquat[info.obj_body_id], dtype=float)
    rel_pos = _quat_rotate_wxyz(_quat_conj_wxyz(ee_quat), obj_pos - ee_pos)
    rel_quat = _quat_mul_wxyz(_quat_conj_wxyz(ee_quat), obj_quat)
    model.eq_data[info.weld_eq_id] = np.concatenate([rel_pos, rel_quat])
    data.eq_active[info.weld_eq_id] = 1
    mj.mj_forward(model, data)
    return rel_pos, rel_quat


def deactivate_weld(data, info: SceneInfo) -> None:
    """Release the weld equality (``eq_active = 0``)."""
    data.eq_active[info.weld_eq_id] = 0


def _contact_geom_id(contact, key: str) -> int:
    """Read a geom id from a contact — attribute access (real MuJoCo
    ``MjContact``) or item access (numpy structured void in tests)."""
    try:
        return int(getattr(contact, key))
    except AttributeError:
        return int(contact[key])


def frame_has_self_collision(model, data, robot_body_ids: set) -> bool:
    """True when any contact this step is between two robot bodies."""
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = int(model.geom_bodyid[_contact_geom_id(c, "geom1")])
        b2 = int(model.geom_bodyid[_contact_geom_id(c, "geom2")])
        if b1 != b2 and b1 in robot_body_ids and b2 in robot_body_ids:
            return True
    return False


# ---------------------------------------------------------------------------
# EpisodeVerifier
# ---------------------------------------------------------------------------
class EpisodeVerifier:
    """Gated physics verifier: MuJoCo replay + weld-grasp trials.

    Args:
        config: VerifierConfig. ``urdf_path`` may be left None and supplied
            per-call via ``verify_episode(..., urdf_path=...)``.
    """

    def __init__(self, config: Optional[VerifierConfig] = None):
        self.config = config or VerifierConfig()
        self._mj: Optional[object] = None

    def is_available(self) -> bool:
        """False when mujoco is not installed. Never raises."""
        ok, _ = _mujoco_import_status()
        return ok

    # -- HDF5 loading -----------------------------------------------------
    def _load_episode(self, hdf5_path: str, episode_id: Optional[str]) -> Tuple[str, dict]:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                f"EpisodeVerifier needs h5py to read {hdf5_path}: {exc}"
            ) from exc
        p = Path(hdf5_path)
        if not p.exists():
            # Loud, never a silent pass.
            raise FileNotFoundError(f"Episode HDF5 not found: {hdf5_path}")
        with h5py.File(str(p), "r") as f:
            keys = list(f.keys())
            if not keys:
                raise ValueError(f"Episode HDF5 has no episode groups: {hdf5_path}")
            ep = episode_id if (episode_id is not None and episode_id in f) else keys[0]
            if episode_id is not None and episode_id not in f:
                raise KeyError(
                    f"episode {episode_id!r} not found in {hdf5_path}; have {keys}"
                )
            obs = f[ep]["steps"]["observation"]
            q = np.asarray(obs["robot_joint_angles"], dtype=float)
            g = np.asarray(obs["robot_gripper_opening_m"], dtype=float).reshape(-1)
            # The exporter writes steps/observation/robot_reachable; the old
            # reference CLI read steps/robot_reachable. Prefer the
            # exporter-written key, fall back for legacy files.
            steps = f[ep]["steps"]
            if "robot_reachable" in obs:
                r = np.asarray(obs["robot_reachable"], dtype=bool)
            elif "robot_reachable" in steps:
                r = np.asarray(steps["robot_reachable"], dtype=bool)
            else:
                raise KeyError(
                    f"{hdf5_path}: no robot_reachable under steps/observation or steps"
                )
        n = len(q)
        if not (len(g) == n and len(r) == n):
            raise ValueError(
                f"{hdf5_path}: length mismatch q={len(q)} g={len(g)} r={len(r)}"
            )
        return ep, {"q": q, "g": g, "r": r, "n": n}

    # -- public API ---------------------------------------------------------
    def verify_episode(
        self,
        hdf5_path: str,
        episode_id: Optional[str] = None,
        object_position: Optional[np.ndarray] = None,
        urdf_path: Optional[str] = None,
    ) -> VerificationReport:
        """Verify one exported episode. Never raises for physics outcomes.

        Args:
            hdf5_path: Exported ``episode_rlds.hdf5`` (exporter layout:
                ``<ep>/steps/observation/{robot_joint_angles,
                robot_gripper_opening_m, robot_reachable}``).
            episode_id: Episode group; defaults to the first group.
            object_position: (3,) world position of the grasped object. When
                None, derived from the end-effector position at the first
                closed-and-reachable frame (object assumed where grasped).
            urdf_path: Overrides ``config.urdf_path``.

        Returns:
            VerificationReport with ``passed`` = AND of all checks. When
            mujoco is unavailable the report fails with an explicit
            ``mujoco_available`` check — never a silent pass.

        Raises:
            FileNotFoundError: the HDF5 (or URDF) does not exist.
            KeyError / ValueError: malformed HDF5 or unresolved episode.
            ImportError: h5py missing.
        """
        cfg = self.config
        ep_id = episode_id or Path(hdf5_path).parent.name

        def _fail(reason: str) -> VerificationReport:
            logger.error("[Verifier] %s: %s", ep_id, reason)
            return VerificationReport(
                episode_id=ep_id,
                passed=False,
                checks=[CheckResult("mujoco_available", False, 0.0, 1.0, reason)],
                metrics={},
            )

        if not self.is_available():
            return _fail(_mujoco_import_status()[1])

        ep, arrays = self._load_episode(hdf5_path, episode_id)
        ep_id = ep
        q, g, r, n = arrays["q"], arrays["g"], arrays["r"], arrays["n"]

        urdf = urdf_path or cfg.urdf_path
        if not urdf:
            raise ValueError(
                "EpisodeVerifier needs a URDF: set VerifierConfig.urdf_path or pass "
                "verify_episode(..., urdf_path=...)."
            )
        mj = self._lazy_mj()
        model, info = build_verification_scene(urdf, ee_link_name=cfg.ee_link_name)
        data = mj.MjData(model)
        data.eq_active[info.weld_eq_id] = 0  # weld starts inactive

        robot_body_ids = set(range(model.nbody)) - {0, info.table_body_id, info.obj_body_id}

        # -- M1: tracking gate ------------------------------------------------
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
        errs: List[float] = []
        ee_positions = np.zeros((n, 3))
        last_ee = np.array(data.xpos[info.ee_body_id], dtype=float)
        coll_frames = 0
        n_reach = 0
        prev_reachable = False
        for i in range(n):
            if not r[i]:
                ee_positions[i] = last_ee
                prev_reachable = False
                continue
            n_reach += 1
            if not prev_reachable:
                # Start of a reachable segment: place the sim exactly at the
                # commanded pose (zero velocity) so M1 measures in-segment
                # tracking quality, not teleportation across unreachable gaps.
                data.qpos[info.arm_q] = q[i]
                data.qvel[info.arm_dof] = 0.0
                mj.mj_forward(model, data)
            replay_frame(mj, model, data, info, q[i], float(g[i]))
            errs.append(float(np.max(np.abs(data.qpos[info.arm_q] - q[i]))))
            last_ee = np.array(data.xpos[info.ee_body_id], dtype=float)
            ee_positions[i] = last_ee
            if frame_has_self_collision(model, data, robot_body_ids):
                coll_frames += 1
            prev_reachable = True
        track = float(np.mean(errs)) if errs else 9.9
        logger.info("[Verifier] M1 tracking err = %.4f rad over %d reachable frames", track, n_reach)

        # -- grasp windows ----------------------------------------------------
        if object_position is None:
            closed_reach = np.where(r & (g < cfg.weld.close_threshold_m))[0]
            if len(closed_reach):
                object_position = ee_positions[closed_reach[0]].copy()
                logger.info("[Verifier] object_position derived from EE at frame %d: %s",
                            closed_reach[0], np.round(object_position, 3))
            else:
                object_position = last_ee.copy()
                logger.warning("[Verifier] no closed+reachable frame; object assumed at EE home")
        windows = find_grasp_windows(r, g, ee_positions, np.asarray(object_position, float), cfg.weld)
        logger.info("[Verifier] %d grasp window(s) found", len(windows))

        # -- per-window weld trials -------------------------------------------
        reachable_pct = 100.0 * n_reach / n if n else 0.0
        checks: List[CheckResult] = [
            CheckResult("m1_tracking_err_rad", track < cfg.tracking_err_rad_max,
                        track, cfg.tracking_err_rad_max,
                        f"mean max|q-q_cmd| over {n_reach} reachable frames"),
            CheckResult("reachable_pct", reachable_pct >= cfg.min_reachable_pct,
                        reachable_pct, cfg.min_reachable_pct,
                        f"{n_reach}/{n} frames kinematically reachable"),
            CheckResult("self_collision_frames", coll_frames <= cfg.max_self_collision_frames,
                        float(coll_frames), float(cfg.max_self_collision_frames),
                        "robot-robot contacts during M1 replay"),
            CheckResult("grasp_windows_found", len(windows) >= 1,
                        float(len(windows)), 1.0, ""),
        ]

        best_lift = 0.0
        for k, w in enumerate(windows):
            # Reference-protocol parity: skip windows where the arm never left home.
            mid = (w.start_idx + w.end_idx) // 2
            if np.all(np.abs(q[mid]) < 0.1):
                detail = "skipped: arm at home position (reference-protocol parity)"
                logger.warning("[Verifier] window %d [%d,%d): %s", k, w.start_idx, w.end_idx, detail)
                checks.append(CheckResult(f"window_{k}_weld_held", False, 0.0,
                                          cfg.weld.slip_tolerance_m, detail))
                checks.append(CheckResult(f"window_{k}_lift_m", False, 0.0,
                                          cfg.lift_min_m, detail))
                continue
            slip, lift, near = self._run_weld_trial(mj, model, data, info, q, g, w)
            w.peak_lift_m = lift
            w.weld_held = bool(slip <= cfg.weld.slip_tolerance_m)
            best_lift = max(best_lift, lift)
            checks.append(CheckResult(
                f"window_{k}_weld_held", w.weld_held, slip, cfg.weld.slip_tolerance_m,
                f"max slip {slip*1000:.1f} mm over [{w.start_idx},{w.end_idx})"))
            checks.append(CheckResult(
                f"window_{k}_lift_m", lift >= cfg.lift_min_m, lift, cfg.lift_min_m,
                f"peak object lift in trial"))
            checks.append(CheckResult(
                f"window_{k}_near_ee_m",
                near < cfg.grasp_proximity_m, near, cfg.grasp_proximity_m,
                "2D EE-to-object distance at grasp frame"))

        passed = bool(checks) and all(c.passed for c in checks)
        metrics = {
            "tracking_err_rad": track,
            "reachable_pct": 100.0 * n_reach / n if n else 0.0,
            "self_collision_frames": float(coll_frames),
            "n_windows": float(len(windows)),
            "best_lift_m": best_lift,
            "n_frames": float(n),
        }
        report = VerificationReport(episode_id=ep_id, passed=passed, checks=checks,
                                    windows=windows, metrics=metrics)
        logger.info("[Verifier] episode %s: %s (%d/%d checks)",
                    ep_id, "PASSED" if passed else "FAILED",
                    sum(c.passed for c in checks), len(checks))
        return report

    # -- weld trial -----------------------------------------------------------
    def _run_weld_trial(self, mj, model, data, info: SceneInfo,
                        q: np.ndarray, g: np.ndarray,
                        window: GraspWindow) -> Tuple[float, float, float]:
        """Run one weld trial. Returns (max_slip_m, max_lift_m, near_ee_m)."""
        cfg = self.config
        n = len(q)
        gf, i0, i1 = window.grasp_frame_idx, window.start_idx, window.end_idx
        trial_end = min(i1 + 15, n)

        mj.mj_resetData(model, data)
        data.eq_active[info.weld_eq_id] = 0
        qa = info.obj_qpos_adr
        data.qpos[qa:qa + 3] = window.object_spawn_pos
        data.qpos[qa + 3:qa + 7] = [1.0, 0.0, 0.0, 0.0]
        mj.mj_forward(model, data)
        z0 = float(window.object_spawn_pos[2])

        # Pre-roll to the grasp frame so the arm settles at the grasp pose.
        for i in range(i0, gf):
            replay_frame(mj, model, data, info, q[i], float(g[i]))
        replay_frame(mj, model, data, info, q[gf], float(g[gf]))
        mj.mj_forward(model, data)

        rel_pos, rel_quat = activate_weld(mj, model, data, info)
        near = float(np.linalg.norm(data.xpos[info.obj_body_id][:2]
                                    - data.xpos[info.ee_body_id][:2]))
        logger.info("[Verifier] window [%d,%d): weld on at frame %d (near=%.3f m)",
                    i0, i1, gf, near)

        max_slip, max_lift = 0.0, 0.0
        for i in range(gf, trial_end):
            replay_frame(mj, model, data, info, q[i], float(g[i]))
            ee_pos = np.array(data.xpos[info.ee_body_id], dtype=float)
            ee_quat = np.array(data.xquat[info.ee_body_id], dtype=float)
            obj_pos = np.array(data.xpos[info.obj_body_id], dtype=float)
            max_lift = max(max_lift, float(obj_pos[2] - z0))
            expected = ee_pos + _quat_rotate_wxyz(ee_quat, rel_pos)
            max_slip = max(max_slip, float(np.linalg.norm(obj_pos - expected)))
        deactivate_weld(data, info)
        logger.info("[Verifier] window [%d,%d): lift=%.4f m slip=%.4f m held=%s",
                    i0, i1, max_lift, max_slip, max_slip <= cfg.weld.slip_tolerance_m)
        return max_slip, max_lift, near

    # -- internals --------------------------------------------------------------
    def _lazy_mj(self):
        if self._mj is None:
            import mujoco as mj  # guarded by is_available() before any use
            self._mj = mj
        return self._mj
