"""Embodiment configuration and model loading (docs/ARCHITECTURE.md section 6).

An embodiment is a YAML file (see ``configs/embodiments``) that points to any
MJCF/URDF model and says which joints are controlled, where the end-effector
frames are and how the human hand frame ``H`` maps to each robot tool frame.
Nothing here is robot specific: Menagerie models, a user's own MJCF or a URDF
all go through the same loader.

Conventions: ``tcp_offset`` is ``T_H_ee``, the pose of the robot tool frame
expressed in the human hand frame H, so the IK target is
``T_B_W @ T_W_H @ tcp_offset``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "embodiments"
DEFAULT_MODELS_ROOT = REPO_ROOT / "data" / "robots" / "mujoco_menagerie"

_FRAME_TYPES = {"site", "body", "geom"}
_DEFAULT_TOLERANCES = {"pos_m": 0.01, "ori_deg": 15.0, "fingertip_m": 0.015}


class EmbodimentError(ValueError):
    """Invalid or inconsistent embodiment configuration."""


def models_root() -> Path:
    return Path(os.environ.get("EGOANNOTATE_MENAGERIE", DEFAULT_MODELS_ROOT))


def _pose_to_matrix(spec: dict | None) -> np.ndarray:
    """Build a 4x4 from ``{pos, rot | quat_wxyz | rpy_deg}``."""
    spec = spec or {}
    T = np.eye(4)
    T[:3, 3] = np.asarray(spec.get("pos", [0.0, 0.0, 0.0]), dtype=float)
    if "rot" in spec:
        R = np.asarray(spec["rot"], dtype=float).reshape(3, 3)
    elif "quat_wxyz" in spec:
        w, x, y, z = spec["quat_wxyz"]
        R = Rotation.from_quat([x, y, z, w]).as_matrix()
    elif "rpy_deg" in spec:
        R = Rotation.from_euler("xyz", spec["rpy_deg"], degrees=True).as_matrix()
    else:
        R = np.eye(3)
    if abs(np.linalg.det(R) - 1.0) > 1e-6 or np.abs(R.T @ R - np.eye(3)).max() > 1e-6:
        raise EmbodimentError(f"tcp_offset rotation is not a proper rotation: {R}")
    T[:3, :3] = R
    return T


@dataclass
class GripperSpec:
    type: str = "none"  # "parallel" | "none"
    joints: list[str] = field(default_factory=list)
    stroke_m: float = 0.0  # total opening range (sum over both fingers)
    actuator: str | None = None  # drives opening in replay (ctrl linear in opening)


@dataclass
class FingerSpec:
    name: str
    human_tip: int  # MediaPipe landmark index
    frame: str
    frame_type: str = "body"


@dataclass
class DexterousSpec:
    fingers: list[FingerSpec]
    hand_length_frames: tuple[str, str]  # robot wrist -> middle MCP bodies
    hand_length_type: str = "body"


@dataclass
class EndEffector:
    hand: str  # "right" | "left": which human hand drives it
    frame: str
    frame_type: str
    tcp_offset: np.ndarray  # T_H_ee, 4x4
    tracking: str = "pose"  # "pose" | "position" (no wrist orientation DoF)
    workspace_center: np.ndarray | None = None  # in B, used by auto placement
    pos_cost: float = 1.0
    ori_cost: float = 0.5
    gripper: GripperSpec = field(default_factory=GripperSpec)
    dexterous: DexterousSpec | None = None


@dataclass
class Embodiment:
    name: str
    kind: str
    license: str
    model: mujoco.MjModel  # fixed-base model used for IK and collision checks
    config: dict
    joint_names: list[str]  # controlled joints, in q order
    q_adr: np.ndarray  # qpos address of each controlled joint
    v_adr: np.ndarray  # dof address of each controlled joint
    joint_limits: np.ndarray  # (nc, 2)
    home_q: np.ndarray  # (model.nq,) locked posture + controlled home
    velocity_limits: np.ndarray  # (nc,)
    acceleration_limits: np.ndarray  # (nc,)
    jerk_limits: np.ndarray | None
    tolerances: dict
    end_effectors: dict[str, EndEffector]
    collision: dict
    ik: dict
    replay_cfg: dict
    mount: dict | None
    notes: list[str]
    source_path: str | None = None
    _allowed_pairs: set = field(default_factory=set, repr=False)

    # ---- joint bookkeeping -------------------------------------------------
    @property
    def nc(self) -> int:
        return len(self.joint_names)

    @property
    def position_only(self) -> bool:
        return any(e.tracking == "position" for e in self.end_effectors.values())

    def full_qpos(self, q_ctrl: np.ndarray, gripper_m: dict[str, float] | None = None) -> np.ndarray:
        """Full-model qpos from controlled joint values (locked joints at posture)."""
        q = self.home_q.copy()
        q[self.q_adr] = q_ctrl
        for hand, opening in (gripper_m or {}).items():
            g = self.end_effectors[hand].gripper
            if g.type == "parallel" and opening == opening:
                for j in g.joints:
                    q[self.model.joint(j).qposadr[0]] = opening / len(g.joints)
        return q

    def joint_violations(self, q_ctrl: np.ndarray, tol: float = 1e-5) -> np.ndarray:
        """Boolean (T, nc): value outside the model joint range (never clipped)."""
        lo, hi = self.joint_limits[:, 0], self.joint_limits[:, 1]
        return (q_ctrl < lo - tol) | (q_ctrl > hi + tol)

    # ---- collision ---------------------------------------------------------
    def body_name(self, bid: int) -> str:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body{bid}"

    def contact_pairs(self, model: mujoco.MjModel, data: mujoco.MjData) -> set[frozenset]:
        """Body-name pairs in penetration, from MuJoCo contacts (data must hold collisions)."""
        pairs = set()
        for i in range(data.ncon):
            c = data.contact[i]
            if c.dist >= 0.0:
                continue
            b1 = int(model.geom_bodyid[c.geom1])
            b2 = int(model.geom_bodyid[c.geom2])
            if b1 == b2 or b1 == 0 or b2 == 0:
                continue
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or f"body{b1}"
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or f"body{b2}"
            pairs.add(frozenset((n1, n2)))
        return pairs

    def self_collisions(self, model: mujoco.MjModel, data: mujoco.MjData) -> list[tuple[str, str]]:
        """Non-adjacent body pairs in contact; pairs ignored by config or at home are skipped."""
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        found = self.contact_pairs(model, data) - self._allowed_pairs
        return sorted(tuple(sorted(p)) for p in found)

    # ---- model variants ----------------------------------------------------
    def replay_model(self) -> mujoco.MjModel:
        """Same model with its own actuators, plus a free 6-DoF mount if configured."""
        return _compile(self.config, self.source_path, replay=True)[0]


# --------------------------------------------------------------------------- loading


def _resolve_model_path(path: str, source_dir: Path | None) -> Path:
    p = Path(path)
    candidates = [p] if p.is_absolute() else [
        models_root() / p,
        REPO_ROOT / p,
        (source_dir / p) if source_dir else p,
        Path.cwd() / p,
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise EmbodimentError(
        f"model file {path!r} not found (tried {[str(c) for c in candidates]}); "
        "run scripts/fetch_robot_models.py for Menagerie models"
    )


def _compile(cfg: dict, source_path: str | None, replay: bool = False):
    """Compile the configured model. Returns (model, original_model_for_keyframes)."""
    mcfg = cfg["model"]
    src_dir = Path(source_path).parent if source_path else None
    path = _resolve_model_path(mcfg["path"], src_dir)
    original = mujoco.MjSpec.from_file(str(path)).compile()
    spec = mujoco.MjSpec.from_file(str(path))
    if mcfg.get("fixed_base", False):
        for j in list(spec.joints):
            if j.type == mujoco.mjtJoint.mjJNT_FREE:
                spec.delete(j)
    for k in list(spec.keys):
        spec.delete(k)
    for s in mcfg.get("sites", []):
        spec.body(s["body"]).add_site(
            name=s["name"],
            pos=s.get("pos", [0, 0, 0]),
            quat=s.get("quat_wxyz", [1, 0, 0, 0]),
        )
    mount = cfg.get("mount")
    if replay and mount and mount.get("type") == "free6":
        spec.body(mount["body"]).add_freejoint()
    return spec.compile(), original


def _per_joint(value: Any, names: list[str], what: str) -> np.ndarray:
    if value is None:
        raise EmbodimentError(f"limits.{what} is required")
    if isinstance(value, dict):
        missing = [n for n in names if n not in value]
        if missing:
            raise EmbodimentError(f"limits.{what} missing joints {missing}")
        return np.array([float(value[n]) for n in names])
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    if arr.size == 1:
        return np.full(len(names), float(arr[0]))
    if arr.size != len(names):
        raise EmbodimentError(f"limits.{what} has {arr.size} entries for {len(names)} controlled joints")
    return arr


def _parse_end_effector(hand: str, e: dict) -> EndEffector:
    ftype = e.get("frame_type", "site")
    if ftype not in _FRAME_TYPES:
        raise EmbodimentError(f"frame_type must be one of {_FRAME_TYPES}")
    tracking = e.get("tracking", "pose")
    if tracking not in {"pose", "position"}:
        raise EmbodimentError("tracking must be 'pose' or 'position'")
    g = e.get("gripper") or {"type": "none"}
    gripper = GripperSpec(
        type=g.get("type", "none"),
        joints=list(g.get("joints", [])),
        stroke_m=float(g.get("stroke_m", 0.0)),
        actuator=g.get("actuator"),
    )
    if gripper.type not in {"parallel", "none"}:
        raise EmbodimentError(f"unknown gripper type {gripper.type!r}")
    if gripper.type == "parallel" and (not gripper.joints or gripper.stroke_m <= 0):
        raise EmbodimentError("parallel gripper needs joints and stroke_m > 0")
    dex = None
    if e.get("dexterous"):
        d = e["dexterous"]
        dex = DexterousSpec(
            fingers=[
                FingerSpec(n, int(f["human_tip"]), f["frame"], f.get("frame_type", "body"))
                for n, f in d["fingers"].items()
            ],
            hand_length_frames=tuple(d["hand_length_frames"]),
            hand_length_type=d.get("hand_length_type", "body"),
        )
    wc = e.get("workspace_center")
    return EndEffector(
        hand=hand,
        frame=e["frame"],
        frame_type=ftype,
        tcp_offset=_pose_to_matrix(e.get("tcp_offset")),
        tracking=tracking,
        workspace_center=None if wc is None else np.asarray(wc, dtype=float),
        pos_cost=float(e.get("pos_cost", 1.0)),
        ori_cost=float(e.get("ori_cost", 0.5)),
        gripper=gripper,
        dexterous=dex,
    )


def load_embodiment(source: str | Path | dict) -> Embodiment:
    """Load an embodiment from a YAML path, a config name (``franka_arm``) or a dict."""
    source_path: str | None = None
    if isinstance(source, dict):
        cfg = source
    else:
        p = Path(source)
        if not p.is_file():
            p = CONFIG_DIR / (str(source) if str(source).endswith(".yaml") else f"{source}.yaml")
        if not p.is_file():
            p = REPO_ROOT / str(source)
        if not p.is_file():
            raise EmbodimentError(f"embodiment config {source!r} not found")
        source_path = str(p)
        cfg = yaml.safe_load(p.read_text())
    for key in ("name", "kind", "model", "controlled_joints", "end_effectors", "limits"):
        if key not in cfg:
            raise EmbodimentError(f"embodiment config missing {key!r}")
    if cfg["kind"] not in {"arm", "bimanual", "hand"}:
        raise EmbodimentError("kind must be arm, bimanual or hand")

    model, original = _compile(cfg, source_path)

    # Posture: keyframe values (read from the unmodified model by joint name), then overrides.
    q_home = model.qpos0.copy()
    posture = cfg.get("posture", {}) or {}
    values: dict[str, float] = {}
    if posture.get("keyframe"):
        kid = mujoco.mj_name2id(original, mujoco.mjtObj.mjOBJ_KEY, posture["keyframe"])
        if kid < 0:
            raise EmbodimentError(f"keyframe {posture['keyframe']!r} not in model")
        for j in range(original.njnt):
            name = mujoco.mj_id2name(original, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name and original.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                values[name] = float(original.key_qpos[kid][original.jnt_qposadr[j]])
    values.update({k: float(v) for k, v in (posture.get("joints") or {}).items()})
    for name, val in values.items():
        try:
            jid = model.joint(name).id
        except KeyError:
            continue  # keyframe joint removed by fixed_base
        q_home[model.jnt_qposadr[jid]] = val

    names = list(cfg["controlled_joints"])
    for n in names:
        try:
            model.joint(n)
        except KeyError as exc:
            raise EmbodimentError(f"controlled joint {n!r} not in model") from exc
    jids = [model.joint(n).id for n in names]
    if any(model.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE) for j in jids):
        raise EmbodimentError("controlled joints must be hinge or slide joints")
    q_adr = np.array([model.jnt_qposadr[j] for j in jids])
    v_adr = np.array([model.jnt_dofadr[j] for j in jids])
    limits = model.jnt_range[jids].copy()
    if not np.all(model.jnt_limited[jids]):
        raise EmbodimentError("every controlled joint must have a model range (no unlimited joints)")
    if np.any(q_home[q_adr] < limits[:, 0]) or np.any(q_home[q_adr] > limits[:, 1]):
        raise EmbodimentError("home posture is outside joint limits")

    lim = cfg["limits"]
    jerk = lim.get("jerk")
    ees = {h: _parse_end_effector(h, e) for h, e in cfg["end_effectors"].items()}
    for h in ees:
        if h not in ("right", "left"):
            raise EmbodimentError("end_effectors keys must be 'right' or 'left'")
    for e in ees.values():
        for fname, ftype in [(e.frame, e.frame_type)] + (
            [(f.frame, f.frame_type) for f in e.dexterous.fingers] if e.dexterous else []
        ):
            obj = {"site": mujoco.mjtObj.mjOBJ_SITE, "body": mujoco.mjtObj.mjOBJ_BODY, "geom": mujoco.mjtObj.mjOBJ_GEOM}[ftype]
            if mujoco.mj_name2id(model, obj, fname) < 0:
                raise EmbodimentError(f"{ftype} {fname!r} not in model")

    tol = dict(_DEFAULT_TOLERANCES)
    tol.update(cfg.get("tolerances") or {})

    emb = Embodiment(
        name=cfg["name"],
        kind=cfg["kind"],
        license=str(cfg.get("license", "")),
        model=model,
        config=cfg,
        joint_names=names,
        q_adr=q_adr,
        v_adr=v_adr,
        joint_limits=limits,
        home_q=q_home,
        velocity_limits=_per_joint(lim.get("velocity"), names, "velocity"),
        acceleration_limits=_per_joint(lim.get("acceleration"), names, "acceleration"),
        jerk_limits=None if jerk is None else _per_joint(jerk, names, "jerk"),
        tolerances=tol,
        end_effectors=ees,
        collision=cfg.get("collision") or {},
        ik=cfg.get("ik") or {},
        replay_cfg=cfg.get("replay") or {},
        mount=cfg.get("mount"),
        notes=list(cfg.get("notes", [])),
        source_path=source_path,
    )
    _init_allowed_pairs(emb)
    return emb


def _init_allowed_pairs(emb: Embodiment) -> None:
    """Pairs ignored by config, plus pairs already in contact at the home posture."""
    allowed = {frozenset(p) for p in emb.collision.get("ignore_pairs", [])}
    if emb.collision.get("ignore_at_home", True):
        data = mujoco.MjData(emb.model)
        data.qpos[:] = emb.home_q
        mujoco.mj_kinematics(emb.model, data)
        mujoco.mj_collision(emb.model, data)
        allowed |= emb.contact_pairs(emb.model, data)
    emb._allowed_pairs = allowed
