"""Pure YAML + URDF/MJCF config-driven multi-embodiment IK retargeting & feasibility auditor."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from src.egoannotate.config import RetargetingConfig
from src.egoannotate.schemas import AtomicActionSegment, FrameAnnotation


@dataclass(frozen=True)
class KinematicJoint:
    name: str
    joint_type: str
    parent_link: str
    child_link: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


@dataclass
class EmbodimentTrajectory:
    name: str
    embodiment_class: str
    joint_names: list[str]
    qpos: np.ndarray
    actions: np.ndarray
    ee_target_m: np.ndarray
    ee_achieved_m: np.ndarray
    fk_pos_error_mm: np.ndarray
    metrics: dict[str, Any]
    segment_feasibility: list[dict[str, Any]]


def _parse_floats(raw: str | None, default: list[float]) -> np.ndarray:
    if not raw:
        return np.asarray(default, dtype=np.float64)
    vals = [float(x) for x in raw.strip().split()]
    return np.asarray(vals, dtype=np.float64)


def parse_urdf_kinematic_chains(urdf_path: Path) -> tuple[list[KinematicJoint], dict[str, list[KinematicJoint]]]:
    """Parse any standard URDF XML into articulated joints and root-to-leaf kinematic chains."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joints: list[KinematicJoint] = []
    by_child: dict[str, KinematicJoint] = {}
    parent_links: set[str] = set()
    child_links: set[str] = set()

    for j_el in root.findall("joint"):
        j_type = j_el.attrib.get("type", "fixed")
        j_name = j_el.attrib.get("name", "joint")
        p_el = j_el.find("parent")
        c_el = j_el.find("child")
        if p_el is None or c_el is None:
            continue
        p_link = p_el.attrib["link"]
        c_link = c_el.attrib["link"]
        parent_links.add(p_link)
        child_links.add(c_link)

        orig = j_el.find("origin")
        xyz = _parse_floats(orig.attrib.get("xyz") if orig is not None else None, [0.0, 0.0, 0.0])
        rpy = _parse_floats(orig.attrib.get("rpy") if orig is not None else None, [0.0, 0.0, 0.0])

        ax_el = j_el.find("axis")
        axis = _parse_floats(ax_el.attrib.get("xyz") if ax_el is not None else None, [0.0, 0.0, 1.0])
        ax_n = float(np.linalg.norm(axis))
        if ax_n > 1e-6:
            axis = axis / ax_n

        lim = j_el.find("limit")
        lower = float(lim.attrib.get("lower", -2.8)) if lim is not None else -2.8
        upper = float(lim.attrib.get("upper", 2.8)) if lim is not None else 2.8
        if lower >= upper:
            lower, upper = -2.8, 2.8

        kj = KinematicJoint(
            name=j_name,
            joint_type=j_type,
            parent_link=p_link,
            child_link=c_link,
            origin_xyz=xyz,
            origin_rpy=rpy,
            axis=axis,
            lower=lower,
            upper=upper,
        )
        joints.append(kj)
        by_child[c_link] = kj

    leaf_links = sorted(child_links - parent_links)
    chains: dict[str, list[KinematicJoint]] = {}
    for leaf in leaf_links:
        chain: list[KinematicJoint] = []
        cur = leaf
        seen: set[str] = set()
        while cur in by_child and cur not in seen:
            seen.add(cur)
            j = by_child[cur]
            chain.append(j)
            cur = j.parent_link
        chain.reverse()
        chains[leaf] = chain

    return joints, chains


def parse_mjcf_kinematic_chains(mjcf_path: Path) -> tuple[list[KinematicJoint], dict[str, list[KinematicJoint]]]:
    """Parse an MJCF XML body tree into kinematic chains so MuJoCo XMLs work via config too."""
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        return [], {}

    all_joints: list[KinematicJoint] = []
    chains: dict[str, list[KinematicJoint]] = {}

    def _walk(body_el: ET.Element, parent_name: str, acc_chain: list[KinematicJoint]) -> None:
        b_name = body_el.attrib.get("name", f"body_{len(all_joints)}")
        pos = _parse_floats(body_el.attrib.get("pos"), [0.0, 0.0, 0.0])
        j_els = body_el.findall("joint")
        local_chain = list(acc_chain)
        if j_els:
            for idx, j_el in enumerate(j_els):
                j_name = j_el.attrib.get("name", f"{b_name}_j{idx}")
                j_type = "prismatic" if j_el.attrib.get("type") == "slide" else "revolute"
                axis = _parse_floats(j_el.attrib.get("axis"), [0.0, 0.0, 1.0])
                rng = _parse_floats(j_el.attrib.get("range"), [-2.8, 2.8])
                kj = KinematicJoint(
                    name=j_name,
                    joint_type=j_type,
                    parent_link=parent_name,
                    child_link=b_name,
                    origin_xyz=pos if idx == 0 else np.zeros(3, dtype=np.float64),
                    origin_rpy=np.zeros(3, dtype=np.float64),
                    axis=axis / max(float(np.linalg.norm(axis)), 1e-6),
                    lower=float(rng[0]),
                    upper=float(rng[1]),
                )
                all_joints.append(kj)
                local_chain.append(kj)
        else:
            local_chain.append(
                KinematicJoint(
                    name=f"{b_name}_fixed",
                    joint_type="fixed",
                    parent_link=parent_name,
                    child_link=b_name,
                    origin_xyz=pos,
                    origin_rpy=np.zeros(3, dtype=np.float64),
                    axis=np.array([0.0, 0.0, 1.0]),
                    lower=0.0,
                    upper=0.0,
                )
            )
        children = body_el.findall("body")
        if not children:
            chains[b_name] = local_chain
        else:
            for ch in children:
                _walk(ch, b_name, local_chain)

    for top in worldbody.findall("body"):
        _walk(top, "world", [])
    return all_joints, chains


def load_kinematic_chains(model_path: Path) -> tuple[list[KinematicJoint], dict[str, list[KinematicJoint]]]:
    if model_path.suffix.lower() == ".xml":
        return parse_mjcf_kinematic_chains(model_path)
    return parse_urdf_kinematic_chains(model_path)


def _select_arm_chain(chains: dict[str, list[KinematicJoint]], ee_spec: str) -> list[KinematicJoint]:
    if not chains:
        return []
    if ee_spec in chains:
        return chains[ee_spec]

    def _actuated_count(ch: list[KinematicJoint]) -> int:
        return sum(1 for j in ch if j.joint_type in ("revolute", "continuous", "prismatic"))

    if ee_spec in ("auto_left", "auto_right"):
        token = "left" if ee_spec == "auto_left" else "right"
        short_tok = "l_" if token == "left" else "r_"
        matching = [
            ch
            for leaf, ch in chains.items()
            if token in leaf.lower()
            or short_tok in leaf.lower()
            or any(token in j.name.lower() or short_tok in j.name.lower() for j in ch)
        ]
        if matching:
            return max(matching, key=_actuated_count)

    # Default: pick the kinematic branch with the most actuated arm joints (skip finger-only sub-branches if arm exists)
    non_finger = [ch for leaf, ch in chains.items() if "finger" not in leaf.lower()]
    pool = non_finger if non_finger else list(chains.values())
    return max(pool, key=_actuated_count)


def _fk_chain_position(chain: list[KinematicJoint], q_actuated: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, str]]]:
    """Compute end-effector position in base frame plus per-actuated-joint (origin, world_axis, type) for analytic Jacobian."""
    rot = np.eye(3, dtype=np.float64)
    pos = np.zeros(3, dtype=np.float64)
    joint_frames: list[tuple[np.ndarray, np.ndarray, str]] = []
    q_idx = 0

    for j in chain:
        pos = pos + rot @ j.origin_xyz
        if np.any(j.origin_rpy != 0.0):
            rot = rot @ Rotation.from_euler("xyz", j.origin_rpy).as_matrix()

        if j.joint_type in ("revolute", "continuous", "prismatic"):
            axis_w = rot @ j.axis
            joint_frames.append((pos.copy(), axis_w.copy(), j.joint_type))
            q_val = float(q_actuated[q_idx])
            q_idx += 1
            if j.joint_type == "prismatic":
                pos = pos + axis_w * q_val
            else:
                rot = rot @ Rotation.from_rotvec(j.axis * q_val).as_matrix()

    return pos, joint_frames


def solve_arm_trajectory_ik(
    chain: list[KinematicJoint],
    targets_base_m: np.ndarray,
    dt_s: float,
    max_vel_radps: float,
    enforce_limits: bool = True,
    smooth_window: int = 7,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Warm-started damped-least-squares analytic-Jacobian IK across all frames 0..N-1."""
    actuated = [j for j in chain if j.joint_type in ("revolute", "continuous", "prismatic")]
    n_frames = len(targets_base_m)
    dof = len(actuated)
    if dof == 0 or n_frames == 0:
        return [], np.zeros((n_frames, 0), dtype=np.float32), np.zeros((n_frames, 3), dtype=np.float32), np.zeros(n_frames, dtype=np.float32)

    lows = np.array([j.lower for j in actuated], dtype=np.float64)
    highs = np.array([j.upper for j in actuated], dtype=np.float64)
    q_mid = 0.5 * (lows + highs)
    q_cur = q_mid.copy()
    q_traj = np.zeros((n_frames, dof), dtype=np.float64)

    # Probe nominal reach from neutral posture so target workspace stays physically reachable
    neutral_ee, _ = _fk_chain_position(chain, q_mid)
    max_step_rad = max_vel_radps * max(dt_s, 1e-3)
    damping_sq = 0.025**2

    for t in range(n_frames):
        target = targets_base_m[t]
        if not np.all(np.isfinite(target)):
            target = neutral_ee
        q = q_cur.copy()
        n_iters = 35 if t == 0 else 16
        for _ in range(n_iters):
            ee_pos, j_frames = _fk_chain_position(chain, q)
            err = target - ee_pos
            if float(np.linalg.norm(err)) < 0.002:
                break
            jac = np.zeros((3, dof), dtype=np.float64)
            for k, (j_pos, j_ax, j_type) in enumerate(j_frames):
                if j_type == "prismatic":
                    jac[:, k] = j_ax
                else:
                    jac[:, k] = np.cross(j_ax, ee_pos - j_pos)
            # Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 err + small posture bias toward q_mid
            jjt = jac @ jac.T + damping_sq * np.eye(3)
            dq = jac.T @ np.linalg.solve(jjt, err)
            dq = dq + 0.01 * (q_mid - q)
            dq_norm = float(np.linalg.norm(dq))
            if dq_norm > 0.35:
                dq *= 0.35 / dq_norm
            q = q + dq
            if enforce_limits:
                q = np.clip(q, lows, highs)

        # Enforce joint velocity continuity between consecutive frames
        if t > 0:
            delta = np.clip(q - q_cur, -max_step_rad, max_step_rad)
            q = q_cur + delta
            if enforce_limits:
                q = np.clip(q, lows, highs)

        q_cur = q
        q_traj[t] = q

    # Temporal smoothing with Savitzky-Golay filter (preserves trajectory peaks while suppressing jitter)
    if smooth_window >= 5 and n_frames >= smooth_window:
        win = smooth_window if smooth_window % 2 == 1 else smooth_window - 1
        for k in range(dof):
            q_traj[:, k] = savgol_filter(q_traj[:, k], window_length=win, polyorder=2)
        if enforce_limits:
            q_traj = np.clip(q_traj, lows[None, :], highs[None, :])

    achieved = np.zeros((n_frames, 3), dtype=np.float64)
    err_mm = np.zeros(n_frames, dtype=np.float64)
    for t in range(n_frames):
        ee_p, _ = _fk_chain_position(chain, q_traj[t])
        achieved[t] = ee_p
        err_mm[t] = float(np.linalg.norm(targets_base_m[t] - ee_p) * 1000.0)

    return [j.name for j in actuated], q_traj.astype(np.float32), achieved.astype(np.float32), err_mm.astype(np.float32)


def _build_arm_targets(
    frames: list[FrameAnnotation],
    side: str,
    workspace_center_m: list[float],
    workspace_scale: float,
    table_z_m: float,
    enforce_table_clearance: bool,
    table_z_margin_m: float,
) -> np.ndarray:
    """Map human world-frame wrist trajectory into the robot's reachable workspace with table clearance."""
    raw = []
    valid_mask = []
    for f in frames:
        h = f.left_hand if side == "left" else f.right_hand
        raw.append(h.wrist_pos_world_m if h.detected else [math.nan, math.nan, math.nan])
        valid_mask.append(h.detected)

    arr = np.asarray(raw, dtype=np.float64)
    center = np.asarray(workspace_center_m, dtype=np.float64)
    if not any(valid_mask):
        return np.tile(center[None, :], (len(frames), 1))

    # Linearly interpolate brief detection dropouts cleanly
    t_idx = np.arange(len(frames), dtype=np.float64)
    v_idx = np.where(np.asarray(valid_mask))[0]
    for c in range(3):
        arr[:, c] = np.interp(t_idx, v_idx, arr[v_idx, c])

    human_center = np.median(arr, axis=0)
    rel = (arr - human_center[None, :]) * float(workspace_scale)
    # Clamp radial excursion to 0.32m around robot workspace center so targets stay inside physical reach
    r_norm = np.linalg.norm(rel, axis=1, keepdims=True)
    rel = np.where(r_norm > 0.32, rel * (0.32 / np.maximum(r_norm, 1e-6)), rel)
    targets = center[None, :] + rel

    if enforce_table_clearance:
        min_z = float(table_z_m) + float(table_z_margin_m)
        targets[:, 2] = np.maximum(targets[:, 2], min_z)
    return targets


def _map_dexterous_hand_joints(
    frames: list[FrameAnnotation],
    sides: list[str],
    joints_per_hand: int,
    lower_rad: float,
    upper_rad: float,
) -> tuple[list[str], np.ndarray]:
    """Map 5D anatomical finger curls to arbitrary N-DoF dexterous hands (e.g. Allegro 16-DoF or G1 10-DoF)."""
    n = len(frames)
    names: list[str] = []
    cols: list[np.ndarray] = []
    span = upper_rad - lower_rad

    for side in sides:
        curls = np.array(
            [
                (f.left_hand if side == "left" else f.right_hand).finger_curls_5d
                if (f.left_hand if side == "left" else f.right_hand).detected
                else [0.0, 0.0, 0.0, 0.0, 0.0]
                for f in frames
            ],
            dtype=np.float64,
        )
        for j in range(joints_per_hand):
            finger_idx = j % 5
            joint_tier = j // 5
            # Distal joints bend proportionally with proximal curl
            scale = 1.0 if joint_tier == 0 else (0.85 if joint_tier == 1 else 0.70)
            q_col = np.clip(lower_rad + curls[:, finger_idx] * span * scale, lower_rad, upper_rad)
            names.append(f"{side}_finger_j{j}")
            cols.append(q_col.astype(np.float32))

    mat = np.column_stack(cols).astype(np.float64) if cols else np.zeros((n, 0), dtype=np.float64)
    if n >= 7 and mat.shape[1] > 0:
        for k in range(mat.shape[1]):
            mat[:, k] = np.clip(savgol_filter(mat[:, k], window_length=7, polyorder=2), lower_rad, upper_rad)
    return names, mat.astype(np.float32)


def retarget_embodiment_from_yaml(
    yaml_path: Path,
    frames: list[FrameAnnotation],
    segments: list[AtomicActionSegment],
    cfg: RetargetingConfig,
) -> EmbodimentTrajectory:
    """Retarget an episode purely from a declarative YAML + URDF/MJCF config (zero code edits per robot)."""
    spec: dict[str, Any] = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    name = str(spec.get("name", yaml_path.stem))
    emb_class = str(spec.get("embodiment_class", "custom"))
    n_frames = len(frames)
    dt_s = (
        (frames[-1].timestamp_s - frames[0].timestamp_s) / max(n_frames - 1, 1)
        if n_frames >= 2
        else (1.0 / 30.0)
    )

    urdf_raw = spec.get("urdf_path")
    chains: dict[str, list[KinematicJoint]] = {}
    if urdf_raw:
        urdf_p = Path(str(urdf_raw))
        if not urdf_p.exists():
            urdf_p = yaml_path.parent / urdf_p
        if not urdf_p.exists():
            raise FileNotFoundError(f"URDF/MJCF file not found for embodiment '{name}': {urdf_raw}")
        _, chains = load_kinematic_chains(urdf_p)

    all_joint_names: list[str] = []
    q_blocks: list[np.ndarray] = []
    arm_targets_by_side: dict[str, np.ndarray] = {}
    arm_achieved_list: list[np.ndarray] = []
    arm_target_list: list[np.ndarray] = []
    arm_err_list: list[np.ndarray] = []
    limit_ok_rates: list[float] = []
    table_ok_rates: list[float] = []

    # 1. Solve all configured arm chains
    arms_spec = spec.get("arms") or []
    for arm_cfg in arms_spec:
        side = str(arm_cfg["side"])
        targets = _build_arm_targets(
            frames=frames,
            side=side,
            workspace_center_m=list(arm_cfg.get("workspace_center_m", [0.40, 0.0, 0.30])),
            workspace_scale=float(arm_cfg.get("workspace_scale", 0.70)),
            table_z_m=float(arm_cfg.get("table_z_m", 0.0)),
            enforce_table_clearance=cfg.enforce_table_clearance,
            table_z_margin_m=cfg.table_z_margin_m,
        )
        arm_targets_by_side[side] = targets

    # Enforce bimanual minimum separation if both arms present
    bimanual_min_sep = float(spec.get("bimanual_min_sep_m", 0.08))
    bimanual_ok_pct = 100.0
    if "left" in arm_targets_by_side and "right" in arm_targets_by_side:
        lt = arm_targets_by_side["left"]
        rt = arm_targets_by_side["right"]
        diff = lt - rt
        dist = np.linalg.norm(diff, axis=1)
        too_close = dist < bimanual_min_sep
        if np.any(too_close):
            push = 0.5 * (bimanual_min_sep - dist[too_close])
            lt[too_close, 1] += push
            rt[too_close, 1] -= push
        bimanual_ok_pct = float(np.mean(np.linalg.norm(lt - rt, axis=1) >= (bimanual_min_sep - 1e-4)) * 100.0)

    for arm_cfg in arms_spec:
        side = str(arm_cfg["side"])
        ee_spec = str(arm_cfg.get("ee_link", "auto"))
        chain = _select_arm_chain(chains, ee_spec)
        targets = arm_targets_by_side[side]
        j_names, q_arm, ee_ach, err_mm = solve_arm_trajectory_ik(
            chain=chain,
            targets_base_m=targets,
            dt_s=dt_s,
            max_vel_radps=float(arm_cfg.get("max_joint_vel_radps", 2.4)),
            enforce_limits=cfg.enforce_joint_limits,
            smooth_window=cfg.smoothness_window,
        )
        prefixed = [f"{side}_{jn}" for jn in j_names]
        all_joint_names.extend(prefixed)
        q_blocks.append(q_arm)
        arm_target_list.append(targets.astype(np.float32))
        arm_achieved_list.append(ee_ach)
        arm_err_list.append(err_mm)

        actuated = [j for j in chain if j.joint_type in ("revolute", "continuous", "prismatic")]
        if actuated and q_arm.size:
            lows = np.array([j.lower - 1e-4 for j in actuated])
            highs = np.array([j.upper + 1e-4 for j in actuated])
            in_lim = (q_arm >= lows[None, :]) & (q_arm <= highs[None, :])
            limit_ok_rates.append(float(np.mean(in_lim) * 100.0))
        min_z = float(arm_cfg.get("table_z_m", -1.0)) - 0.02
        table_ok_rates.append(float(np.mean(ee_ach[:, 2] >= min_z) * 100.0))

    # 2. Gripper channels (for single/dual arm manipulators)
    if spec.get("gripper"):
        g_cfg = spec["gripper"]
        g_min = float(g_cfg.get("min_opening_m", 0.0))
        g_max = float(g_cfg.get("max_opening_m", 0.08))
        g_sides = [a["side"] for a in arms_spec] if arms_spec else ["right"]
        for side in g_sides:
            ap = np.array(
                [
                    np.clip((f.left_hand if side == "left" else f.right_hand).gripper_aperture_m, g_min, g_max)
                    for f in frames
                ],
                dtype=np.float32,
            )[:, None]
            all_joint_names.append(f"{side}_gripper_opening_m")
            q_blocks.append(ap)

    # 3. Dexterous hand channels (for humanoids and standalone dexterous hands)
    if spec.get("dexterous_hand"):
        dh = spec["dexterous_hand"]
        d_names, d_mat = _map_dexterous_hand_joints(
            frames=frames,
            sides=[str(s) for s in dh.get("sides", ["right"])],
            joints_per_hand=int(dh.get("joints_per_hand", 16)),
            lower_rad=float(dh.get("lower_rad", 0.0)),
            upper_rad=float(dh.get("upper_rad", 1.55)),
        )
        all_joint_names.extend(d_names)
        q_blocks.append(d_mat)
        limit_ok_rates.append(100.0)

    # 4. Glove / Teleop representation (world-frame SE(3) wrist + 5D curls + 6D tactile per hand)
    if spec.get("teleop_glove"):
        tg = spec["teleop_glove"]
        for side in [str(s) for s in tg.get("sides", ["left", "right"])]:
            rows = []
            for f in frames:
                h = f.left_hand if side == "left" else f.right_hand
                tac = f.tactile_left if side == "left" else f.tactile_right
                rows.append(
                    list(h.wrist_pos_world_m)
                    + list(h.wrist_rot6d_world)
                    + list(h.finger_curls_5d)
                    + list(tac.contact_zones_6d)
                )
            g_names = (
                [f"{side}_wrist_x", f"{side}_wrist_y", f"{side}_wrist_z"]
                + [f"{side}_rot6d_{k}" for k in range(6)]
                + [f"{side}_curl_{k}" for k in range(5)]
                + [f"{side}_tactile_z{k}" for k in range(6)]
            )
            all_joint_names.extend(g_names)
            g_mat = np.asarray(rows, dtype=np.float64)
            if n_frames >= 7 and g_mat.shape[1] > 0:
                for k in range(g_mat.shape[1]):
                    g_mat[:, k] = savgol_filter(g_mat[:, k], window_length=7, polyorder=2)
            q_blocks.append(g_mat.astype(np.float32))
            limit_ok_rates.append(100.0)

    qpos = np.hstack(q_blocks).astype(np.float32) if q_blocks else np.zeros((n_frames, 0), dtype=np.float32)
    actions = np.zeros_like(qpos, dtype=np.float32)
    if n_frames >= 2 and qpos.shape[1] > 0:
        actions[:-1] = qpos[1:] - qpos[:-1]
        actions[-1] = actions[-2]

    ee_target = np.mean(arm_target_list, axis=0) if arm_target_list else np.zeros((n_frames, 3), dtype=np.float32)
    ee_achieved = np.mean(arm_achieved_list, axis=0) if arm_achieved_list else np.zeros((n_frames, 3), dtype=np.float32)
    fk_err_mm = np.mean(arm_err_list, axis=0) if arm_err_list else np.zeros(n_frames, dtype=np.float32)

    mean_fk_mm = float(np.mean(fk_err_mm)) if fk_err_mm.size else 0.0
    p95_fk_mm = float(np.percentile(fk_err_mm, 95)) if fk_err_mm.size else 0.0
    limit_pct = float(np.mean(limit_ok_rates)) if limit_ok_rates else 100.0
    table_pct = float(np.mean(table_ok_rates)) if table_ok_rates else 100.0

    # Smoothness score based on mean absolute joint acceleration
    if n_frames >= 3 and qpos.shape[1] > 0:
        acc = np.diff(qpos.astype(np.float64), n=2, axis=0) / max(dt_s**2, 1e-4)
        med_acc = float(np.median(np.abs(acc)))
        smooth_score = float(np.clip(1.0 - med_acc / 25.0, 0.2, 1.0))
    else:
        smooth_score = 1.0

    ik_score = float(np.clip(1.0 - mean_fk_mm / 150.0, 0.0, 1.0)) if arm_err_list else 1.0
    overall_feasibility = round(
        0.40 * ik_score + 0.25 * (limit_pct / 100.0) + 0.20 * smooth_score + 0.15 * (table_pct / 100.0),
        4,
    )

    seg_feas: list[dict[str, Any]] = []
    for seg in segments:
        s0 = max(0, seg.start_frame)
        s1 = min(n_frames - 1, seg.end_frame)
        sub_err = float(np.mean(fk_err_mm[s0 : s1 + 1])) if (arm_err_list and s1 >= s0) else 0.0
        sub_ik = float(np.clip(1.0 - sub_err / 150.0, 0.0, 1.0)) if arm_err_list else 1.0
        seg_feas.append(
            {
                "segment_id": seg.segment_id,
                "fk_pos_error_mm": round(sub_err, 2),
                "feasibility_score": round(0.6 * sub_ik + 0.4 * smooth_score, 4),
            }
        )

    return EmbodimentTrajectory(
        name=name,
        embodiment_class=emb_class,
        joint_names=all_joint_names,
        qpos=qpos,
        actions=actions,
        ee_target_m=ee_target,
        ee_achieved_m=ee_achieved,
        fk_pos_error_mm=fk_err_mm,
        metrics={
            "embodiment_class": emb_class,
            "dof": int(qpos.shape[1]),
            "frames": n_frames,
            "mean_fk_pos_error_mm": round(mean_fk_mm, 2),
            "p95_fk_pos_error_mm": round(p95_fk_mm, 2),
            "joint_limit_compliance_pct": round(limit_pct, 2),
            "table_clearance_pct": round(table_pct, 2),
            "bimanual_clearance_pct": round(bimanual_ok_pct, 2),
            "smoothness_score": round(smooth_score, 4),
            "overall_feasibility": overall_feasibility,
        },
        segment_feasibility=seg_feas,
    )


def retarget_all_configured_embodiments(
    frames: list[FrameAnnotation],
    segments: list[AtomicActionSegment],
    cfg: RetargetingConfig,
) -> dict[str, EmbodimentTrajectory]:
    """Run pluggable config-driven retargeting for every embodiment enabled in PipelineConfig."""
    results: dict[str, EmbodimentTrajectory] = {}
    if not cfg.enabled:
        return results
    cfg_dir = Path(cfg.embodiment_config_dir)
    for emb_name in cfg.embodiments:
        yp = cfg_dir / f"{emb_name}.yaml"
        if not yp.exists():
            raise FileNotFoundError(f"Embodiment config not found: {yp}")
        results[emb_name] = retarget_embodiment_from_yaml(yp, frames, segments, cfg)
    return results
