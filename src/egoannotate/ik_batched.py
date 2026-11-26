"""Vectorized GPU/CPU batched Levenberg-Marquardt IK with reach clamping and PyRoki hook."""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from scipy.signal import savgol_filter


def has_pyroki_backend() -> bool:
    """Return True if optional JAX-based PyRoki solver is installed."""
    return (
        importlib.util.find_spec("pyroki") is not None
        and importlib.util.find_spec("jax") is not None
    )


def _rpy_to_rot_np(rpy: Sequence[float]) -> np.ndarray:
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (rz @ ry @ rx).astype(np.float32)


def _rodrigues_batch(axes: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Compute (T, 3, 3) rotation matrices for unit axis (3,) and angles (T,)."""
    t_len = angles.shape[0]
    ax = axes / (torch.linalg.norm(axes) + 1e-8)
    kx, ky, kz = ax[0], ax[1], ax[2]
    zero = torch.zeros((), device=angles.device, dtype=angles.dtype)
    k_mat = torch.stack(
        [
            torch.stack([zero, -kz, ky]),
            torch.stack([kz, zero, -kx]),
            torch.stack([-ky, kx, zero]),
        ]
    )
    k2 = k_mat @ k_mat
    eye = torch.eye(3, device=angles.device, dtype=angles.dtype).unsqueeze(0).expand(t_len, 3, 3)
    s = torch.sin(angles).view(t_len, 1, 1)
    c = (1.0 - torch.cos(angles)).view(t_len, 1, 1)
    return eye + s * k_mat.unsqueeze(0) + c * k2.unsqueeze(0)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device)
    return torch.device("cpu")


def _extract_chain_arrays(
    chain: Sequence[Any],
) -> tuple[
    list[str],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[int],
    np.ndarray,
    np.ndarray,
]:
    j_types: list[str] = []
    origins: list[np.ndarray] = []
    rot_origins: list[np.ndarray] = []
    axes: list[np.ndarray] = []
    actuated_idx: list[int] = []
    lowers: list[float] = []
    uppers: list[float] = []

    for idx, j in enumerate(chain):
        jtype = str(getattr(j, "joint_type", "fixed")).lower()
        j_types.append(jtype)
        origins.append(np.asarray(getattr(j, "origin_xyz", (0.0, 0.0, 0.0)), dtype=np.float32))
        rot_origins.append(_rpy_to_rot_np(getattr(j, "origin_rpy", (0.0, 0.0, 0.0))))
        ax = np.asarray(getattr(j, "axis", (0.0, 0.0, 1.0)), dtype=np.float32)
        norm = float(np.linalg.norm(ax))
        axes.append((ax / norm) if norm > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=np.float32))
        if jtype in ("revolute", "continuous", "prismatic"):
            actuated_idx.append(idx)
            lo = float(getattr(j, "lower", -math.pi))
            hi = float(getattr(j, "upper", math.pi))
            if jtype == "continuous":
                lo, hi = -2.0 * math.pi, 2.0 * math.pi
            if hi <= lo:
                lo, hi = -math.pi, math.pi
            lowers.append(lo)
            uppers.append(hi)

    return (
        j_types,
        origins,
        rot_origins,
        axes,
        actuated_idx,
        np.asarray(lowers, dtype=np.float32),
        np.asarray(uppers, dtype=np.float32),
    )


def compute_chain_reach_bounds(chain: Sequence[Any]) -> tuple[np.ndarray, float, float]:
    """Compute shoulder/base anchor and feasible [min_reach, max_reach] envelope."""
    _, origins, _, _, actuated_idx, _, _ = _extract_chain_arrays(chain)
    if not origins:
        return np.zeros(3, dtype=np.float32), 0.05, 0.85
    first_act = actuated_idx[0] if actuated_idx else 0
    anchor = np.zeros(3, dtype=np.float32)
    for k in range(first_act + 1):
        anchor = anchor + origins[k]
    link_lengths = [float(np.linalg.norm(origins[k])) for k in range(first_act + 1, len(origins))]
    max_reach = float(sum(link_lengths))
    if max_reach < 1e-3:
        max_reach = float(sum(float(np.linalg.norm(o)) for o in origins))
    max_reach = max(max_reach, 0.15)
    min_reach = min(0.08, 0.15 * max_reach)
    return anchor.astype(np.float32), min_reach, max_reach


def clamp_targets_to_workspace(
    targets_base_m: np.ndarray,
    chain: Sequence[Any],
    reach_margin: float = 0.96,
) -> np.ndarray:
    """Radially project unreachable targets onto the robot arm's feasible reach shell."""
    pts = np.asarray(targets_base_m, dtype=np.float32).copy()
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        return pts
    anchor, min_r, max_r = compute_chain_reach_bounds(chain)
    safe_max = max(min_r + 0.02, float(max_r) * float(reach_margin))
    diff = pts - anchor[None, :]
    dists = np.linalg.norm(diff, axis=1, keepdims=True)
    too_far = dists[:, 0] > safe_max
    if np.any(too_far):
        scale = safe_max / np.maximum(dists[too_far], 1e-6)
        pts[too_far] = anchor[None, :] + diff[too_far] * scale
    too_close = (dists[:, 0] < min_r) & (dists[:, 0] > 1e-6)
    if np.any(too_close):
        scale = min_r / np.maximum(dists[too_close], 1e-6)
        pts[too_close] = anchor[None, :] + diff[too_close] * scale
    return pts


def _fk_and_jacobian_torch(
    j_types: list[str],
    origins_t: list[torch.Tensor],
    rot_origins_t: list[torch.Tensor],
    axes_t: list[torch.Tensor],
    actuated_idx: list[int],
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched forward kinematics and analytical geometric Jacobian for all T frames."""
    t_len = q.shape[0]
    dof = q.shape[1]
    dev = q.device
    dtype = q.dtype

    pos = torch.zeros((t_len, 3), device=dev, dtype=dtype)
    rot = torch.eye(3, device=dev, dtype=dtype).unsqueeze(0).expand(t_len, 3, 3).clone()

    act_map = {joint_i: dof_i for dof_i, joint_i in enumerate(actuated_idx)}
    joint_pos_list: list[torch.Tensor] = [torch.zeros((t_len, 3), device=dev, dtype=dtype)] * dof
    joint_axis_list: list[torch.Tensor] = [torch.zeros((t_len, 3), device=dev, dtype=dtype)] * dof
    joint_prismatic: list[bool] = [False] * dof

    for idx, jtype in enumerate(j_types):
        p_off = torch.einsum("tij,j->ti", rot, origins_t[idx])
        pos = pos + p_off
        rot = torch.matmul(rot, rot_origins_t[idx].unsqueeze(0))

        if idx in act_map:
            d_i = act_map[idx]
            world_axis = torch.einsum("tij,j->ti", rot, axes_t[idx])
            joint_pos_list[d_i] = pos.clone()
            joint_axis_list[d_i] = world_axis
            if jtype == "prismatic":
                joint_prismatic[d_i] = True
                pos = pos + world_axis * q[:, d_i : d_i + 1]
            else:
                r_j = _rodrigues_batch(axes_t[idx], q[:, d_i])
                rot = torch.matmul(rot, r_j)

    jac = torch.zeros((t_len, 3, dof), device=dev, dtype=dtype)
    for d_i in range(dof):
        if joint_prismatic[d_i]:
            jac[:, :, d_i] = joint_axis_list[d_i]
        else:
            jac[:, :, d_i] = torch.cross(joint_axis_list[d_i], pos - joint_pos_list[d_i], dim=1)
    return pos, jac


def batched_fk_positions(
    chain: Sequence[Any],
    q_batch: np.ndarray,
    device: str = "auto",
) -> np.ndarray:
    """Evaluate batched end-effector XYZ positions for a trajectory (T, D)."""
    j_types, origins, rot_origins, axes, actuated_idx, _, _ = _extract_chain_arrays(chain)
    dev = _resolve_device(device)
    q_t = torch.as_tensor(np.asarray(q_batch, dtype=np.float32), device=dev)
    if q_t.ndim == 1:
        q_t = q_t.unsqueeze(0)
    origins_t = [torch.as_tensor(o, device=dev) for o in origins]
    rot_origins_t = [torch.as_tensor(r, device=dev) for r in rot_origins]
    axes_t = [torch.as_tensor(a, device=dev) for a in axes]
    with torch.inference_mode():
        pos, _ = _fk_and_jacobian_torch(j_types, origins_t, rot_origins_t, axes_t, actuated_idx, q_t)
    return pos.detach().cpu().numpy().astype(np.float32)


def _smooth_and_rate_limit(
    q_np: np.ndarray,
    dt_s: float,
    max_vel_radps: float,
    lowers: np.ndarray,
    uppers: np.ndarray,
    enforce_limits: bool,
    smooth_window: int,
) -> np.ndarray:
    out = np.asarray(q_np, dtype=np.float32).copy()
    t_len = out.shape[0]
    if t_len >= 5 and smooth_window >= 3:
        win = min(int(smooth_window), t_len)
        if win % 2 == 0:
            win -= 1
        if win >= 3:
            poly = min(2, win - 1)
            out = savgol_filter(out, window_length=win, polyorder=poly, axis=0).astype(np.float32)
    max_step = max(1e-4, float(max_vel_radps) * max(1e-4, float(dt_s)))
    for t in range(1, t_len):
        dq = np.clip(out[t] - out[t - 1], -max_step, max_step)
        out[t] = out[t - 1] + dq
    if enforce_limits and lowers.size == out.shape[1]:
        out = np.clip(out, lowers[None, :], uppers[None, :]).astype(np.float32)
    return out


def solve_arm_trajectory_ik_batched(
    chain: Sequence[Any],
    targets_base_m: np.ndarray,
    dt_s: float = 1.0 / 30.0,
    max_vel_radps: float = 2.8,
    enforce_limits: bool = True,
    smooth_window: int = 7,
    clamp_reach: bool = True,
    max_iters: int = 45,
    damping: float = 0.035,
    device: str = "auto",
    backend: str = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve full-trajectory IK in parallel on GPU/CPU.

    Returns (qpos [T, D], ee_achieved_m [T, 3], fk_error_mm [T], targets_used_m [T, 3]).
    """
    _ = backend
    targets_raw = np.asarray(targets_base_m, dtype=np.float32)
    t_len = int(targets_raw.shape[0])
    j_types, origins, rot_origins, axes, actuated_idx, lowers, uppers = _extract_chain_arrays(chain)
    dof = len(actuated_idx)
    if t_len == 0 or dof == 0:
        empty_q = np.zeros((t_len, dof), dtype=np.float32)
        empty_ee = np.zeros((t_len, 3), dtype=np.float32)
        empty_err = np.zeros((t_len,), dtype=np.float32)
        return empty_q, empty_ee, empty_err, targets_raw

    targets_used = clamp_targets_to_workspace(targets_raw, chain) if clamp_reach else targets_raw
    dev = _resolve_device(device)
    origins_t = [torch.as_tensor(o, device=dev) for o in origins]
    rot_origins_t = [torch.as_tensor(r, device=dev) for r in rot_origins]
    axes_t = [torch.as_tensor(a, device=dev) for a in axes]
    lo_t = torch.as_tensor(lowers, device=dev)
    hi_t = torch.as_tensor(uppers, device=dev)
    q_mid = 0.5 * (lo_t + hi_t)
    q_span = torch.clamp(hi_t - lo_t, min=0.2)

    q = q_mid.unsqueeze(0).expand(t_len, dof).clone()
    q = q + 0.08 * q_span.unsqueeze(0) * torch.sin(
        torch.arange(dof, device=dev, dtype=torch.float32) + 0.7
    ).unsqueeze(0)
    if enforce_limits:
        q = torch.clamp(q, lo_t, hi_t)

    tgt_t = torch.as_tensor(targets_used, device=dev, dtype=torch.float32)
    eye3 = torch.eye(3, device=dev, dtype=torch.float32).unsqueeze(0)

    with torch.inference_mode():
        lam = float(damping)
        for it in range(max_iters):
            pos, jac = _fk_and_jacobian_torch(
                j_types, origins_t, rot_origins_t, axes_t, actuated_idx, q
            )
            err = tgt_t - pos
            max_err = float(torch.linalg.norm(err, dim=1).max().item())
            if max_err < 1e-4:
                break

            jjt = torch.bmm(jac, jac.transpose(1, 2)) + (lam * lam) * eye3
            sol = torch.linalg.solve(jjt, err.unsqueeze(-1))
            dq = torch.bmm(jac.transpose(1, 2), sol).squeeze(-1)

            null_weight = 0.02 if it < max_iters - 10 else 0.005
            dq = dq + null_weight * (q_mid.unsqueeze(0) - q)
            dq = torch.clamp(dq, -0.35, 0.35)
            q = q + dq
            if enforce_limits:
                q = torch.clamp(q, lo_t, hi_t)
            if it > 15:
                lam = max(0.01, lam * 0.96)

        q_np = _smooth_and_rate_limit(
            q.detach().cpu().numpy(),
            dt_s=dt_s,
            max_vel_radps=max_vel_radps,
            lowers=lowers,
            uppers=uppers,
            enforce_limits=enforce_limits,
            smooth_window=smooth_window,
        )

        q_final = torch.as_tensor(q_np, device=dev, dtype=torch.float32)
        for _ in range(6):
            pos, jac = _fk_and_jacobian_torch(
                j_types, origins_t, rot_origins_t, axes_t, actuated_idx, q_final
            )
            err = tgt_t - pos
            jjt = torch.bmm(jac, jac.transpose(1, 2)) + (0.015 * 0.015) * eye3
            dq = torch.bmm(jac.transpose(1, 2), torch.linalg.solve(jjt, err.unsqueeze(-1))).squeeze(
                -1
            )
            q_final = q_final + torch.clamp(dq, -0.06, 0.06)
            if enforce_limits:
                q_final = torch.clamp(q_final, lo_t, hi_t)

        ee_final, _ = _fk_and_jacobian_torch(
            j_types, origins_t, rot_origins_t, axes_t, actuated_idx, q_final
        )

    q_out = q_final.detach().cpu().numpy().astype(np.float32)
    ee_out = ee_final.detach().cpu().numpy().astype(np.float32)
    err_mm = (np.linalg.norm(targets_used - ee_out, axis=1) * 1000.0).astype(np.float32)
    return q_out, ee_out, err_mm, targets_used
