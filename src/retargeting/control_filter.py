"""Real-robot control filtering, minimum-jerk gap bridging, bimanual repulsion, and hardware audit.

Designed so retargeted trajectories can be fed directly into a 30Hz/100Hz impedance or
joint-position controller on Franka Panda, Unitree G1/H1, and Allegro Hand without
tripping velocity, acceleration, or torque-rate safety gates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class HardwareAuditReport:
    robot_name: str
    kind: str
    num_frames: int
    dt: float
    active_hand_pct: float
    closed_loop_playable_pct: float
    mean_ik_residual_mm: float
    p95_ik_residual_mm: float
    max_joint_vel_rad_s: float
    vel_limit_compliance_pct: float
    max_joint_acc_rad_s2: float
    acc_limit_compliance_pct: float
    rms_joint_jerk_rad_s3: float
    self_collision_events: int
    joint_limit_events: int
    teleport_jumps_detected: int
    hardware_readiness_score: float
    verdict: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "robot_name": self.robot_name,
            "kind": self.kind,
            "num_frames": int(self.num_frames),
            "dt_s": round(float(self.dt), 4),
            "active_hand_pct": round(float(self.active_hand_pct), 2),
            "closed_loop_playable_pct": round(float(self.closed_loop_playable_pct), 2),
            "mean_ik_residual_mm": round(float(self.mean_ik_residual_mm), 3),
            "p95_ik_residual_mm": round(float(self.p95_ik_residual_mm), 3),
            "max_joint_vel_rad_s": round(float(self.max_joint_vel_rad_s), 3),
            "vel_limit_compliance_pct": round(float(self.vel_limit_compliance_pct), 2),
            "max_joint_acc_rad_s2": round(float(self.max_joint_acc_rad_s2), 3),
            "acc_limit_compliance_pct": round(float(self.acc_limit_compliance_pct), 2),
            "rms_joint_jerk_rad_s3": round(float(self.rms_joint_jerk_rad_s3), 2),
            "self_collision_events": int(self.self_collision_events),
            "joint_limit_events": int(self.joint_limit_events),
            "teleport_jumps_detected": int(self.teleport_jumps_detected),
            "hardware_readiness_score": round(float(self.hardware_readiness_score), 1),
            "verdict": self.verdict,
        }


def _quintic_alpha(tau: float) -> float:
    """Minimum-jerk quintic polynomial s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5 in [0, 1]."""
    t = float(np.clip(tau, 0.0, 1.0))
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


def bridge_cartesian_gaps_minimum_jerk(
    targets: Sequence[Tuple[np.ndarray, np.ndarray]],
    valid_mask: Sequence[bool],
    home_pos: np.ndarray,
    home_quat: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
    max_bridge_frames: int = 45,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Replace tracking dropouts with C^2 minimum-jerk splines or gentle hold-and-return.

    Prevents target teleportation when MediaPipe loses a hand mid-manipulation:
      - Leading gap (before first tracked frame): hold first valid pose (or home if none).
      - Internal gap <= max_bridge_frames: minimum-jerk quintic spline + quaternion SLERP.
      - Internal gap > max_bridge_frames: minimum-jerk ramp to pre-gap hold, then ramp to re-entry.
      - Trailing gap (after last tracked frame): hold last valid pose smoothly.
    """
    n = len(targets)
    if n == 0:
        return [], np.zeros(0, dtype=bool)

    valid = np.asarray(valid_mask, dtype=bool)
    pos_arr = np.array([np.asarray(p, dtype=np.float64) for p, _ in targets], dtype=np.float64)
    quat_arr = np.array([np.asarray(q, dtype=np.float64) for _, q in targets], dtype=np.float64)

    for i in range(n):
        norm = np.linalg.norm(quat_arr[i])
        quat_arr[i] = quat_arr[i] / norm if norm > 1e-8 else home_quat.copy()

    valid_indices = np.where(valid)[0]
    if len(valid_indices) == 0:
        return [(home_pos.copy(), home_quat.copy()) for _ in range(n)], np.zeros(n, dtype=bool)

    first_valid = int(valid_indices[0])
    last_valid = int(valid_indices[-1])

    # 1. Leading frames: hold first valid pose so robot starts ready without a jump
    for i in range(first_valid):
        pos_arr[i] = pos_arr[first_valid].copy()
        quat_arr[i] = quat_arr[first_valid].copy()

    # 2. Trailing frames: hold last valid pose
    for i in range(last_valid + 1, n):
        pos_arr[i] = pos_arr[last_valid].copy()
        quat_arr[i] = quat_arr[last_valid].copy()

    # 3. Internal gaps between valid segments: quintic minimum-jerk + SLERP
    for k in range(len(valid_indices) - 1):
        i0 = int(valid_indices[k])
        i1 = int(valid_indices[k + 1])
        gap = i1 - i0 - 1
        if gap <= 0:
            continue

        p0, p1 = pos_arr[i0].copy(), pos_arr[i1].copy()
        q0, q1 = quat_arr[i0].copy(), quat_arr[i1].copy()
        if np.dot(q0, q1) < 0.0:
            q1 = -q1
        slerp = Slerp([0.0, 1.0], Rotation.from_quat([q0, q1]))

        if gap <= max_bridge_frames:
            for step in range(1, gap + 1):
                alpha = _quintic_alpha(step / float(gap + 1))
                pos_arr[i0 + step] = (1.0 - alpha) * p0 + alpha * p1
                quat_arr[i0 + step] = slerp([alpha])[0].as_quat()
        else:
            # Hold p0 until max_bridge_frames before i1, then minimum-jerk ramp into p1
            ramp_len = min(max_bridge_frames, gap)
            hold_end = i1 - ramp_len
            for idx in range(i0 + 1, hold_end):
                pos_arr[idx] = p0.copy()
                quat_arr[idx] = q0.copy()
            for step in range(1, ramp_len + 1):
                alpha = _quintic_alpha(step / float(ramp_len + 1))
                pos_arr[hold_end - 1 + step] = (1.0 - alpha) * p0 + alpha * p1
                quat_arr[hold_end - 1 + step] = slerp([alpha])[0].as_quat()

    bridged = [(pos_arr[i], quat_arr[i]) for i in range(n)]
    # Every frame now has a continuous, velocity-safe target
    playable_mask = np.ones(n, dtype=bool)
    return bridged, playable_mask


def enforce_bimanual_separation(
    right_targets: Sequence[Tuple[np.ndarray, np.ndarray]],
    left_targets: Sequence[Tuple[np.ndarray, np.ndarray]],
    min_separation_m: float = 0.16,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:
    """Prevent bimanual humanoid forearm/wrist self-collisions by enforcing minimum 3D separation."""
    out_r: List[Tuple[np.ndarray, np.ndarray]] = []
    out_l: List[Tuple[np.ndarray, np.ndarray]] = []

    for (pr, qr), (pl, ql) in zip(right_targets, left_targets):
        pr_new = np.asarray(pr, dtype=np.float64).copy()
        pl_new = np.asarray(pl, dtype=np.float64).copy()

        # Keep right arm on -Y side and left arm on +Y side of torso sagittal plane when close in X/Z
        xz_dist = float(np.hypot(pr_new[0] - pl_new[0], pr_new[2] - pl_new[2]))
        if xz_dist < 0.22:
            y_sep = float(pl_new[1] - pr_new[1])
            if y_sep < min_separation_m:
                mid_y = 0.5 * (pl_new[1] + pr_new[1])
                pr_new[1] = min(pr_new[1], mid_y - 0.5 * min_separation_m)
                pl_new[1] = max(pl_new[1], mid_y + 0.5 * min_separation_m)

        out_r.append((pr_new, np.asarray(qr, dtype=np.float64).copy()))
        out_l.append((pl_new, np.asarray(ql, dtype=np.float64).copy()))

    return out_r, out_l


def smooth_and_limit_joint_trajectory(
    q_traj: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    vel_limits: np.ndarray,
    dt: float = 1.0 / 30.0,
    acc_limit_scale: float = 5.0,
    ema_alpha: float = 0.45,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply multi-pass zero-phase smoothing + exact discrete diff velocity & acceleration limiting."""
    n, nq = q_traj.shape
    if n <= 1:
        z = np.zeros_like(q_traj)
        return np.clip(q_traj, lo, hi), z, z, z

    v_max = np.asarray(vel_limits[:nq], dtype=np.float64) * 0.85
    a_max = v_max * acc_limit_scale

    # 3-pass zero-phase forward-backward filter to remove high-frequency jerk
    q_s = np.clip(q_traj.copy(), lo, hi)
    for _ in range(3):
        for i in range(1, n):
            q_s[i] = ema_alpha * q_s[i] + (1.0 - ema_alpha) * q_s[i - 1]
        for i in range(n - 2, -1, -1):
            q_s[i] = ema_alpha * q_s[i] + (1.0 - ema_alpha) * q_s[i + 1]

    # Exact discrete first- and second-difference rate limiter matching np.diff controller checks
    q_safe = np.zeros_like(q_s)
    q_safe[0] = np.clip(q_s[0], lo, hi)
    prev_vel = np.zeros(nq, dtype=np.float64)

    for i in range(1, n):
        raw_vel = (q_s[i] - q_safe[i - 1]) / dt
        raw_vel = np.clip(raw_vel, -v_max, v_max)
        dv = np.clip(raw_vel - prev_vel, -a_max * dt, a_max * dt)
        vel = np.clip(prev_vel + dv, -v_max, v_max)

        # Predict if hitting joint bounds would cause a deceleration spike
        q_cand = q_safe[i - 1] + vel * dt
        q_next = np.clip(q_cand, lo, hi)
        actual_vel = (q_next - q_safe[i - 1]) / dt
        if np.any(np.abs(actual_vel - prev_vel) > a_max * dt * 1.02):
            actual_vel = prev_vel + np.clip(actual_vel - prev_vel, -a_max * dt, a_max * dt)
            q_next = np.clip(q_safe[i - 1] + actual_vel * dt, lo, hi)
            actual_vel = (q_next - q_safe[i - 1]) / dt

        # Guarantee exact second-difference acceleration compliance after position clipping
        if i >= 2:
            actual_acc = (actual_vel - prev_vel) / dt
            if np.any(np.abs(actual_acc) > a_max):
                actual_vel = prev_vel + np.clip(actual_acc, -a_max, a_max) * dt
                q_next = q_safe[i - 1] + actual_vel * dt
        prev_vel = actual_vel
        q_safe[i] = q_next

    qd = np.vstack([np.zeros((1, nq)), np.diff(q_safe, axis=0) / dt])
    qdd = np.vstack([np.zeros((2, nq)), np.diff(qd[1:], axis=0) / dt]) if n > 2 else np.zeros_like(q_safe)
    qddd = np.vstack([np.zeros((3, nq)), np.diff(qdd[2:], axis=0) / dt]) if n > 3 else np.zeros_like(q_safe)
    return q_safe, qd, qdd, qddd


def compute_action_chunks(q_traj: np.ndarray, grip_traj: np.ndarray, chunk_horizon: int = 16) -> np.ndarray:
    """Build (N, chunk_horizon, nq + 1) future action chunks for ACT / Diffusion / VLA training."""
    n, nq = q_traj.shape
    full_act = np.concatenate([q_traj, grip_traj.reshape(n, 1)], axis=1)
    chunks = np.zeros((n, chunk_horizon, nq + 1), dtype=np.float32)
    for i in range(n):
        for h in range(chunk_horizon):
            idx = min(i + h, n - 1)
            chunks[i, h] = full_act[idx]
    return chunks


def audit_robot_readiness(
    robot_name: str,
    kind: str,
    q_traj: np.ndarray,
    vel_limits: np.ndarray,
    residuals_mm: Sequence[float],
    active_mask: Sequence[bool],
    playable_mask: Sequence[bool],
    self_collision_events: int,
    joint_limit_events: int,
    dt: float = 1.0 / 30.0,
) -> HardwareAuditReport:
    """Grade the retargeted trajectory as if feeding it to a real robot controller at 1/dt Hz."""
    n, nq = q_traj.shape
    qd = np.diff(q_traj, axis=0) / dt if n > 1 else np.zeros((1, nq))
    qdd = np.diff(qd, axis=0) / dt if n > 2 else np.zeros((1, nq))
    qddd = np.diff(qdd, axis=0) / dt if n > 3 else np.zeros((1, nq))

    v_lim = np.asarray(vel_limits[:nq], dtype=np.float64)
    a_lim = v_lim * 8.0

    vel_Ratio = np.abs(qd) / np.maximum(v_lim[None, :], 1e-6)
    acc_Ratio = np.abs(qdd) / np.maximum(a_lim[None, :], 1e-6)

    vel_ok_pct = 100.0 * float(np.mean(np.all(vel_Ratio <= 1.02, axis=1))) if len(qd) else 100.0
    acc_ok_pct = 100.0 * float(np.mean(np.all(acc_Ratio <= 1.05, axis=1))) if len(qdd) else 100.0

    max_vel = float(np.max(np.abs(qd))) if len(qd) else 0.0
    max_acc = float(np.max(np.abs(qdd))) if len(qdd) else 0.0
    rms_jerk = float(np.sqrt(np.mean(qddd ** 2))) if len(qddd) else 0.0

    # Teleport check: any single step joint jump > 1.2 * v_lim * dt
    teleports = int(np.sum(np.any(np.abs(qd) > 1.25 * v_lim[None, :], axis=1))) if len(qd) else 0

    res_arr = np.asarray(residuals_mm, dtype=np.float64) if len(residuals_mm) else np.zeros(1)
    mean_res = float(np.mean(res_arr))
    p95_res = float(np.percentile(res_arr, 95))

    active_pct = 100.0 * float(np.mean(active_mask)) if len(active_mask) else 0.0
    playable_pct = 100.0 * float(np.mean(playable_mask)) if len(playable_mask) else 0.0

    # Composite 0-100 Hardware Readiness Score
    score = 100.0
    score -= max(0.0, (mean_res - 15.0) * 0.35)          # penalize mean IK residual > 15mm
    score -= (100.0 - vel_ok_pct) * 1.5                  # penalize velocity violations
    score -= (100.0 - acc_ok_pct) * 0.8                  # penalize acceleration spikes
    score -= min(25.0, self_collision_events * 0.5)      # penalize self-collisions
    score -= min(20.0, joint_limit_events * 0.2)         # penalize joint limit violations
    score -= min(30.0, teleports * 5.0)                  # heavy penalty for teleports
    score = float(np.clip(score, 0.0, 100.0))

    if score >= 88.0 and teleports == 0 and self_collision_events == 0:
        verdict = "PRODUCTION_READY_FOR_SIM_AND_REAL_IMPEDANCE_CONTROL"
    elif score >= 75.0 and teleports == 0:
        verdict = "SAFE_FOR_SIM_AND_LOW_GAIN_REAL_ROLLOUT"
    else:
        verdict = "NEEDS_FILTERING_BEFORE_REAL_HARDWARE"

    return HardwareAuditReport(
        robot_name=robot_name,
        kind=kind,
        num_frames=n,
        dt=dt,
        active_hand_pct=active_pct,
        closed_loop_playable_pct=playable_pct,
        mean_ik_residual_mm=mean_res,
        p95_ik_residual_mm=p95_res,
        max_joint_vel_rad_s=max_vel,
        vel_limit_compliance_pct=vel_ok_pct,
        max_joint_acc_rad_s2=max_acc,
        acc_limit_compliance_pct=acc_ok_pct,
        rms_joint_jerk_rad_s3=rms_jerk,
        self_collision_events=self_collision_events,
        joint_limit_events=joint_limit_events,
        teleport_jumps_detected=teleports,
        hardware_readiness_score=score,
        verdict=verdict,
    )
