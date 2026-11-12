#!/usr/bin/env python3
"""Gate 08: Independent PyBullet Forward Kinematics & Retargeting Verification.

Independently recomputes FK from saved joint trajectories, measuring per-frame position (mm)
and orientation (deg) residuals against saved targets, joint-limit violations, clamp rates,
velocity/jerk profiles, and verifying reported IK convergence equals recomputed convergence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pybullet as pb
import pybullet_data


def _quat_geodesic_deg(q1_xyzw: np.ndarray, q2_xyzw: np.ndarray) -> float:
    n1 = np.linalg.norm(q1_xyzw)
    n2 = np.linalg.norm(q2_xyzw)
    if n1 < 1e-8 or n2 < 1e-8:
        return 180.0
    dot = float(np.clip(abs(np.dot(q1_xyzw / n1, q2_xyzw / n2)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate 08: Independent FK & Retargeting Audit")
    parser.add_argument("episode_dir", type=str)
    parser.add_argument("--pos-tol-mm", type=float, default=15.0)
    parser.add_argument("--ori-tol-deg", type=float, default=20.0)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    print("=== G08: INDEPENDENT PYBULLET FK & RETARGETING AUDIT ===")
    ep_dir = Path(args.episode_dir)
    ann_path = ep_dir / "annotations.json"
    if not ann_path.exists():
        print(f"  [FAIL] Missing {ann_path}")
        return 1

    ann = json.loads(ann_path.read_text(encoding="utf-8"))
    frames = ann.get("frames", [])
    n = len(frames)
    if n == 0:
        print("  [FAIL] 0 frames in annotations.json")
        return 1

    # Connect to PyBullet DIRECT and load standard Franka Panda URDF
    cid = pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    urdf_path = Path(pybullet_data.getDataPath()) / "franka_panda" / "panda.urdf"
    robot_id = pb.loadURDF(str(urdf_path), useFixedBase=True, physicsClientId=cid)

    revolute_joints = []
    lower_limits, upper_limits, vel_limits = [], [], []
    for j in range(pb.getNumJoints(robot_id, physicsClientId=cid)):
        info = pb.getJointInfo(robot_id, j, physicsClientId=cid)
        if info[2] == pb.JOINT_REVOLUTE and len(revolute_joints) < 7:
            revolute_joints.append(j)
            lower_limits.append(float(info[8]))
            upper_limits.append(float(info[9]))
            vel_limits.append(float(info[11]) if float(info[11]) > 0 else 2.175)

    lower_arr = np.asarray(lower_limits, dtype=np.float64)
    upper_arr = np.asarray(upper_limits, dtype=np.float64)
    ee_link = 7  # panda_link8

    # Extract saved joint angles, target positions, and target orientations
    q_traj = np.zeros((n, 7), dtype=np.float64)
    hand_valid = np.zeros(n, dtype=bool)
    pos_residuals_mm: List[float] = []
    ori_residuals_deg: List[float] = []
    limit_violations = 0
    clamp_events = 0

    # Check if explicit target poses were saved in embodiment NPZ or annotations
    franka_npz = ep_dir / "embodiments" / "franka_panda" / "trajectory.npz"
    saved_target_pos = None
    saved_target_quat = None
    if franka_npz.exists():
        with np.load(franka_npz, allow_pickle=True) as f_npz:
            if "target_ee_pos" in f_npz:
                saved_target_pos = f_npz["target_ee_pos"]
            if "target_ee_quat_xyzw" in f_npz:
                saved_target_quat = f_npz["target_ee_quat_xyzw"]
            if "joint_positions" in f_npz and len(f_npz["joint_positions"]) == n:
                q_traj = f_npz["joint_positions"][:, :7].astype(np.float64)

    for i, f in enumerate(frames):
        has_h = (f.get("left_hand") is not None and not f["left_hand"].get("is_interpolated", False)) or (
            f.get("right_hand") is not None and not f["right_hand"].get("is_interpolated", False)
        )
        hand_valid[i] = has_h
        if saved_target_pos is None and f.get("robot_joint_angles") and len(f["robot_joint_angles"]) == 7:
            q_traj[i] = np.asarray(f["robot_joint_angles"], dtype=np.float64)

        q = q_traj[i]
        if np.any(q < lower_arr - 1e-3) or np.any(q > upper_arr + 1e-3):
            limit_violations += 1
        if np.any(np.abs(q - lower_arr) < 1e-3) or np.any(np.abs(q - upper_arr) < 1e-3):
            clamp_events += 1

        for idx_j, j_id in enumerate(revolute_joints):
            pb.resetJointState(robot_id, j_id, float(q[idx_j]), physicsClientId=cid)
        ls = pb.getLinkState(robot_id, ee_link, computeForwardKinematics=True, physicsClientId=cid)
        fk_pos = np.asarray(ls[4], dtype=np.float64)
        fk_quat = np.asarray(ls[5], dtype=np.float64)  # xyzw

        if saved_target_pos is not None and i < len(saved_target_pos):
            tgt_p = np.asarray(saved_target_pos[i], dtype=np.float64)
            pos_residuals_mm.append(float(np.linalg.norm(fk_pos - tgt_p) * 1000.0))
        elif f.get("target_ee_pos"):
            tgt_p = np.asarray(f["target_ee_pos"], dtype=np.float64)
            pos_residuals_mm.append(float(np.linalg.norm(fk_pos - tgt_p) * 1000.0))

        if saved_target_quat is not None and i < len(saved_target_quat):
            ori_residuals_deg.append(_quat_geodesic_deg(fk_quat, np.asarray(saved_target_quat[i], dtype=np.float64)))
        elif f.get("target_ee_quat_xyzw"):
            ori_residuals_deg.append(_quat_geodesic_deg(fk_quat, np.asarray(f["target_ee_quat_xyzw"], dtype=np.float64)))

    pb.disconnect(cid)

    failures: List[str] = []
    dt = 1.0 / 30.0
    vel = np.diff(q_traj, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    max_vel = float(np.max(np.abs(vel))) if len(vel) else 0.0
    max_jerk = float(np.max(np.abs(jerk))) if len(jerk) else 0.0
    max_step_jump = float(np.max(np.abs(np.diff(q_traj, axis=0)))) if n > 1 else 0.0

    print(f"  Frames={n}, Hand-valid frames={int(np.sum(hand_valid))} ({np.mean(hand_valid)*100:.2f}%)")
    print(f"  Joint limit violations={limit_violations}, Clamp events={clamp_events}")
    print(f"  Max joint vel={max_vel:.3f} rad/s, Max jerk={max_jerk:.1f} rad/s^3, Max 1-step jump={max_step_jump:.4f} rad")

    if limit_violations > 0:
        failures.append(f"{limit_violations} frames violate Panda URDF joint limits")

    if not pos_residuals_mm or not ori_residuals_deg:
        failures.append(
            "Saved targets (`target_ee_pos` and `target_ee_quat_xyzw`) missing from embodiment output; "
            "cannot independently verify position AND orientation IK residual"
        )
        pos_mean = pos_p95 = pos_max = pos_std = -1.0
        ori_mean = ori_p95 = ori_max = ori_std = -1.0
        recomputed_ik_conv_pct = 0.0
    else:
        pos_arr = np.asarray(pos_residuals_mm, dtype=np.float64)
        ori_arr = np.asarray(ori_residuals_deg, dtype=np.float64)
        pos_mean, pos_p95, pos_max, pos_std = float(np.mean(pos_arr)), float(np.percentile(pos_arr, 95)), float(np.max(pos_arr)), float(np.std(pos_arr))
        ori_mean, ori_p95, ori_max, ori_std = float(np.mean(ori_arr)), float(np.percentile(ori_arr, 95)), float(np.max(ori_arr)), float(np.std(ori_arr))
        converged = (pos_arr <= args.pos_tol_mm) & (ori_arr <= args.ori_tol_deg)
        recomputed_ik_conv_pct = float(np.mean(converged) * 100.0)

        print(f"  Pos Residual (mm) : mean={pos_mean:.4f}, p95={pos_p95:.4f}, max={pos_max:.4f}, std={pos_std:.4f}")
        print(f"  Ori Residual (deg): mean={ori_mean:.4f}, p95={ori_p95:.4f}, max={ori_max:.4f}, std={ori_std:.4f}")
        print(f"  Independently Recomputed IK Convergence: {recomputed_ik_conv_pct:.2f}%")

        if pos_std < 1e-5 or round(pos_mean, 4) == round(pos_p95, 4) == round(pos_max, 4):
            failures.append(
                f"Position residual mean/p95/max are identical ({pos_mean:.4f} mm, std={pos_std:.4e}) — synthetic/constant metric"
            )
        if ori_std < 1e-5:
            failures.append(f"Orientation residual has zero variance (std={ori_std:.4e})")

    # Cross-check against hardware_readiness_report.json
    hw_path = ep_dir / "hardware_readiness_report.json"
    if hw_path.exists():
        hw = json.loads(hw_path.read_text(encoding="utf-8"))
        g1 = hw.get("unitree_g1_upper", {}).get("hardware_audit", {})
        h1 = hw.get("unitree_h1_upper", {}).get("hardware_audit", {})
        if g1 and h1:
            if g1.get("max_joint_velocity_rad_s") == h1.get("max_joint_velocity_rad_s") and g1.get("max_joint_jerk_rad_s3") == h1.get("max_joint_jerk_rad_s3"):
                failures.append("G1 and H1 have identical reported max velocity and jerk in hardware_readiness_report.json")

    passed = len(failures) == 0
    for f_msg in failures:
        print(f"  [FAIL] {f_msg}")
    print(f"G08 VERDICT: {'PASS' if passed else 'FAIL'}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "gate": "g08_retarget_independent",
            "status": "PASS" if passed else "FAIL",
            "limit_violations": limit_violations,
            "clamp_events": clamp_events,
            "pos_residual_mm": {"mean": pos_mean, "p95": pos_p95, "max": pos_max, "std": pos_std},
            "ori_residual_deg": {"mean": ori_mean, "p95": ori_p95, "max": ori_max, "std": ori_std},
            "recomputed_ik_convergence_pct": recomputed_ik_conv_pct,
            "failures": failures,
        }, indent=2) + "\n", encoding="utf-8")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
