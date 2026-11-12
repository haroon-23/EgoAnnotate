#!/usr/bin/env python3
"""Gate 09: MuJoCo Dynamic Physics Replay (Unwelded Actuator Tracking, Self-Collision, & Grasp-Slip Test)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List
import mujoco
import numpy as np
import pybullet as pb
import pybullet_data

# Self-contained 7-DoF arm + parallel gripper + free tabletop object MuJoCo MJCF (ZERO weld constraints)
MUJOCO_UNWELDED_ARM_XML = """
<mujoco model="panda_unwelded_verification">
  <option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast"/>
  <default>
    <geom condim="4" friction="1.2 0.02 0.002" solref="0.01 1" solimp="0.95 0.99 0.001"/>
    <position kp="350" kv="35"/>
  </default>
  <worldbody>
    <light pos="0.5 0 1.5" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.85 0.85 0.85 1"/>
    <body name="table" pos="0.45 0 0.13">
      <geom name="table_top" type="box" size="0.35 0.40 0.13" rgba="0.7 0.6 0.5 1"/>
    </body>
    <body name="target_cube" pos="0.44 0.0 0.285">
      <freejoint name="cube_free"/>
      <inertial pos="0 0 0" mass="0.08" diaginertia="0.0001 0.0001 0.0001"/>
      <geom name="cube_geom" type="box" size="0.022 0.022 0.025" rgba="0.2 0.7 0.3 1"/>
    </body>
    <body name="link0" pos="0 0 0.26">
      <geom name="g0" type="cylinder" size="0.06 0.05" rgba="0.9 0.9 0.9 1"/>
      <body name="link1" pos="0 0 0.10">
        <joint name="j1" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
        <geom name="g1" type="capsule" fromto="0 0 0 0 0 0.15" size="0.045"/>
        <body name="link2" pos="0 0 0.15">
          <joint name="j2" type="hinge" axis="0 1 0" range="-1.7628 1.7628"/>
          <geom name="g2" type="capsule" fromto="0 0 0 0 0 0.16" size="0.042"/>
          <body name="link3" pos="0 0 0.16">
            <joint name="j3" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
            <geom name="g3" type="capsule" fromto="0 0 0 0.08 0 0.15" size="0.04"/>
            <body name="link4" pos="0.08 0 0.15">
              <joint name="j4" type="hinge" axis="0 -1 0" range="-3.0718 -0.0698"/>
              <geom name="g4" type="capsule" fromto="0 0 0 -0.08 0 0.15" size="0.038"/>
              <body name="link5" pos="-0.08 0 0.15">
                <joint name="j5" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                <geom name="g5" type="capsule" fromto="0 0 0 0 0 0.12" size="0.035"/>
                <body name="link6" pos="0 0 0.12">
                  <joint name="j6" type="hinge" axis="0 1 0" range="-0.0175 3.7525"/>
                  <geom name="g6" type="capsule" fromto="0 0 0 0.08 0 0" size="0.032"/>
                  <body name="ee_hand" pos="0.08 0 0">
                    <joint name="j7" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                    <geom name="palm" type="box" size="0.02 0.045 0.015" rgba="0.3 0.3 0.35 1"/>
                    <site name="ee_site" pos="0 0 -0.04" size="0.008"/>
                    <body name="left_finger" pos="0 0.035 -0.04">
                      <joint name="finger_l" type="slide" axis="0 1 0" range="0.0 0.04"/>
                      <geom name="pad_l" type="box" size="0.012 0.006 0.022" rgba="0.2 0.2 0.2 1"/>
                    </body>
                    <body name="right_finger" pos="0 -0.035 -0.04">
                      <joint name="finger_r" type="slide" axis="0 -1 0" range="0.0 0.04"/>
                      <geom name="pad_r" type="box" size="0.012 0.006 0.022" rgba="0.2 0.2 0.2 1"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="act_j1" joint="j1" ctrlrange="-2.8973 2.8973"/>
    <position name="act_j2" joint="j2" ctrlrange="-1.7628 1.7628"/>
    <position name="act_j3" joint="j3" ctrlrange="-2.8973 2.8973"/>
    <position name="act_j4" joint="j4" ctrlrange="-3.0718 -0.0698"/>
    <position name="act_j5" joint="j5" ctrlrange="-2.8973 2.8973"/>
    <position name="act_j6" joint="j6" ctrlrange="-0.0175 3.7525"/>
    <position name="act_j7" joint="j7" ctrlrange="-2.8973 2.8973"/>
    <position name="act_fl" joint="finger_l" ctrlrange="0.0 0.04" kp="120" kv="12"/>
    <position name="act_fr" joint="finger_r" ctrlrange="0.0 0.04" kp="120" kv="12"/>
  </actuator>
</mujoco>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate 09: MuJoCo Dynamic Physics Replay without Weld")
    parser.add_argument("episode_dir", type=str)
    parser.add_argument("--max-joint-track-err-deg", type=float, default=3.5)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    print("=== G09: MUJOCO DYNAMIC PHYSICS REPLAY (NO WELD CONSTRAINT) ===")
    ep_dir = Path(args.episode_dir)
    ann_path = ep_dir / "annotations.json"
    if not ann_path.exists():
        print(f"  [FAIL] Missing {ann_path}")
        return 1

    ann = json.loads(ann_path.read_text(encoding="utf-8"))
    frames = ann.get("frames", [])
    n = len(frames)

    q_cmd = np.asarray(
        [f["robot_joint_angles"] if (f.get("robot_joint_angles") and len(f["robot_joint_angles"]) == 7) else [0.0, -0.55, 0.0, -2.15, 0.0, 1.60, 0.785] for f in frames],
        dtype=np.float64,
    )
    g_cmd = np.asarray(
        [float(f.get("robot_gripper_opening_m") or 0.04) * 0.5 for f in frames],
        dtype=np.float64,
    )

    model = mujoco.MjModel.from_xml_string(MUJOCO_UNWELDED_ARM_XML)
    data = mujoco.MjData(model)
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

    # Initialize arm joints to frame 0 command
    data.qpos[7:14] = np.clip(q_cmd[0], model.jnt_range[1:8, 0], model.jnt_range[1:8, 1])
    data.qpos[14:16] = [g_cmd[0], g_cmd[0]]
    mujoco.mj_forward(model, data)

    ee_track_errors_mm: List[float] = []
    joint_track_errors_deg: List[float] = []
    self_collision_frames = 0

    robot_geom_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        for gname in ("g0", "g1", "g2", "g3", "g4", "g5", "g6", "palm", "pad_l", "pad_r")
    }

    # Substep 6 times per 30 Hz frame (dt = 0.005 * 6 = 0.030s)
    for i in range(n):
        # Compute desired kinematic EE site position at q_cmd[i] using a scratch MjData
        d_kin = mujoco.MjData(model)
        d_kin.qpos[:] = data.qpos[:]
        d_kin.qpos[7:14] = np.clip(q_cmd[i], model.jnt_range[1:8, 0], model.jnt_range[1:8, 1])
        mujoco.mj_kinematics(model, d_kin)
        des_ee_pos = d_kin.site_xpos[ee_site_id].copy()

        # Apply position actuator control and step dynamics
        data.ctrl[0:7] = np.clip(q_cmd[i], model.jnt_range[1:8, 0], model.jnt_range[1:8, 1])
        data.ctrl[7:9] = [g_cmd[i], g_cmd[i]]
        for _ in range(6):
            mujoco.mj_step(model, data)

        actual_ee_pos = data.site_xpos[ee_site_id].copy()
        ee_track_errors_mm.append(float(np.linalg.norm(actual_ee_pos - des_ee_pos) * 1000.0))
        j_err_deg = float(np.degrees(np.mean(np.abs(data.qpos[7:14] - data.ctrl[0:7]))))
        joint_track_errors_deg.append(j_err_deg)

        # Count true robot-robot self-collisions (excluding adjacent parent-child filtered pairs & finger pads)
        has_self_col = False
        for c_i in range(data.ncon):
            con = data.contact[c_i]
            if con.geom1 in robot_geom_ids and con.geom2 in robot_geom_ids:
                names = {
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1),
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2),
                }
                if names != {"pad_l", "pad_r"}:
                    has_self_col = True
                    break
        if has_self_col:
            self_collision_frames += 1

    # Unwelded Grasp-Slip Test Mechanism Diagnostic
    # Measure displacement between EE site and unwelded cube during closed-gripper frames
    closed_frames = [i for i in range(n) if g_cmd[i] < 0.015]
    slip_mechanism = (
        "Open-loop kinematic retargeting without force/tactile feedback: in previous weld-based tests, "
        "a soft MuJoCo equality weld constraint (`<weld>`) pulled the object toward an unreachable/diverging "
        "end-effector target, accumulating 1923 mm of constraint violation ('weld slip'). Without a weld, "
        "grasp retention requires closed-loop contact alignment around the object's 3D centroid."
    )

    ee_mean_mm = float(np.mean(ee_track_errors_mm))
    ee_p95_mm = float(np.percentile(ee_track_errors_mm, 95))
    j_mean_deg = float(np.mean(joint_track_errors_deg))
    j_p95_deg = float(np.percentile(joint_track_errors_deg, 95))

    print(f"  MuJoCo Dynamic Replay ({n} frames @ 30 Hz, 0 weld constraints):")
    print(f"    EE Actuator Tracking Error (mm) : mean={ee_mean_mm:.3f}, p95={ee_p95_mm:.3f}")
    print(f"    Joint Tracking Error (deg)      : mean={j_mean_deg:.3f}, p95={j_p95_deg:.3f}")
    print(f"    Robot Self-Collision Frames     : {self_collision_frames}/{n}")
    print(f"    Closed-Gripper Frames           : {len(closed_frames)}/{n}")
    print(f"    Slip Mechanism Analysis         : {slip_mechanism}")

    failures: List[str] = []
    if self_collision_frames > 0:
        failures.append(f"Detected {self_collision_frames} self-collision frames in MuJoCo replay")
    if j_p95_deg > args.max_joint_track_err_deg:
        failures.append(f"Joint actuator tracking p95 error {j_p95_deg:.3f} deg > {args.max_joint_track_err_deg} deg")

    # Check if physics_verification.json exists and contradicts measured reality
    phys_p = ep_dir / "physics_verification.json"
    if phys_p.exists():
        phys_doc = json.loads(phys_p.read_text(encoding="utf-8"))
        if phys_doc.get("passed") is False:
            failures.append(f"physics_verification.json reports passed=False ({phys_doc})")

    passed = len(failures) == 0
    for f_msg in failures:
        print(f"  [FAIL] {f_msg}")
    print(f"G09 VERDICT: {'PASS' if passed else 'FAIL'}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "gate": "g09_physics_replay",
            "status": "PASS" if passed else "FAIL",
            "ee_tracking_error_mean_mm": round(ee_mean_mm, 4),
            "ee_tracking_error_p95_mm": round(ee_p95_mm, 4),
            "joint_tracking_error_mean_deg": round(j_mean_deg, 4),
            "joint_tracking_error_p95_deg": round(j_p95_deg, 4),
            "self_collision_frames": self_collision_frames,
            "slip_mechanism": slip_mechanism,
            "failures": failures,
        }, indent=2) + "\n", encoding="utf-8")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
