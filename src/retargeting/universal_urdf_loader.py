from __future__ import annotations
"""Multi-Embodiment Robot URDF Loader & Kinematic Workspace Profiler.

Supports custom robot URDFs (e.g. Franka Panda, UR5e, Unitree G1, Unitree H1, Allegro Hand)
and automatically discovers active joints, joint limits, velocity limits, and Forward Kinematics end-effector links.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pybullet as pb
import pybullet_data


@dataclass
class UniversalRobotProfile:
    robot_name: str
    urdf_path: str
    ee_link_index: int
    ee_link_name: str
    active_joint_indices: List[int]
    active_joint_names: List[str]
    lower_limits: np.ndarray
    upper_limits: np.ndarray
    velocity_limits: np.ndarray
    capability_status: str


def load_any_robot_urdf_profile(
    urdf_path: Optional[str] = None,
    robot_name: str = "franka_panda",
    ee_link_name: Optional[str] = None,
) -> UniversalRobotProfile:
    """Inspect a robot URDF file via PyBullet and return its kinematic profile."""
    cid = pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    if not urdf_path:
        urdf_path = str(Path(pybullet_data.getDataPath()) / "franka_panda" / "panda.urdf")

    robot_id = pb.loadURDF(str(urdf_path), useFixedBase=True, physicsClientId=cid)
    num_j = pb.getNumJoints(robot_id, physicsClientId=cid)

    rev_indices, rev_names, lowers, uppers, vels = [], [], [], [], []
    resolved_ee_idx = num_j - 1
    resolved_ee_name = "ee_link"

    for j in range(num_j):
        info = pb.getJointInfo(robot_id, j, physicsClientId=cid)
        j_name = info[1].decode("utf-8")
        j_type = info[2]
        l_name = info[12].decode("utf-8")
        if ee_link_name and l_name == ee_link_name:
            resolved_ee_idx = j
            resolved_ee_name = l_name
        elif not ee_link_name and l_name in ("panda_link8", "panda_hand", "ee_link", "tool0", "wrist_3_link"):
            resolved_ee_idx = j
            resolved_ee_name = l_name

        if j_type == pb.JOINT_REVOLUTE:
            rev_indices.append(j)
            rev_names.append(j_name)
            lowers.append(float(info[8]))
            uppers.append(float(info[9]))
            vels.append(float(info[11]) if float(info[11]) > 0 else 2.175)

    pb.disconnect(cid)

    # For Franka Panda, enforce the stricter intersection of libfranka and panda.urdf
    if "panda" in robot_name.lower() and len(rev_indices) >= 7:
        rev_indices = rev_indices[:7]
        rev_names = rev_names[:7]
        lowers = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        uppers = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
        vels = [2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610]
        resolved_ee_idx = 7
        resolved_ee_name = "panda_link8"

    return UniversalRobotProfile(
        robot_name=robot_name,
        urdf_path=str(urdf_path),
        ee_link_index=resolved_ee_idx,
        ee_link_name=resolved_ee_name,
        active_joint_indices=rev_indices,
        active_joint_names=rev_names,
        lower_limits=np.asarray(lowers, dtype=np.float64),
        upper_limits=np.asarray(uppers, dtype=np.float64),
        velocity_limits=np.asarray(vels, dtype=np.float64),
        capability_status="REAL" if "panda" in robot_name.lower() else "EXPERIMENTAL",
    )
