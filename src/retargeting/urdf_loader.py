"""URDF loading and kinematic chain extraction for robot retargeting.

Task 1 of the retargeting pipeline: parse the target robot's URDF using PyBullet,
extract joint metadata, and optionally validate against known published specifications.

All joint limits come EXCLUSIVELY from the URDF — nothing is hardcoded.
This module is URDF-agnostic: it works with any robot URDF. No Franka-specific
assumptions are made at runtime — end-effector link and gripper joint names are
always supplied by the caller (via RetargetingConfig / YAML).

The _PANDA_KNOWN_LIMITS table is used ONLY when the loaded URDF is identified as
a Franka Panda (robot_name == "panda") to cross-validate limits. It is completely
silent for any other robot.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known URDF joint limits for cross-validation — Franka Panda only.
#
# This table is ONLY consulted when robot_name == "panda". For all other
# robots, validation is skipped (no table exists for them yet — add entries
# here if you want cross-validation for additional robots).
#
# NOTE ON TWO SETS OF LIMITS for Franka:
# The PyBullet bundled panda.urdf uses "URDF position limits" which are exactly
# 0.0698 rad (4°) wider than Franka's published "software safety limits".
# The table below matches the BUNDLED URDF limits (the source of truth).
#
# For real deployment: re-clip to Franka safety limits (±2.8973 etc.) before
# commanding the physical robot — the firmware will reject the wider values.
# ---------------------------------------------------------------------------
_KNOWN_LIMITS_PANDA: Dict[str, Tuple[float, float]] = {
    "panda_joint1": (-2.9671, 2.9671),
    "panda_joint2": (-1.8326, 1.8326),
    "panda_joint3": (-2.9671, 2.9671),
    "panda_joint4": (-3.1416, 0.0000),
    "panda_joint5": (-2.9671, 2.9671),
    "panda_joint6": (-0.0873, 3.8223),
    "panda_joint7": (-2.9671, 2.9671),
    "panda_finger_joint1": (0.0, 0.04),   # meters (prismatic)
    "panda_finger_joint2": (0.0, 0.04),   # meters (prismatic)
}

# Registry of known-spec tables, keyed by robot_name.
# Add entries here to enable cross-validation for additional robots.
_KNOWN_LIMITS_BY_ROBOT: Dict[str, Dict[str, Tuple[float, float]]] = {
    "panda": _KNOWN_LIMITS_PANDA,
}

# Keep the old name as an alias for tests that import it directly
_PANDA_KNOWN_LIMITS = _KNOWN_LIMITS_PANDA

LIMIT_WARN_THRESHOLD_RAD = 0.005  # radians — flag if URDF deviates from known table
LIMIT_WARN_THRESHOLD_M = 0.001    # metres — for prismatic (finger) joints

# PyBullet joint type constants (from pybullet source)
_PB_REVOLUTE = 0
_PB_PRISMATIC = 1
_PB_SPHERICAL = 2
_PB_PLANAR = 3
_PB_FIXED = 4

_JOINT_TYPE_NAMES = {
    _PB_REVOLUTE: "revolute",
    _PB_PRISMATIC: "prismatic",
    _PB_SPHERICAL: "spherical",
    _PB_PLANAR: "planar",
    _PB_FIXED: "fixed",
}


@dataclass
class JointInfo:
    """Metadata for a single robot joint extracted from the URDF."""
    index: int              # PyBullet joint index
    name: str               # Joint name as in URDF
    type: str               # "revolute", "prismatic", "fixed", etc.
    lower_limit: float      # Lower joint limit (rad or m)
    upper_limit: float      # Upper joint limit (rad or m)
    parent_link: str        # Name of the parent link
    child_link: str         # Name of the child link
    axis: Tuple[float, float, float]  # Joint axis in local frame


@dataclass
class RobotKinematics:
    """Complete kinematic description extracted from a URDF.

    Attributes:
        urdf_path: Absolute path to the URDF file.
        robot_name: Name attribute from the URDF root element.
        base_link: Name of the robot base link.
        end_effector_link: Name of the end-effector link used for IK.
        end_effector_index: PyBullet link index for the end-effector.
        joints: All joints (including fixed) in PyBullet index order.
        active_joints: Revolute and prismatic joints only (controllable DOFs).
        arm_joints: Revolute arm joints (excludes gripper fingers).
        gripper_joints: Prismatic finger joints.
        arm_joint_indices: PyBullet joint indices for the arm DOFs.
        gripper_joint_indices: PyBullet joint indices for the gripper.
        lower_limits: Lower joint limits array for arm joints, shape (n_arm,).
        upper_limits: Upper joint limits array for arm joints, shape (n_arm,).
        rest_poses: Neutral/home joint angles for arm joints, shape (n_arm,).
        gripper_max_opening_m: Maximum gripper opening in metres (sum of both
            finger joint maxima for a parallel gripper).
        validation_warnings: List of limit-mismatch warnings vs known spec.
    """
    urdf_path: str
    robot_name: str
    base_link: str
    end_effector_link: str
    end_effector_index: int
    joints: List[JointInfo]
    active_joints: List[JointInfo]
    arm_joints: List[JointInfo]
    gripper_joints: List[JointInfo]
    arm_joint_indices: List[int]
    gripper_joint_indices: List[int]
    lower_limits: "np.ndarray"  # shape (n_arm,)
    upper_limits: "np.ndarray"  # shape (n_arm,)
    rest_poses: "np.ndarray"    # shape (n_arm,)
    gripper_max_opening_m: float
    validation_warnings: List[str] = field(default_factory=list)


class URDFLoader:
    """Loads a robot URDF via PyBullet and extracts kinematic metadata.

    Uses PyBullet in DIRECT (headless) mode — no display required.
    URDF-agnostic: works with any robot. No Franka-specific defaults.

    The caller MUST supply end_effector_link and gripper_joint_names
    (via RetargetingConfig loaded from a YAML file). There are no fallback
    defaults — omitting them raises a clear error rather than silently
    using wrong Franka-specific values.

    Example::

        loader = URDFLoader()
        kinematics = loader.load(
            "path/to/kuka_iiwa/model.urdf",
            end_effector_link="lbr_iiwa_link_7",
            gripper_joint_names=[],
        )
    """

    def __init__(self) -> None:
        self._pb = None  # lazy import

    def _get_pybullet(self):
        """Lazy-import pybullet and open a DIRECT physics client."""
        try:
            import pybullet as pb
            import pybullet_data
        except ImportError as exc:
            raise ImportError(
                "pybullet is required for URDFLoader. "
                "Install with: pip install pybullet>=3.2.5"
            ) from exc
        return pb, pybullet_data

    def find_panda_urdf(self) -> str:
        """Return the path to the bundled Franka Panda URDF from pybullet_data.

        Raises:
            FileNotFoundError: if pybullet_data does not contain the panda URDF.
        """
        import os
        pb, pybullet_data = self._get_pybullet()
        data_path = pybullet_data.getDataPath()
        urdf_path = os.path.join(data_path, "franka_panda", "panda.urdf")
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(
                f"Franka Panda URDF not found at expected location: {urdf_path}\n"
                f"pybullet_data root: {data_path}\n"
                "Ensure pybullet>=3.2.5 is installed."
            )
        return urdf_path

    def load(
        self,
        urdf_path: Optional[str] = None,
        end_effector_link: Optional[str] = None,
        gripper_joint_names: Optional[Tuple[str, ...]] = None,
        urdf_key: Optional[str] = None,
    ) -> RobotKinematics:
        """Load a URDF and extract kinematic metadata.

        This method is URDF-agnostic. All robot-specific names must be
        provided by the caller — there are no Franka fallback defaults.

        Args:
            urdf_path: Absolute path to the URDF file. If None and urdf_key
                is None, falls back to the bundled Franka Panda URDF.
            end_effector_link: Name of the link to use as IK target.
                REQUIRED — raises ValueError if None.
            gripper_joint_names: Names of the prismatic gripper joints.
                Pass an empty tuple/list for robots with no gripper URDF.
                REQUIRED — raises ValueError if None.
            urdf_key: Path relative to pybullet_data root (e.g.
                "kuka_iiwa/model.urdf"). Used when urdf_path is None and
                you want a bundled URDF other than the Franka Panda default.

        Returns:
            Populated :class:`RobotKinematics` dataclass.

        Raises:
            ValueError: If end_effector_link or gripper_joint_names is None,
                or if end_effector_link is not found in the loaded URDF.
        """
        if end_effector_link is None:
            raise ValueError(
                "end_effector_link is required and was not provided. "
                "Set it in your retargeting YAML under robot.end_effector_link."
            )
        if gripper_joint_names is None:
            raise ValueError(
                "gripper_joint_names is required and was not provided. "
                "Set it in your retargeting YAML under robot.gripper_joint_names. "
                "Use an empty list ([]) for robots with no gripper in the URDF."
            )
        import numpy as np
        pb, pybullet_data = self._get_pybullet()

        if urdf_path is None:
            if urdf_key is not None:
                # Load a bundled URDF by relative key (e.g. "kuka_iiwa/model.urdf")
                import os
                urdf_path = os.path.join(pybullet_data.getDataPath(), urdf_key)
                if not os.path.exists(urdf_path):
                    raise FileNotFoundError(
                        f"urdf_key '{urdf_key}' not found under pybullet_data: {urdf_path}"
                    )
            else:
                # Default: bundled Franka Panda (backward compatibility only)
                urdf_path = self.find_panda_urdf()
        logger.info("Loading URDF from: %s", urdf_path)

        # Open a temporary DIRECT physics client
        client_id = pb.connect(pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)

        try:
            robot_id = pb.loadURDF(
                urdf_path,
                basePosition=[0, 0, 0],
                useFixedBase=True,
                physicsClientId=client_id,
            )
        except Exception as exc:
            pb.disconnect(client_id)
            raise RuntimeError(f"Failed to load URDF from {urdf_path}: {exc}") from exc

        n_joints = pb.getNumJoints(robot_id, physicsClientId=client_id)
        logger.info("URDF loaded: %d joints found", n_joints)

        # ----------------------------------------------------------------
        # Enumerate all joints
        # ----------------------------------------------------------------
        all_joints: List[JointInfo] = []
        link_name_to_index: Dict[str, int] = {}  # child link name → joint index

        for ji in range(n_joints):
            info = pb.getJointInfo(robot_id, ji, physicsClientId=client_id)
            # info tuple: (index, name, type, qindex, uindex, flags,
            #   damping, friction, lower, upper, maxForce, maxVel,
            #   linkName, axis, parentFramePos, parentFrameOrn, parentIndex)
            joint_index = int(info[0])
            joint_name = info[1].decode("utf-8")
            joint_type_int = int(info[2])
            lower = float(info[8])
            upper = float(info[9])
            child_link = info[12].decode("utf-8")
            axis = (float(info[13][0]), float(info[13][1]), float(info[13][2]))
            parent_index = int(info[16])
            parent_link = (
                all_joints[parent_index].child_link
                if parent_index >= 0 and parent_index < len(all_joints)
                else "base"
            )

            jinfo = JointInfo(
                index=joint_index,
                name=joint_name,
                type=_JOINT_TYPE_NAMES.get(joint_type_int, f"unknown({joint_type_int})"),
                lower_limit=lower,
                upper_limit=upper,
                parent_link=parent_link,
                child_link=child_link,
                axis=axis,
            )
            all_joints.append(jinfo)
            link_name_to_index[child_link] = joint_index

        # ----------------------------------------------------------------
        # Resolve end-effector link index
        # ----------------------------------------------------------------
        if end_effector_link not in link_name_to_index:
            pb.disconnect(client_id)
            available = sorted(link_name_to_index.keys())
            raise ValueError(
                f"End-effector link '{end_effector_link}' not found in URDF.\n"
                f"Available links: {available}"
            )
        ee_index = link_name_to_index[end_effector_link]

        # ----------------------------------------------------------------
        # Separate active (non-fixed) joints
        # ----------------------------------------------------------------
        active_joints = [j for j in all_joints if j.type in ("revolute", "prismatic")]
        gripper_joints = [j for j in active_joints if j.name in gripper_joint_names]
        gripper_joint_name_set = set(gripper_joint_names)
        arm_joints = [
            j for j in active_joints
            if j.type == "revolute" and j.name not in gripper_joint_name_set
        ]

        logger.info(
            "Active joints: %d arm revolute + %d gripper prismatic",
            len(arm_joints), len(gripper_joints),
        )

        # ----------------------------------------------------------------
        # Build limit arrays for IK solver
        # ----------------------------------------------------------------
        lower_limits = np.array([j.lower_limit for j in arm_joints], dtype=np.float64)
        upper_limits = np.array([j.upper_limit for j in arm_joints], dtype=np.float64)
        # Rest pose: midpoint of each joint range as neutral start
        rest_poses = (lower_limits + upper_limits) / 2.0

        # ----------------------------------------------------------------
        # Gripper max opening (sum of both finger maxima for parallel gripper)
        # ----------------------------------------------------------------
        gripper_max_opening_m = sum(j.upper_limit for j in gripper_joints)

        # ----------------------------------------------------------------
        # Validate against known spec (robot-conditional, silent for others)
        # ----------------------------------------------------------------
        warnings: List[str] = []
        import os as _os
        _robot_name_for_check = _os.path.splitext(_os.path.basename(str(urdf_path)))[0]
        known_limits = _KNOWN_LIMITS_BY_ROBOT.get(_robot_name_for_check, {})
        if known_limits:
            logger.info("Cross-validating limits against known spec for robot: %s",
                        _robot_name_for_check)
        for jinfo in all_joints:
            if jinfo.name in known_limits:
                expected_low, expected_high = known_limits[jinfo.name]
                threshold = (
                    LIMIT_WARN_THRESHOLD_M
                    if jinfo.type == "prismatic"
                    else LIMIT_WARN_THRESHOLD_RAD
                )
                unit = "m" if jinfo.type == "prismatic" else "rad"
                if abs(jinfo.lower_limit - expected_low) > threshold:
                    msg = (
                        f"⚠  {jinfo.name} lower limit: URDF={jinfo.lower_limit:.4f}{unit}, "
                        f"known spec={expected_low:.4f}{unit}, "
                        f"delta={abs(jinfo.lower_limit - expected_low):.4f}{unit}"
                    )
                    warnings.append(msg)
                    logger.warning(msg)
                if abs(jinfo.upper_limit - expected_high) > threshold:
                    msg = (
                        f"⚠  {jinfo.name} upper limit: URDF={jinfo.upper_limit:.4f}{unit}, "
                        f"known spec={expected_high:.4f}{unit}, "
                        f"delta={abs(jinfo.upper_limit - expected_high):.4f}{unit}"
                    )
                    warnings.append(msg)
                    logger.warning(msg)

        # ----------------------------------------------------------------
        # Get robot name from URDF (not directly exposed by PyBullet —
        # parse from the filename as a reasonable fallback)
        # ----------------------------------------------------------------
        import os
        robot_name = os.path.splitext(os.path.basename(urdf_path))[0]

        pb.disconnect(client_id)

        kin = RobotKinematics(
            urdf_path=str(urdf_path),
            robot_name=robot_name,
            base_link="base",
            end_effector_link=end_effector_link,
            end_effector_index=ee_index,
            joints=all_joints,
            active_joints=active_joints,
            arm_joints=arm_joints,
            gripper_joints=gripper_joints,
            arm_joint_indices=[j.index for j in arm_joints],
            gripper_joint_indices=[j.index for j in gripper_joints],
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            rest_poses=rest_poses,
            gripper_max_opening_m=gripper_max_opening_m,
            validation_warnings=warnings,
        )

        self._print_report(kin)
        return kin

    def _print_report(self, kin: RobotKinematics) -> None:
        """Print a human-readable summary of the extracted kinematics."""
        sep = "=" * 70
        print(f"\n{sep}")
        print(f"  URDF KINEMATIC REPORT — {kin.robot_name}")
        print(f"{sep}")
        print(f"  URDF path       : {kin.urdf_path}")
        print(f"  End-effector    : {kin.end_effector_link} (link index {kin.end_effector_index})")
        print(f"  Arm DOF         : {len(kin.arm_joints)}")
        print(f"  Gripper DOF     : {len(kin.gripper_joints)}")
        print(f"  Gripper max open: {kin.gripper_max_opening_m*100:.1f} cm")
        print()
        print(f"  {'Joint':<28} {'Type':<10} {'Lower (rad/m)':>14} {'Upper (rad/m)':>14}")
        print(f"  {'-'*28} {'-'*10} {'-'*14} {'-'*14}")
        for j in kin.arm_joints:
            print(f"  {j.name:<28} {j.type:<10} {j.lower_limit:>14.4f} {j.upper_limit:>14.4f}")
        for j in kin.gripper_joints:
            print(f"  {j.name:<28} {j.type:<10} {j.lower_limit:>14.4f} {j.upper_limit:>14.4f}")
        if kin.validation_warnings:
            print()
            print("  VALIDATION WARNINGS (URDF vs known spec):")
            for w in kin.validation_warnings:
                print(f"  {w}")
        else:
            print()
            known = _KNOWN_LIMITS_BY_ROBOT.get(kin.robot_name)
            if known:
                print(f"  ✓  All joint limits match known spec for '{kin.robot_name}' within threshold.")
            else:
                print(f"  ℹ  No known spec table for robot '{kin.robot_name}' — skipping limit cross-validation.")
                print("     (Add an entry to _KNOWN_LIMITS_BY_ROBOT in urdf_loader.py to enable it.)")
        print(f"{sep}\n")
