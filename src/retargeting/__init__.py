"""Retargeting subpackage for human-to-robot kinematic retargeting.

This package converts EgoAnnotate 21-keypoint hand tracking output (MediaPipe
HandLandmarks + GraspType) into joint angle trajectories for a target robot arm.

Starting embodiment: Franka Panda (7 DOF arm + 2 DOF parallel gripper).

IMPORTANT MODELING CAVEATS — read before using outputs:
  - MediaPipe z-depth is RELATIVE (monocular estimate), not metric 3D.
  - Workspace mapping is a linear normalization approximation, not ground-truth
    metric correspondence.  Outputs represent retargeted motion style, not a
    metrically accurate replay of the human demonstration.
  - All these limitations are flagged per-frame in output metadata.
"""
from .urdf_loader import URDFLoader, RobotKinematics
from .pose_mapper import PoseMapper, PoseMapperConfig, TargetPose
from .ik_solver import IKSolver, IKSolverConfig, IKResult
from .gripper_mapper import (
    GripperMapper,
    GripperMapperConfig,
    GripperCommand,
    PANDA_FINGER_JOINT_MAX,
    opening_from_finger_joint,
    finger_joint_from_opening,
)
from .retargeter import Retargeter, RetargetingConfig, RetargetingResult
from .trajectory_smoother import TrajectorySmoother, TrajectorySmootherConfig
from .metric_calibration import MetricCalibrator, MetricCalibrationConfig, MetricCalibrationResult

__all__ = [
    "URDFLoader",
    "RobotKinematics",
    "PoseMapper",
    "PoseMapperConfig",
    "TargetPose",
    "IKSolver",
    "IKSolverConfig",
    "IKResult",
    "GripperMapper",
    "GripperMapperConfig",
    "GripperCommand",
    "PANDA_FINGER_JOINT_MAX",
    "opening_from_finger_joint",
    "finger_joint_from_opening",
    "Retargeter",
    "RetargetingConfig",
    "RetargetingResult",
    "TrajectorySmoother",
    "TrajectorySmootherConfig",
    "MetricCalibrator",
    "MetricCalibrationConfig",
    "MetricCalibrationResult",
]

