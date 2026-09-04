"""Retargeting orchestrator: hand tracking → robot joint angle trajectories.

Chains URDFLoader → PoseMapper → IKSolver → GripperMapper to convert a
frame_annotations.json episode into a complete retargeting result.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..datatypes import AnnotationFrame
from .urdf_loader import URDFLoader, RobotKinematics
from .pose_mapper import PoseMapper, PoseMapperConfig, TargetPose
from .ik_solver import IKSolver, IKSolverConfig, IKResult
from .gripper_mapper import GripperMapper, GripperMapperConfig, GripperCommand

logger = logging.getLogger(__name__)


@dataclass
class RetargetingConfig:
    """Top-level configuration for the retargeting pipeline.

    Attributes:
        urdf_path: Absolute path to robot URDF. If None, uses urdf_key or Franka.
        urdf_key: Relative path in pybullet_data (e.g. "kuka_iiwa/model.urdf").
        end_effector_link: URDF link name for IK target.
        gripper_joint_names: List of prismatic finger joint names in URDF.
        pose_mapper: PoseMapper configuration.
        ik_solver: IKSolver configuration.
        gripper_mapper: GripperMapper configuration.
        output_dir: Directory for output files.
    """
    urdf_path: Optional[str] = None
    urdf_key: Optional[str] = None
    end_effector_link: str = "panda_link8"
    gripper_joint_names: List[str] = field(default_factory=lambda: ["panda_finger_joint1", "panda_finger_joint2"])
    pose_mapper: PoseMapperConfig = field(default_factory=PoseMapperConfig)
    ik_solver: IKSolverConfig = field(default_factory=IKSolverConfig)
    gripper_mapper: GripperMapperConfig = field(default_factory=GripperMapperConfig)
    output_dir: str = "data/output/retargeting"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "RetargetingConfig":
        """Load RetargetingConfig from a YAML file."""
        import yaml
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        robot_cfg = data.get("robot", {})
        pm_data = data.get("pose_mapper", {})
        ik_data = data.get("ik_solver", {})
        gm_data = data.get("gripper_mapper", {})
        out_data = data.get("output", {})

        wb = robot_cfg.get("workspace_bounds")
        pm_kwargs = {}
        if wb:
            pm_kwargs["robot_workspace_bounds"] = wb
        if "preferred_hand" in pm_data:
            pm_kwargs["preferred_hand"] = pm_data["preferred_hand"]
        if "z_floor_m" in pm_data:
            pm_kwargs["z_floor_m"] = pm_data["z_floor_m"]
        if "orientation_scale" in pm_data:
            pm_kwargs["orientation_scale"] = pm_data["orientation_scale"]

        pose_mapper = PoseMapperConfig(**pm_kwargs)

        ik_kwargs = {}
        for k in ("max_iterations", "residual_threshold_m", "joint_damping", "num_attempts"):
            if k in ik_data:
                ik_kwargs[k] = ik_data[k]
        ik_solver = IKSolverConfig(**ik_kwargs)

        gm_kwargs = {}
        for k in ("confidence_threshold", "hand_ref_size_m", "smoothing_alpha"):
            if k in gm_data:
                gm_kwargs[k] = gm_data[k]
        gripper_mapper = GripperMapperConfig(**gm_kwargs)

        return cls(
            urdf_path=robot_cfg.get("target_urdf_path"),
            urdf_key=robot_cfg.get("urdf_key"),
            end_effector_link=robot_cfg.get("end_effector_link", "panda_link8"),
            gripper_joint_names=robot_cfg.get(
                "gripper_joint_names", ["panda_finger_joint1", "panda_finger_joint2"]
            ),
            pose_mapper=pose_mapper,
            ik_solver=ik_solver,
            gripper_mapper=gripper_mapper,
            output_dir=out_data.get("output_dir", "data/output/retargeting"),
        )


@dataclass
class RetargetingResult:
    """Complete retargeting output for an episode.

    Attributes:
        episode_id: Source episode identifier.
        n_frames: Total number of frames processed.
        joint_trajectories: Arm joint angles, shape (N, n_dof), in radians.
        gripper_trajectory: Per-finger gripper opening in metres, shape (N,).
        reachability_mask: Boolean array, True where IK was valid, shape (N,).
        target_poses: List of TargetPose (workspace-mapped wrist poses).
        ik_results: List of IKResult (per-frame IK outcomes with full diagnostics).
        gripper_commands: List of GripperCommand (per-frame gripper targets).
        summary: Statistics dict.
    """
    episode_id: str
    n_frames: int
    joint_trajectories: np.ndarray    # (N, n_dof)
    gripper_trajectory: np.ndarray    # (N,)
    reachability_mask: np.ndarray     # (N,) bool
    target_poses: List[TargetPose]
    ik_results: List[IKResult]
    gripper_commands: List[GripperCommand]
    summary: Dict

    def save_json(self, path: str) -> None:
        """Serialise retargeting result to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "episode_id": self.episode_id,
            "n_frames": self.n_frames,
            "summary": self.summary,
            "frames": [
                {
                    "frame_idx": ik.frame_idx,
                    "timestamp": ik.timestamp,
                    "joint_angles_rad": ik.joint_angles.tolist(),
                    "gripper_opening_m": float(self.gripper_trajectory[i]),
                    "gripper_mapping_method": self.gripper_commands[i].gripper_mapping_method,
                    "gripper_opening_normalized": self.gripper_commands[i].opening_normalized,
                    "reachable": ik.reachable,
                    "ik_residual_m": (
                        float(ik.ik_residual_m) if not np.isnan(ik.ik_residual_m) else None
                    ),
                    "solve_time_ms": ik.solve_time_ms,
                    "joint_limit_violations": ik.joint_limit_violations,
                    "fallback_used": ik.fallback_used,
                    "has_self_collision": ik.has_self_collision,
                    "hand_detected": self.target_poses[i].hand_detected,
                    "hand_used": self.target_poses[i].hand_used,
                    "is_interpolated": self.target_poses[i].is_interpolated,
                    "target_position_m": self.target_poses[i].position.tolist(),
                    "target_quaternion": self.target_poses[i].quaternion.tolist(),
                }
                for i, ik in enumerate(self.ik_results)
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Retargeting result saved to %s", path)


class Retargeter:
    """Orchestrates the full human-to-robot kinematic retargeting pipeline.

    Usage::

        retargeter = Retargeter(config)
        result = retargeter.run_from_annotations("data/output/test_10s/frame_annotations.json")
        result.save_json("data/output/test_10s/retargeting/result.json")
    """

    def __init__(self, config: Optional[RetargetingConfig] = None) -> None:
        self.config = config or RetargetingConfig()
        self._kinematics: Optional[RobotKinematics] = None

    def load_kinematics(self) -> RobotKinematics:
        """Load robot kinematics (cached after first call)."""
        if self._kinematics is None:
            loader = URDFLoader()
            self._kinematics = loader.load(
                urdf_path=self.config.urdf_path,
                urdf_key=self.config.urdf_key,
                end_effector_link=self.config.end_effector_link,
                gripper_joint_names=tuple(self.config.gripper_joint_names)
                if self.config.gripper_joint_names is not None else (),
            )
        return self._kinematics

    def run_from_annotations(
        self,
        annotations_path: str,
        episode_id: Optional[str] = None,
    ) -> RetargetingResult:
        """Run retargeting from a frame_annotations.json file.

        Args:
            annotations_path: Path to frame_annotations.json produced by EgoAnnotate.
            episode_id: Optional identifier. If None, inferred from directory name.

        Returns:
            RetargetingResult with joint trajectories and full diagnostics.
        """
        t_start = time.perf_counter()

        # --- Load annotations -----------------------------------------------
        annotations_path = Path(annotations_path)
        if episode_id is None:
            episode_id = annotations_path.parent.name
        logger.info("Loading annotations from %s (episode: %s)", annotations_path, episode_id)
        with open(annotations_path, "r") as f:
            raw_frames = json.load(f)
        logger.info("Loaded %d raw frames", len(raw_frames))

        # Detect and parse frame format:
        # - "nested" format: AnnotationFrame.to_dict() — has keys like "left_hand" (dict or null)
        # - "flat" format: DatasetExporter output — has keys like "left_hand_keypoints" (array)
        frames: List[AnnotationFrame] = []
        if raw_frames and "left_hand_keypoints" in raw_frames[0]:
            # Flat exported format — parse manually
            logger.info("Detected flat exported frame_annotations.json format")
            frames = [self._parse_flat_frame(fr) for fr in raw_frames]
        else:
            # Nested AnnotationFrame format
            logger.info("Detected nested AnnotationFrame format")
            frames = [AnnotationFrame.from_dict(fr) for fr in raw_frames]


        # --- Load kinematics ------------------------------------------------
        kin = self.load_kinematics()

        # --- Task 2: PoseMapper ---------------------------------------------
        logger.info("Running PoseMapper...")
        pose_mapper = PoseMapper(self.config.pose_mapper)
        target_poses = pose_mapper.map_frames(frames)

        # --- Task 3: IK Solver ----------------------------------------------
        logger.info("Running IKSolver on %d target poses...", len(target_poses))
        with IKSolver(kin, self.config.ik_solver) as solver:
            ik_results = solver.solve_sequence(target_poses)

        solver.print_summary(ik_results)

        # --- Task 4: GripperMapper ------------------------------------------
        logger.info("Running GripperMapper...")
        gripper_mapper = GripperMapper(kin, self.config.gripper_mapper)
        gripper_commands = gripper_mapper.map_frames(frames)
        gripper_mapper.print_method_summary(gripper_commands)

        # --- Assemble outputs -----------------------------------------------
        n = len(frames)
        joint_traj = np.stack([r.joint_angles for r in ik_results], axis=0)  # (N, 7)
        gripper_traj = np.array([c.opening_m for c in gripper_commands], dtype=np.float64)
        reachability = np.array([r.reachable for r in ik_results], dtype=bool)

        t_elapsed = time.perf_counter() - t_start

        # Gripper method summary counts
        from collections import Counter
        method_counts = Counter(c.gripper_mapping_method for c in gripper_commands)

        summary = {
            "episode_id": episode_id,
            "n_frames": n,
            "n_reachable": int(reachability.sum()),
            "pct_reachable": float(100.0 * reachability.sum() / max(n, 1)),
            "n_fallback": sum(1 for r in ik_results if r.fallback_used),
            "n_self_collision": sum(1 for r in ik_results if r.has_self_collision),
            "avg_ik_residual_mm": float(
                np.nanmean([r.ik_residual_m * 1000 for r in ik_results])
            ),
            "avg_solve_time_ms": float(
                np.mean([r.solve_time_ms for r in ik_results if not r.fallback_used or r.reachable])
            ),
            "wall_clock_seconds": float(t_elapsed),
            "gripper_method_counts": dict(method_counts),
            "kinematics_urdf": kin.urdf_path,
            "end_effector_link": kin.end_effector_link,
            "n_arm_dof": len(kin.arm_joints),
            "arm_joint_names": [j.name for j in kin.arm_joints],
            "modeling_caveats": [
                "MediaPipe z-depth is monocular relative, not metric 3D.",
                "Workspace mapping is a linear normalization approximation — "
                "motion shape preserved, metric accuracy is NOT.",
                "Gripper mapping reduces 21-DOF human hand to 1-DOF parallel gripper.",
            ],
        }

        result = RetargetingResult(
            episode_id=episode_id,
            n_frames=n,
            joint_trajectories=joint_traj,
            gripper_trajectory=gripper_traj,
            reachability_mask=reachability,
            target_poses=target_poses,
            ik_results=ik_results,
            gripper_commands=gripper_commands,
            summary=summary,
        )

        self._print_result_summary(summary)
        return result

    def _print_result_summary(self, summary: Dict) -> None:
        sep = "=" * 70
        print(f"\n{sep}")
        print("  RETARGETING RESULT SUMMARY")
        print(f"{sep}")
        print(f"  Episode             : {summary['episode_id']}")
        print(f"  Frames              : {summary['n_frames']}")
        print(f"  Reachable           : {summary['n_reachable']} ({summary['pct_reachable']:.1f}%)")
        print(f"  Fallback used       : {summary['n_fallback']}")
        print(f"  Self-collision evts : {summary['n_self_collision']}")
        print(f"  Avg IK residual     : {summary['avg_ik_residual_mm']:.2f} mm")
        print(f"  Avg IK solve time   : {summary['avg_solve_time_ms']:.2f} ms/frame")
        print(f"  Total wall-clock    : {summary['wall_clock_seconds']:.1f} s")
        print()
        print("  Gripper mapping method breakdown:")
        for method, cnt in summary["gripper_method_counts"].items():
            pct = 100.0 * cnt / max(summary["n_frames"], 1)
            print(f"    {method:<30} {cnt:>5} frames ({pct:.1f}%)")
        print()
        print("  MODELING CAVEATS:")
        for caveat in summary["modeling_caveats"]:
            print(f"    • {caveat}")
        print(f"{sep}\n")

    @staticmethod
    def _parse_flat_frame(fr: Dict) -> AnnotationFrame:
        """Parse a flat DatasetExporter frame record into an AnnotationFrame.

        The flat format uses:
            - left_hand_keypoints / right_hand_keypoints: flat list of 63 floats [x0,y0,z0, x1,y1,z1, ...]
            - left_hand_present / right_hand_present: bool
            - left_contact / right_contact: bool (contact state)
            - left_grasp_type / right_grasp_type: str or null
            - action_gripper_openness: float
        """
        from ..datatypes import HandLandmarks, GraspType, ContactState

        def _parse_hand(kp_flat, present: bool, handedness: str) -> Optional["HandLandmarks"]:
            if not present or kp_flat is None:
                return None
            arr = np.array(kp_flat, dtype=np.float32).reshape(21, 3)
            return HandLandmarks(
                x=arr[:, 0],
                y=arr[:, 1],
                z=arr[:, 2],
                confidence=0.8,  # not stored in flat format, use default
                handedness=handedness,
                is_interpolated=False,
            )

        def _parse_grasp(grasp_type_str: Optional[str], hand: Optional["HandLandmarks"]) -> Optional["GraspType"]:
            if grasp_type_str is None or hand is None:
                return None
            # Estimate thumb-index distance from keypoints
            p4 = np.array([hand.x[4], hand.y[4], hand.z[4]])
            p8 = np.array([hand.x[8], hand.y[8], hand.z[8]])
            dist = float(np.linalg.norm(p4 - p8))
            return GraspType(
                type=grasp_type_str,
                confidence=0.7,  # not stored in flat format, use default
                thumb_index_distance=dist,
                num_curled_fingers=2,  # not stored in flat format
            )

        left_hand = _parse_hand(
            fr.get("left_hand_keypoints"), fr.get("left_hand_present", False), "Left"
        )
        right_hand = _parse_hand(
            fr.get("right_hand_keypoints"), fr.get("right_hand_present", False), "Right"
        )

        return AnnotationFrame(
            frame_idx=int(fr["frame_idx"]),
            timestamp=float(fr["timestamp"]),
            image_path=str(fr.get("image_path", "")),
            left_hand=left_hand,
            right_hand=right_hand,
            left_grasp=_parse_grasp(fr.get("left_grasp_type"), left_hand),
            right_grasp=_parse_grasp(fr.get("right_grasp_type"), right_hand),
        )
