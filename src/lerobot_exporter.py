"""LeRobot v3.0 dataset exporter for annotated egocentric video episodes.

VECTOR LAYOUT SPECIFICATION:

1. observation.state (24, float32):
   - Indices 0..2  : Human Wrist Position [x, y, z] in normalized workspace coordinates (metres).
   - Indices 3..8  : Continuous 6D Palm Rotation [r0..r5] (first two columns of 3x3 rotation matrix).
   - Index 9       : Gripper Openness scalar [0.0 = closed/pinch, 1.0 = fully open].
   - Indices 10..23: Hand Finger Joint Pose [f0..f13] (14 finger joint flexions).

2. action (24, float32):
   - Indices 0..2  : Wrist Delta Translation [w_dx, w_dy, w_dz].
   - Indices 3..8  : Wrist Delta 6D Rotation [rot_0..rot_5].
   - Indices 9..23 : Finger Joint Angle Delta / Actions [f0..f14] (15 finger action deltas).

3. observation.robot_joint_angles (7, float32) [when retargeting enabled]:
   - Indices 0..6  : Robot Arm Joint Angles in radians (j1..j7).

4. observation.robot_gripper_opening_m (1, float32) [when retargeting enabled]:
   - Index 0       : Parallel-jaw Gripper Opening Distance in metres (0.0 to 0.08m for Franka).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List

import cv2
import h5py
import imageio
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def export_to_lerobot(episode_id: str, output_dir: str) -> Path:
    """Read episode HDF5 output and convert it into LeRobot v3.0 format."""
    output_path = Path(output_dir)
    episode_dir = output_path / episode_id
    lerobot_dir = episode_dir / "lerobot_v3"

    # Create subdirectories according to LeRobot v3.0 spec
    meta_dir = lerobot_dir / "meta"
    data_dir = lerobot_dir / "data" / "chunk-000"
    videos_dir = lerobot_dir / "videos" / "observation.image" / "chunk-000"
    ep_meta_dir = meta_dir / "episodes" / "chunk-000"

    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    ep_meta_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read episode_rlds.hdf5
    hdf5_path = episode_dir / "episode_rlds.hdf5"
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as f:
        ep_grp = f[episode_id]
        steps_grp = ep_grp["steps"]
        obs_grp = steps_grp["observation"]
        meta_grp = ep_grp["metadata"]

        wrist_trans = obs_grp["wrist_translation"][:]
        wrist_rot = obs_grp["wrist_rotation"][:]
        hand_pose = obs_grp["hand_pose"][:]
        proprio = obs_grp["proprioception"][:]
        action = steps_grp["action"][:]

        has_robot_retargeting = "robot_joint_angles" in obs_grp
        robot_joint_angles = obs_grp["robot_joint_angles"][:] if has_robot_retargeting else None
        robot_gripper_openings = obs_grp["robot_gripper_opening_m"][:] if has_robot_retargeting else None

        task_desc = meta_grp["task_description"][()].decode("utf-8")
        num_frames = len(wrist_trans)

    # Reconstruct observation.state [N, 24]
    gripper_openness = proprio[:, 3:4]  # shape [N, 1]
    hand_pose_14 = hand_pose[:, :14]   # shape [N, 14]
    obs_state = np.hstack([wrist_trans, wrist_rot, gripper_openness, hand_pose_14]).astype(np.float32)

    # 2. Get frame timestamps
    timestamps = []
    frame_annot_path = episode_dir / "frame_annotations.json"
    if frame_annot_path.exists():
        try:
            with open(frame_annot_path, "r") as fa:
                fa_data = json.load(fa)
                timestamps = [frame["timestamp"] for frame in fa_data]
        except Exception as e:
            logger.warning("Failed to parse timestamps from frame annotations: %s", e)

    if len(timestamps) != num_frames:
        timestamps = [i / 30.0 for i in range(num_frames)]

    # 3. Write data/chunk-000/file-000.parquet
    df_data_dict = {
        "frame_index": list(range(num_frames)),
        "timestamp": [float(t) for t in timestamps],
        "episode_index": [0] * num_frames,
        "index": list(range(num_frames)),
        "task_index": [0] * num_frames,
        "observation.state": obs_state.tolist(),
        "action": action.tolist(),
    }

    features_dict = {
        "observation.state": {
            "dtype": "float32",
            "shape": [24],
            "names": [
                "wrist_x", "wrist_y", "wrist_z",
                "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5",
                "gripper", "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13"
            ]
        },
        "action": {
            "dtype": "float32",
            "shape": [24],
            "names": [
                "w_dx", "w_dy", "w_dz",
                "rot_0", "rot_1", "rot_2", "rot_3", "rot_4", "rot_5",
                "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14"
            ]
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None}
    }

    if has_robot_retargeting:
        df_data_dict["observation.robot_joint_angles"] = robot_joint_angles.tolist()
        df_data_dict["observation.robot_gripper_opening_m"] = [float(r[0]) for r in robot_gripper_openings]
        features_dict["observation.robot_joint_angles"] = {
            "dtype": "float32",
            "shape": [7],
            "names": ["j1", "j2", "j3", "j4", "j5", "j6", "j7"]
        }
        features_dict["observation.robot_gripper_opening_m"] = {
            "dtype": "float32",
            "shape": [1],
            "names": None
        }

    df_data = pd.DataFrame(df_data_dict)
    parquet_path = data_dir / "file-000.parquet"
    df_data.to_parquet(parquet_path, index=False)
    logger.info("Saved parquet to %s", parquet_path)

    # 4. Write meta metadata files
    # meta/info.json
    info = {
        "codebase_version": "v3.0",
        "robot_type": "human_egocentric",
        "fps": 30,
        "total_episodes": 1,
        "total_frames": num_frames,
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 50,
        "video_files_size_in_mb": 500,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features_dict
    }

    info_path = meta_dir / "info.json"
    with open(info_path, "w") as f_info:
        json.dump(info, f_info, indent=2)

    # meta/tasks.parquet
    df_tasks = pd.DataFrame([{"task_index": 0, "task": task_desc}])
    df_tasks.to_parquet(meta_dir / "tasks.parquet", index=False)

    # meta/episodes/chunk-000/file-000.parquet
    df_episodes = pd.DataFrame([{
        "episode_index": 0,
        "tasks": [task_desc],
        "length": num_frames,
        "data/chunk_index": 0,
        "data/file_index": 0,
        "dataset_from_index": 0,
        "dataset_to_index": num_frames,
    }])
    df_episodes.to_parquet(ep_meta_dir / "file-000.parquet", index=False)

    # meta/episodes_stats.parquet
    df_stats = pd.DataFrame([{
        "episode_index": 0,
        "stats/observation.state/mean": obs_state.mean(axis=0).tolist(),
        "stats/observation.state/std": (obs_state.std(axis=0) + 1e-6).tolist(),
        "stats/observation.state/min": obs_state.min(axis=0).tolist(),
        "stats/observation.state/max": obs_state.max(axis=0).tolist(),
        "stats/action/mean": action.mean(axis=0).tolist(),
        "stats/action/std": (action.std(axis=0) + 1e-6).tolist(),
        "stats/action/min": action.min(axis=0).tolist(),
        "stats/action/max": action.max(axis=0).tolist(),
    }])
    df_stats.to_parquet(meta_dir / "episodes_stats.parquet", index=False)

    logger.info("Saved metadata files to %s", meta_dir)
    return lerobot_dir


def export_all_lerobot(output_dir: str = "data/output") -> List[Path]:
    """Scan output directory and export all episodes with RLDS HDF5 to LeRobot format."""
    output_path = Path(output_dir)
    if not output_path.exists():
        logger.warning("Output directory %s does not exist. Nothing to export.", output_dir)
        return []

    exported_paths = []
    for path in output_path.iterdir():
        if path.is_dir() and (path / "episode_rlds.hdf5").exists():
            try:
                lerobot_path = export_to_lerobot(path.name, str(output_path))
                exported_paths.append(lerobot_path)
                print(f"Successfully exported LeRobot for episode: {path.name} -> {lerobot_path.relative_to(output_path)}")
            except Exception as e:
                logger.error("Failed to export episode %s to LeRobot: %s", path.name, e, exc_info=True)
                print(f"Error exporting LeRobot for episode {path.name}: {e}")

    return exported_paths
