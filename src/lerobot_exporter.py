"""LeRobot v2.1 dataset exporter for annotated egocentric video episodes."""

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
    """Read episode HDF5 output and convert it into LeRobot v2.1 format."""
    output_path = Path(output_dir)
    episode_dir = output_path / episode_id
    lerobot_dir = episode_dir / "lerobot_v2"

    # Create subdirectories
    meta_dir = lerobot_dir / "meta"
    data_dir = lerobot_dir / "data" / "chunk-000"
    videos_dir = lerobot_dir / "videos" / "chunk-000" / "observation.image"

    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

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

        task_desc = meta_grp["task_description"][()].decode("utf-8")
        num_frames = len(wrist_trans)

    # Reconstruct observation.state [N, 24]
    # wrist_translation(3) + wrist_rotation(6) + gripper_openness(1) + hand_pose(14) = 24D
    # gripper openness is at index 3 of proprioception dataset
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

    # Relative video path
    video_rel_path = "videos/chunk-000/observation.image/episode_000000.mp4"

    # Prepare DataFrame rows
    rows = []
    for i in range(num_frames):
        row = {
            "frame_index": i,
            "timestamp": float(timestamps[i]),
            "observation.state": obs_state[i].tolist(),
            "action": action[i].tolist(),
            "task": task_desc,
            "episode_index": 0,
            "observation.image": video_rel_path,
            "video_path": video_rel_path,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    parquet_path = data_dir / "episode_000000.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info("Saved parquet to %s", parquet_path)

    # 3. Create meta/info.json
    info = {
        "codebase_version": "v2.1",
        "robot_type": "human_egocentric",
        "fps": 30,
        "total_episodes": 1,
        "total_frames": num_frames,
        "total_tasks": 1,
        "total_videos": 1,
        "splits": {"train": "0:1"},
        "features": {
            "observation.image": {"dtype": "video", "shape": [224, 224, 3], "names": ["height", "width", "channel"]},
            "observation.state": {
                "dtype": "float32",
                "shape": [24],
                "names": [
                    "wrist_x",
                    "wrist_y",
                    "wrist_z",
                    "rot6d_0",
                    "rot6d_1",
                    "rot6d_2",
                    "rot6d_3",
                    "rot6d_4",
                    "rot6d_5",
                    "gripper",
                    "finger_0",
                    "finger_1",
                    "finger_2",
                    "finger_3",
                    "finger_4",
                    "finger_5",
                    "finger_6",
                    "finger_7",
                    "finger_8",
                    "finger_9",
                    "finger_10",
                    "finger_11",
                    "finger_12",
                    "finger_13",
                ],
            },
            "action": {
                "dtype": "float32",
                "shape": [24],
                "names": [
                    "wrist_dx",
                    "wrist_dy",
                    "wrist_dz",
                    "wrist_rot_d0",
                    "wrist_rot_d1",
                    "wrist_rot_d2",
                    "wrist_rot_d3",
                    "wrist_rot_d4",
                    "wrist_rot_d5",
                    "finger_0",
                    "finger_1",
                    "finger_2",
                    "finger_3",
                    "finger_4",
                    "finger_5",
                    "finger_6",
                    "finger_7",
                    "finger_8",
                    "finger_9",
                    "finger_10",
                    "finger_11",
                    "finger_12",
                    "finger_13",
                    "finger_14",
                ],
            },
            "task": {"dtype": "string", "shape": []},
        },
    }

    info_path = meta_dir / "info.json"
    with open(info_path, "w") as f_info:
        json.dump(info, f_info, indent=2)
    logger.info("Saved info.json to %s", info_path)

    # 4. Compile video: videos/chunk-000/observation.image/episode_000000.mp4
    project_root = Path(__file__).resolve().parent.parent
    frames_dir = project_root / "data" / "frames" / episode_id
    if not frames_dir.exists():
        frames_dir = Path("data/frames") / episode_id
        if not frames_dir.exists():
            frames_dir = Path(os.getcwd()) / "data" / "frames" / episode_id

    frame_files = sorted(list(frames_dir.glob("frame_*.png")))
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {frames_dir} to compile video.")

    video_output_path = videos_dir / "episode_000000.mp4"
    logger.info("Compiling video to %s with %d frames...", video_output_path, len(frame_files))

    writer = imageio.get_writer(str(video_output_path), fps=30, format="FFMPEG", mode="I")
    for frame_file in frame_files:
        img_bgr = cv2.imread(str(frame_file))
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if img_rgb.shape[:2] != (224, 224):
                img_rgb = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
            writer.append_data(img_rgb)
        else:
            logger.warning("Failed to read frame %s with cv2", frame_file)
    writer.close()

    logger.info("Video compiled successfully: %s", video_output_path)
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
