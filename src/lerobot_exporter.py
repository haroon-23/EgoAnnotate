"""LeRobot v3.0 dataset exporter for annotated egocentric video episodes.

Two export modes (see ``configs/default.yaml`` ``export:`` section):

- ``human`` (default): ``observation.state`` / ``action`` are the 24D human-hand
  vectors (wrist xyz, 6D palm rotation, gripper scalar, 14 finger flexions;
  deltas for action). Retargeted robot joints, when present in the episode
  HDF5, are kept as observation side-channels. For human-video pretraining and
  analysis — NOT for training a robot policy's action head.
- ``robot``: ``observation.state`` / ``action`` are ROBOT-native for the target
  embodiment. state = [7 arm joints (rad), gripper opening (m)] as reported by
  the retargeter; action = absolute joint+gripper targets
  (state at t+1, last frame holds). This is the ACT / Pi0 / SmolVLA convention.
  Episodes without retargeted joint trajectories are EXCLUDED with a logged
  reason — never silently padded.

Layout follows LeRobot v3.0::

    <root>/data/chunk-000/file-000.parquet
    <root>/videos/observation.images.<cam>/chunk-000/file-000.mp4
    <root>/meta/info.json                       # v3 templates, splits, video info block
    <root>/meta/stats.json                      # pooled per-feature stats (incl. q01/q99)
    <root>/meta/tasks.parquet
    <root>/meta/episodes/chunk-000/file-000.parquet   # per-episode stats + video-lookup cols
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HUMAN_STATE_DIM = 24
HUMAN_ACTION_DIM = 24
ROBOT_STATE_DIM = 8  # 7 arm joints + 1 gripper opening
ROBOT_JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "gripper_m"]

HUMAN_STATE_NAMES = [
    "wrist_x", "wrist_y", "wrist_z",
    "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5",
    "gripper", "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9",
    "f10", "f11", "f12", "f13",
]
HUMAN_ACTION_NAMES = [
    "w_dx", "w_dy", "w_dz",
    "rot_0", "rot_1", "rot_2", "rot_3", "rot_4", "rot_5",
    "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9",
    "f10", "f11", "f12", "f13", "f14",
]

ACTION_CONVENTION = {
    "human": "human_hand_deltas",
    "robot": "absolute_joint_positions",
}

VIDEO_CODEC = "h264"
VIDEO_PIX_FMT = "yuv420p"


class MissingRobotRetargetingError(ValueError):
    """Raised when robot-mode export is requested but the episode has no
    retargeted joint trajectory (retargeting not run or failed)."""


# ---------------------------------------------------------------------------
# Stats helpers (pooled-ready: same function serves per-episode stats.json and
# the Phase-B multi-episode merge)
# ---------------------------------------------------------------------------

def compute_feature_stats(arrays: Dict[str, np.ndarray]) -> Dict[str, dict]:
    """Compute per-feature aggregate stats over frame rows.

    Returns ``{feature_key: {min, max, mean, std, count, q01, q99}}`` with
    per-dimension lists. Quantiles are pooled over all rows (never averaged
    per-episode) for Pi0-style quantile normalization consumers.
    """
    stats: Dict[str, dict] = {}
    for key, arr in arrays.items():
        a = np.asarray(arr, dtype=np.float64)
        flat = a.reshape(a.shape[0], -1)
        stats[key] = {
            "min": flat.min(axis=0).tolist(),
            "max": flat.max(axis=0).tolist(),
            "mean": flat.mean(axis=0).tolist(),
            "std": flat.std(axis=0).tolist(),
            "count": int(a.shape[0]),
            "q01": np.quantile(flat, 0.01, axis=0).tolist(),
            "q99": np.quantile(flat, 0.99, axis=0).tolist(),
        }
    return stats


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def _find_source_video(episode_dir: Path) -> Tuple[Optional[Path], Optional[str]]:
    """Locate the original capture video for an episode.

    Source of truth is ``metadata.json`` -> ``video_path`` (written by
    DatasetExporter). We deliberately do NOT fall back to
    ``overlay_annotated.mp4``: it has annotation graphics burned in and would
    poison vision training.
    """
    meta_path = episode_dir / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as e:
            logger.warning("Could not parse %s: %s", meta_path, e)
            meta = {}
        vp = meta.get("video_path")
        if vp:
            p = Path(vp)
            if p.exists():
                return p, "metadata.video_path"
            logger.warning("Source video recorded in metadata not found on disk: %s", vp)
    return None, None


def _probe_video(path: Path) -> Tuple[float, int, int, int]:
    """Return (fps, width, height, frame_count) for a video file."""
    cap = cv2.VideoCapture(str(path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return fps, w, h, n


def _transcode_video(src: Path, dst: Path, fps: float) -> Tuple[str, str]:
    """Transcode ``src`` to h264/yuv420p mp4 at ``fps``. Returns (codec, pix_fmt)
    actually used (honest reporting for the v3 video info block)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        exe = get_ffmpeg_exe()
        cmd = [
            exe, "-y", "-i", str(src),
            "-c:v", "libx264", "-pix_fmt", VIDEO_PIX_FMT,
            "-r", f"{fps:.3f}", "-an", "-movflags", "+faststart",
            str(dst),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return VIDEO_CODEC, VIDEO_PIX_FMT
    except Exception as e:
        logger.warning("ffmpeg transcode failed (%s); falling back to cv2 mp4v", e)

    # Fallback: cv2 re-encode (mp4v). Slower, but dependency-free.
    cap = cv2.VideoCapture(str(src))
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)
        finally:
            writer.release()
    finally:
        cap.release()
    return "mp4v", VIDEO_PIX_FMT


# ---------------------------------------------------------------------------
# Episode reading
# ---------------------------------------------------------------------------

def _read_episode_hdf5(episode_dir: Path, episode_id: str) -> dict:
    hdf5_path = episode_dir / "episode_rlds.hdf5"
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    out: dict = {}
    with h5py.File(hdf5_path, "r") as f:
        ep_grp = f[episode_id]
        steps_grp = ep_grp["steps"]
        obs_grp = steps_grp["observation"]
        meta_grp = ep_grp["metadata"]

        out["wrist_trans"] = obs_grp["wrist_translation"][:]
        out["wrist_rot"] = obs_grp["wrist_rotation"][:]
        out["hand_pose"] = obs_grp["hand_pose"][:]
        out["proprio"] = obs_grp["proprioception"][:]
        out["action_human"] = steps_grp["action"][:]

        out["has_robot"] = "robot_joint_angles" in obs_grp
        if out["has_robot"]:
            out["robot_joints"] = obs_grp["robot_joint_angles"][:]
            out["robot_gripper"] = obs_grp["robot_gripper_opening_m"][:]

        out["task_description"] = meta_grp["task_description"][()].decode("utf-8")
    return out


def _read_episode_metadata(episode_dir: Path) -> dict:
    meta_path = episode_dir / "metadata.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except Exception as e:
            logger.warning("Could not parse %s: %s", meta_path, e)
    return {}


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def _build_human_features(ep: dict) -> Tuple[np.ndarray, np.ndarray, dict, dict]:
    """Return (obs_state [N,24], action [N,24], extra_obs_cols, extra_features)."""
    wrist_trans = ep["wrist_trans"]
    wrist_rot = ep["wrist_rot"]
    hand_pose = ep["hand_pose"]
    proprio = ep["proprio"]
    action = ep["action_human"]
    num_frames = len(wrist_trans)

    gripper_openness = proprio[:, 3:4]
    hand_pose_14 = hand_pose[:, :14]
    obs_state = np.hstack([wrist_trans, wrist_rot, gripper_openness, hand_pose_14]).astype(np.float32)
    action = np.asarray(action, dtype=np.float32)
    assert obs_state.shape[1] == HUMAN_STATE_DIM, obs_state.shape
    assert action.shape[1] == HUMAN_ACTION_DIM, action.shape

    extra_cols: dict = {}
    extra_features: dict = {}
    if ep["has_robot"]:
        joints = np.asarray(ep["robot_joints"], dtype=np.float32)
        grip = np.asarray(ep["robot_gripper"], dtype=np.float32).reshape(num_frames, -1)
        extra_cols["observation.robot_joint_angles"] = joints.tolist()
        extra_cols["observation.robot_gripper_opening_m"] = [float(r[0]) for r in grip]
        extra_features["observation.robot_joint_angles"] = {
            "dtype": "float32", "shape": [int(joints.shape[1])],
            "names": [f"j{i+1}" for i in range(joints.shape[1])],
        }
        extra_features["observation.robot_gripper_opening_m"] = {
            "dtype": "float32", "shape": [1], "names": None,
        }
    return obs_state, action, extra_cols, extra_features


def _build_robot_features(ep: dict, num_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    """Robot-native features. Raises MissingRobotRetargetingError when the
    episode has no retargeted joint trajectory."""
    if not ep["has_robot"]:
        raise MissingRobotRetargetingError(
            "robot export_mode requires retargeted joint trajectories "
            "('robot_joint_angles' in episode_rlds.hdf5), but this episode has none. "
            "Run the pipeline with retargeting enabled first."
        )
    joints = np.asarray(ep["robot_joints"], dtype=np.float32)
    grip = np.asarray(ep["robot_gripper"], dtype=np.float32).reshape(num_frames, -1)
    if joints.shape[0] != num_frames or grip.shape[0] != num_frames:
        raise ValueError(
            f"Robot trajectory length mismatch: joints {joints.shape[0]}, "
            f"gripper {grip.shape[0]}, frames {num_frames}"
        )
    state = np.hstack([joints, grip[:, :1]]).astype(np.float32)
    assert state.shape[1] == ROBOT_STATE_DIM, state.shape
    # Absolute action convention (ACT/Pi0/SmolVLA): action[t] = state[t+1],
    # last frame holds its own state (zero-motion target).
    action = np.empty_like(state)
    action[:-1] = state[1:]
    action[-1] = state[-1]
    return state, action


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def export_to_lerobot(
    episode_id: str,
    output_dir: str,
    *,
    export_mode: str = "human",
    camera: str = "ego",
    fps: Optional[float] = None,
    target_embodiment: Optional[str] = None,
) -> Path:
    """Convert one episode's HDF5 output into LeRobot v3.0 format.

    Args:
        episode_id: episode directory name under ``output_dir``.
        output_dir: pipeline output root containing ``<episode_id>/``.
        export_mode: ``"human"`` (24D hand vectors) or ``"robot"`` (robot-native
            8D absolute joint+gripper). Episodes lacking retargeted joints are
            rejected in robot mode via :class:`MissingRobotRetargetingError`.
        camera: camera key suffix → feature/video key ``observation.images.<camera>``.
        fps: override fps; default probes the source video, falls back to 30.
        target_embodiment: fallback ``robot_type`` when episode metadata lacks it.
    """
    if export_mode not in ("human", "robot"):
        raise ValueError(f"export_mode must be 'human' or 'robot', got {export_mode!r}")

    output_path = Path(output_dir)
    episode_dir = output_path / episode_id
    lerobot_dir = episode_dir / "lerobot_v3"
    video_key = f"observation.images.{camera}"

    meta_dir = lerobot_dir / "meta"
    data_dir = lerobot_dir / "data" / "chunk-000"
    videos_dir = lerobot_dir / "videos" / video_key / "chunk-000"
    ep_meta_dir = meta_dir / "episodes" / "chunk-000"
    for d in (meta_dir, data_dir, ep_meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Read inputs
    ep = _read_episode_hdf5(episode_dir, episode_id)
    ep_meta = _read_episode_metadata(episode_dir)
    num_frames = len(ep["wrist_trans"])
    task_desc = ep["task_description"]

    # 2. Build state/action
    extra_obs_cols: dict = {}
    extra_features: dict = {}
    if export_mode == "human":
        obs_state, action, extra_obs_cols, extra_features = _build_human_features(ep)
        state_names, action_names = HUMAN_STATE_NAMES, HUMAN_ACTION_NAMES
        robot_type = "human_egocentric"
    else:
        obs_state, action = _build_robot_features(ep, num_frames)
        state_names, action_names = ROBOT_JOINT_NAMES, ROBOT_JOINT_NAMES
        robot_type = (
            ep_meta.get("target_robot") or target_embodiment or "unknown_robot"
        )
        if robot_type == "unknown_robot":
            logger.warning("robot_type unknown for episode %s (no target_robot in metadata)", episode_id)

    # 3. Video (P2): transcode source capture video when discoverable
    video_written = False
    video_info: dict = {}
    src_video, video_source = _find_source_video(episode_dir)
    if src_video is not None:
        probed_fps, vw, vh, vn = _probe_video(src_video)
        use_fps = float(fps) if fps else (probed_fps if probed_fps > 0 else 30.0)
        fps_source = "config" if fps else ("video_probe" if probed_fps > 0 else "default")
        if vn != num_frames:
            logger.warning(
                "Video frame count (%d) != annotation frames (%d) for %s; "
                "keeping annotation frame count as ground truth.",
                vn, num_frames, episode_id,
            )
        codec, pix_fmt = _transcode_video(src_video, videos_dir / "file-000.mp4", use_fps)
        video_written = True
        video_info = {
            "video.fps": use_fps, "video.codec": codec, "video.pix_fmt": pix_fmt,
            "video.height": vh, "video.width": vw, "video.channels": 3,
            "video.is_depth_map": False,
        }
        logger.info("Wrote video %s (%dx%d @ %.2f fps) from %s",
                    videos_dir / "file-000.mp4", vw, vh, use_fps, video_source)
    else:
        use_fps = float(fps) if fps else 30.0
        fps_source = "config" if fps else "default"
        logger.warning(
            "No source video found for episode %s — exporting WITHOUT video columns. "
            "Vision policies cannot train on this export.", episode_id)

    # 4. Frame table (v3 R4: timestamp float32 = frame_index / fps, monotonic index)
    timestamps = np.arange(num_frames, dtype=np.float64) / use_fps
    df_data_dict = {
        "observation.state": obs_state.tolist(),
        "action": action.tolist(),
        "timestamp": timestamps.astype(np.float32).tolist(),
        "frame_index": list(range(num_frames)),
        "episode_index": [0] * num_frames,
        "index": list(range(num_frames)),
        "task_index": [0] * num_frames,
    }
    df_data_dict.update(extra_obs_cols)

    features_dict = {
        "observation.state": {
            "dtype": "float32", "shape": [int(obs_state.shape[1])], "names": state_names,
        },
        "action": {
            "dtype": "float32", "shape": [int(action.shape[1])], "names": action_names,
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    features_dict.update(extra_features)
    if video_written:
        vw = video_info["video.width"]; vh = video_info["video.height"]
        features_dict[video_key] = {
            "dtype": "video", "shape": [3, vh, vw],
            "names": ["channel", "height", "width"],
            "info": video_info,
        }

    df_data = pd.DataFrame(df_data_dict).astype({
        "timestamp": "float32",
        "frame_index": "int64",
        "episode_index": "int64",
        "index": "int64",
        "task_index": "int64",
    })
    parquet_path = data_dir / "file-000.parquet"
    df_data.to_parquet(parquet_path, index=False)
    logger.info("Saved parquet to %s", parquet_path)

    # 5. Stats (v3 R2 + P4: pooled-style aggregates incl. quantiles, over ALL
    # frame-table features so stats.json keys match info.json feature keys)
    stats_arrays: dict = {
        "observation.state": obs_state,
        "action": action,
        "timestamp": timestamps.astype(np.float32),
        "frame_index": np.arange(num_frames),
        "episode_index": np.zeros(num_frames, dtype=np.int64),
        "index": np.arange(num_frames),
        "task_index": np.zeros(num_frames, dtype=np.int64),
    }
    for col_key, col_vals in extra_obs_cols.items():
        stats_arrays[col_key] = np.asarray(col_vals, dtype=np.float32)
    stats = compute_feature_stats(stats_arrays)
    stats_path = meta_dir / "stats.json"
    with open(stats_path, "w") as f_stats:
        json.dump(stats, f_stats, indent=2)

    # 6. meta/info.json (v3 templates, splits, real sizes)
    data_mb = parquet_path.stat().st_size / 1e6
    video_mb = (videos_dir / "file-000.mp4").stat().st_size / 1e6 if video_written else 0.0
    info = {
        "codebase_version": "v3.0",
        "robot_type": robot_type,
        "export_mode": export_mode,
        "action_convention": ACTION_CONVENTION[export_mode],
        "fps": use_fps,
        "fps_source": fps_source,
        "total_episodes": 1,
        "total_frames": num_frames,
        "total_tasks": 1,
        "total_videos": 1 if video_written else 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "splits": {"train": "0:1"},
        "data_files_size_in_mb": round(data_mb, 3),
        "video_files_size_in_mb": round(video_mb, 3),
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features_dict,
    }
    with open(meta_dir / "info.json", "w") as f_info:
        json.dump(info, f_info, indent=2)

    # 7. meta/tasks.parquet
    pd.DataFrame([{"task_index": 0, "task": task_desc}]).to_parquet(
        meta_dir / "tasks.parquet", index=False)

    # 8. meta/episodes/chunk-000/file-000.parquet
    #    (v3 R1: per-episode stats live here, not in a standalone episodes_stats.parquet;
    #     v3 R3: video-lookup columns so the reader can resolve (episode, camera) -> mp4)
    ep_row: dict = {
        "episode_index": [0],
        "tasks": [[task_desc]],
        "length": [num_frames],
        "data/chunk_index": [0],
        "data/file_index": [0],
        "dataset_from_index": [0],
        "dataset_to_index": [num_frames],
        "export_mode": [export_mode],
        "action_convention": [ACTION_CONVENTION[export_mode]],
        "robot_type": [robot_type],
    }
    for feat_key in stats:
        s = stats[feat_key]
        for agg in ("min", "max", "mean", "std", "count"):
            ep_row[f"stats/{feat_key}/{agg}"] = [s[agg]]
    if video_written:
        ep_row[f"videos/{video_key}/chunk_index"] = [0]
        ep_row[f"videos/{video_key}/file_index"] = [0]
        ep_row[f"videos/{video_key}/from_timestamp"] = [0.0]
        ep_row[f"videos/{video_key}/to_timestamp"] = [float((num_frames - 1) / use_fps)]
    pd.DataFrame(ep_row).to_parquet(ep_meta_dir / "file-000.parquet", index=False)

    logger.info("LeRobot v3 export complete for %s (mode=%s, video=%s): %s",
                episode_id, export_mode, video_written, lerobot_dir)
    return lerobot_dir


def export_all_lerobot(
    output_dir: str = "data/output",
    *,
    export_mode: str = "human",
    **kwargs,
) -> List[Path]:
    """Export every episode with an RLDS HDF5 under ``output_dir``.

    In ``robot`` mode, episodes without retargeted joint trajectories are
    skipped with a logged reason (never silently padded).
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        logger.warning("Output directory %s does not exist. Nothing to export.", output_dir)
        return []

    exported_paths: List[Path] = []
    skipped: List[Tuple[str, str]] = []
    for path in sorted(output_path.iterdir()):
        if not (path.is_dir() and (path / "episode_rlds.hdf5").exists()):
            continue
        try:
            lerobot_path = export_to_lerobot(
                path.name, str(output_path), export_mode=export_mode, **kwargs)
            exported_paths.append(lerobot_path)
            print(f"Exported LeRobot for episode: {path.name} -> "
                  f"{lerobot_path.relative_to(output_path)}")
        except MissingRobotRetargetingError as e:
            reason = str(e)
            skipped.append((path.name, reason))
            logger.warning("Skipping episode %s in robot mode: %s", path.name, reason)
            print(f"Skipped episode {path.name} (robot mode): {reason}")
        except Exception as e:
            logger.error("Failed to export episode %s to LeRobot: %s", path.name, e, exc_info=True)
            print(f"Error exporting LeRobot for episode {path.name}: {e}")

    if skipped:
        logger.info("Robot-mode export skipped %d episode(s) lacking retargeting: %s",
                    len(skipped), [s[0] for s in skipped])
    return exported_paths
