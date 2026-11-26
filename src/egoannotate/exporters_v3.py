"""Multi-format exporters: LeRobot v3.0 (+GR00T modality), RLDS, HDF5, Parquet+JSON, MCAP & dataset_card."""

from __future__ import annotations

import importlib
import json
import struct
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.egoannotate.config import PipelineConfig
from src.egoannotate.privacy_v3 import export_sanitized_mp4
from src.egoannotate.registry import REGISTRY
from src.egoannotate.retargeting_v3 import EmbodimentTrajectory
from src.egoannotate.robot_contracts import CANONICAL_DELTA_ACTION_NAMES
from src.egoannotate.schemas import EpisodeIR

_MCAP_MAGIC = b"\x89MCAP0\r\n"


def _build_embodiment_agnostic_arrays(ir: EpisodeIR) -> dict[str, np.ndarray]:
    """Pack 30D state, 30D delta action, 16D canonical SE(3) delta, 18D tactile, 7D cam pose & masks."""
    n = len(ir.frames)
    state = np.zeros((n, 30), dtype=np.float32)
    canonical_delta = np.zeros((n, 16), dtype=np.float32)
    tactile = np.zeros((n, 18), dtype=np.float32)
    cam_pose = np.zeros((n, 7), dtype=np.float32)
    hand_valid = np.zeros((n, 2), dtype=np.int64)
    proprio_valid = np.zeros(n, dtype=np.bool_)
    timestamps = np.zeros(n, dtype=np.float64)

    for i, fr in enumerate(ir.frames):
        timestamps[i] = float(fr.timestamp_s)
        cam_pose[i, :3] = fr.camera_pose.pos_world_m
        cam_pose[i, 3:7] = fr.camera_pose.quat_xyzw_world

        lh, rh = fr.left_hand, fr.right_hand
        hand_valid[i, 0] = 1 if lh.detected else 0
        hand_valid[i, 1] = 1 if rh.detected else 0
        proprio_valid[i] = bool(fr.proprio_valid)

        state[i, 0:3] = lh.wrist_pos_world_m
        state[i, 3:9] = lh.wrist_rot6d_world
        state[i, 9:14] = lh.finger_curls_5d
        state[i, 14] = lh.gripper_aperture_m

        state[i, 15:18] = rh.wrist_pos_world_m
        state[i, 18:24] = rh.wrist_rot6d_world
        state[i, 24:29] = rh.finger_curls_5d
        state[i, 29] = rh.gripper_aperture_m

        if len(fr.canonical_delta_action) == 16:
            canonical_delta[i] = np.asarray(fr.canonical_delta_action, dtype=np.float32)

        tl, tr = fr.tactile_left, fr.tactile_right
        tactile[i, 0] = tl.contact_prob
        tactile[i, 1:7] = tl.contact_zones_6d
        tactile[i, 7] = tl.normal_force_proxy_01
        tactile[i, 8] = tl.slip_proxy_01
        tactile[i, 9] = tr.contact_prob
        tactile[i, 10:16] = tr.contact_zones_6d
        tactile[i, 16] = tr.normal_force_proxy_01
        tactile[i, 17] = tr.slip_proxy_01

    action = np.zeros_like(state, dtype=np.float32)
    if n >= 2:
        action[:-1] = state[1:] - state[:-1]
        action[-1] = action[-2]

    return {
        "observation.state": state,
        "action": action,
        "action.canonical_delta": canonical_delta,
        "observation.tactile": tactile,
        "observation.camera_pose_world": cam_pose,
        "observation.hand_valid": hand_valid,
        "observation.proprio_valid": proprio_valid,
        "timestamp": timestamps,
    }


def _array_stats(arr: np.ndarray) -> dict[str, list[float]]:
    if arr.size == 0:
        return {"min": [], "max": [], "mean": [], "std": []}
    f = arr.astype(np.float64)
    return {
        "min": [round(float(v), 6) for v in np.min(f, axis=0)],
        "max": [round(float(v), 6) for v in np.max(f, axis=0)],
        "mean": [round(float(v), 6) for v in np.mean(f, axis=0)],
        "std": [round(float(v), 6) for v in np.std(f, axis=0)],
    }


def export_lerobot_v3(
    ir: EpisodeIR,
    retargeted: dict[str, EmbodimentTrajectory],
    src_video: Path,
    out_dir: Path,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    """Export LeRobotDataset v3.0 Parquet + synced sanitized MP4 + Parquet/JSONL metadata."""
    lr_root = out_dir / "lerobot_v3"
    data_dir = lr_root / "data" / "chunk-000"
    vid_dir = lr_root / "videos" / "observation.images.ego" / "chunk-000"
    meta_dir = lr_root / "meta"
    ep_meta_dir = meta_dir / "episodes" / "chunk-000"
    for d in (data_dir, vid_dir, meta_dir, ep_meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    arrays = _build_embodiment_agnostic_arrays(ir)
    n = len(ir.frames)

    seg_lookup = np.zeros(n, dtype=np.int64)
    subtask_lookup = np.zeros(n, dtype=np.int64)
    tasks_list = [{"task_index": 0, "task": ir.language.task_goal}]
    for seg in ir.segments:
        t_idx = len(tasks_list)
        tasks_list.append(
            {
                "task_index": t_idx,
                "task": seg.semantic_description or ir.language.task_goal,
            }
        )
        s0 = max(0, seg.start_frame)
        s1 = min(n - 1, seg.end_frame)
        if s1 >= s0:
            seg_lookup[s0 : s1 + 1] = t_idx
            subtask_lookup[s0 : s1 + 1] = int(seg.subtask_id)

    subtasks_rows: list[dict[str, Any]] = []
    for st in ir.language.subtasks:
        subtasks_rows.append(
            {
                "subtask_index": int(st.get("subtask_id", len(subtasks_rows))),
                "subtask": str(st.get("goal", ir.language.task_goal)),
                "start_frame": int(st.get("start_frame", 0)),
                "end_frame": int(st.get("end_frame", max(0, n - 1))),
                "start_s": float(st.get("start_s", 0.0)),
                "end_s": float(st.get("end_s", ir.video.duration_s)),
            }
        )
    if not subtasks_rows:
        subtasks_rows.append(
            {
                "subtask_index": 0,
                "subtask": ir.language.task_goal,
                "start_frame": 0,
                "end_frame": max(0, n - 1),
                "start_s": 0.0,
                "end_s": float(ir.video.duration_s),
            }
        )

    cols: dict[str, Any] = {
        "observation.state": pa.array(
            arrays["observation.state"].tolist(),
            type=pa.list_(pa.float32()),
        ),
        "action": pa.array(
            arrays["action"].tolist(),
            type=pa.list_(pa.float32()),
        ),
        "action.canonical_delta": pa.array(
            arrays["action.canonical_delta"].tolist(),
            type=pa.list_(pa.float32()),
        ),
        "observation.tactile": pa.array(
            arrays["observation.tactile"].tolist(),
            type=pa.list_(pa.float32()),
        ),
        "observation.camera_pose_world": pa.array(
            arrays["observation.camera_pose_world"].tolist(),
            type=pa.list_(pa.float32()),
        ),
        "observation.hand_valid": pa.array(
            arrays["observation.hand_valid"].tolist(),
            type=pa.list_(pa.int64()),
        ),
        "observation.proprio_valid": pa.array(
            arrays["observation.proprio_valid"].tolist(),
            type=pa.bool_(),
        ),
        "timestamp": pa.array(arrays["timestamp"], type=pa.float64()),
        "frame_index": pa.array(np.arange(n, dtype=np.int64), type=pa.int64()),
        "episode_index": pa.array(np.zeros(n, dtype=np.int64), type=pa.int64()),
        "index": pa.array(np.arange(n, dtype=np.int64), type=pa.int64()),
        "task_index": pa.array(seg_lookup, type=pa.int64()),
        "subtask_index": pa.array(subtask_lookup, type=pa.int64()),
    }

    retargeted_features: dict[str, Any] = {}
    for emb_name, emb_traj in retargeted.items():
        if emb_traj.qpos.ndim == 2 and emb_traj.qpos.shape[1] > 0:
            q_key = f"retargeted.{emb_name}.qpos"
            a_key = f"retargeted.{emb_name}.action"
            cols[q_key] = pa.array(
                emb_traj.qpos.astype(np.float32).tolist(),
                type=pa.list_(pa.float32()),
            )
            cols[a_key] = pa.array(
                emb_traj.actions.astype(np.float32).tolist(),
                type=pa.list_(pa.float32()),
            )
            retargeted_features[q_key] = {
                "dtype": "float32",
                "shape": [int(emb_traj.qpos.shape[1])],
                "names": list(emb_traj.joint_names),
                "units": "rad_or_m",
            }
            retargeted_features[a_key] = {
                "dtype": "float32",
                "shape": [int(emb_traj.actions.shape[1])],
                "names": list(emb_traj.joint_names),
                "units": "rad_or_m_target",
            }

    parquet_path = data_dir / "file-000.parquet"
    pq.write_table(pa.table(cols), parquet_path)

    mp4_path = vid_dir / "file-000.mp4"
    privacy_report = export_sanitized_mp4(
        src_video=src_video,
        dst_video=mp4_path,
        cfg=cfg.privacy,
        crf=cfg.export.video_crf,
        target_fps=(ir.video.processing_fps if cfg.runtime.normalize_fps_cfr else None),
    )

    features_meta: dict[str, Any] = {
        "observation.images.ego": {
            "dtype": "video",
            "shape": [ir.video.height, ir.video.width, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.fps": ir.video.processing_fps,
                "video.codec": "h264",
                "has_audio": False,
            },
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [30],
            "units": "meters_rot6d_unitless",
            "coordinate_frame": "world_z_up",
        },
        "action": {
            "dtype": "float32",
            "shape": [30],
            "units": "delta_state_per_step",
            "coordinate_frame": "world_z_up",
        },
        "action.canonical_delta": {
            "dtype": "float32",
            "shape": [16],
            "names": list(CANONICAL_DELTA_ACTION_NAMES),
            "units": "m_rad_m_binary",
            "coordinate_frame": "world_z_up",
        },
        "observation.tactile": {
            "dtype": "float32",
            "shape": [18],
            "source": ir.frames[0].tactile_right.source if ir.frames else "derived_rgb_proxy",
        },
        "observation.camera_pose_world": {
            "dtype": "float32",
            "shape": [7],
            "units": "xyz_m_quat_xyzw",
            "coordinate_frame": "world_z_up",
        },
        "observation.hand_valid": {"dtype": "int64", "shape": [2]},
        "observation.proprio_valid": {"dtype": "bool", "shape": [1]},
        "timestamp": {"dtype": "float64", "shape": [1], "units": "seconds"},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "subtask_index": {"dtype": "int64", "shape": [1]},
        **retargeted_features,
    }

    info_json = {
        "codebase_version": "v3.0",
        "robot_type": "embodiment_agnostic_human_ego",
        "coordinate_frame": "world_z_up_metric_meters",
        "depth_provenance": ir.scale_provenance.depth_provenance,
        "total_episodes": 1,
        "total_frames": n,
        "total_tasks": len(tasks_list),
        "total_subtasks": len(subtasks_rows),
        "fps": ir.video.processing_fps,
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "scale_provenance": ir.scale_provenance.model_dump(),
        "features": features_meta,
    }
    (meta_dir / "info.json").write_text(json.dumps(info_json, indent=2), encoding="utf-8")

    stats_json = {
        "observation.state": _array_stats(arrays["observation.state"]),
        "action": _array_stats(arrays["action"]),
        "action.canonical_delta": _array_stats(arrays["action.canonical_delta"]),
        "observation.tactile": _array_stats(arrays["observation.tactile"]),
        "observation.camera_pose_world": _array_stats(arrays["observation.camera_pose_world"]),
    }
    (meta_dir / "stats.json").write_text(json.dumps(stats_json, indent=2), encoding="utf-8")

    with open(meta_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row) + "\n" for row in tasks_list)
    pq.write_table(
        pa.table(
            {
                "task_index": pa.array([r["task_index"] for r in tasks_list], type=pa.int64()),
                "task": pa.array([str(r["task"]) for r in tasks_list], type=pa.string()),
            }
        ),
        meta_dir / "tasks.parquet",
    )

    pq.write_table(
        pa.table(
            {
                "subtask_index": pa.array(
                    [r["subtask_index"] for r in subtasks_rows], type=pa.int64()
                ),
                "subtask": pa.array([r["subtask"] for r in subtasks_rows], type=pa.string()),
                "start_frame": pa.array(
                    [r["start_frame"] for r in subtasks_rows], type=pa.int64()
                ),
                "end_frame": pa.array([r["end_frame"] for r in subtasks_rows], type=pa.int64()),
                "start_s": pa.array([r["start_s"] for r in subtasks_rows], type=pa.float64()),
                "end_s": pa.array([r["end_s"] for r in subtasks_rows], type=pa.float64()),
            }
        ),
        meta_dir / "subtasks.parquet",
    )

    ep_meta = {
        "episode_index": 0,
        "episode_id": ir.episode_id,
        "tasks": [ir.language.task_goal],
        "length": n,
        "data/chunk_index": 0,
        "data/file_index": 0,
        "dataset_from_index": 0,
        "dataset_to_index": n,
        "videos/observation.images.ego/chunk_index": 0,
        "videos/observation.images.ego/file_index": 0,
        "videos/observation.images.ego/from_timestamp": 0.0,
        "videos/observation.images.ego/to_timestamp": float(ir.video.duration_s),
        "robot_readiness_score": ir.quality.robot_readiness_score,
        "quality_verdict": ir.quality.verdict,
        "environment_tag": ir.language.environment_tag,
        "domain_taxonomy": ir.language.domain_taxonomy,
        "depth_provenance": ir.scale_provenance.depth_provenance,
    }
    (meta_dir / "episodes.jsonl").write_text(json.dumps(ep_meta) + "\n", encoding="utf-8")
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([0], type=pa.int64()),
                "episode_id": pa.array([ir.episode_id], type=pa.string()),
                "tasks": pa.array([[ir.language.task_goal]], type=pa.list_(pa.string())),
                "length": pa.array([n], type=pa.int64()),
                "data/chunk_index": pa.array([0], type=pa.int64()),
                "data/file_index": pa.array([0], type=pa.int64()),
                "dataset_from_index": pa.array([0], type=pa.int64()),
                "dataset_to_index": pa.array([n], type=pa.int64()),
                "videos/observation.images.ego/chunk_index": pa.array([0], type=pa.int64()),
                "videos/observation.images.ego/file_index": pa.array([0], type=pa.int64()),
                "videos/observation.images.ego/from_timestamp": pa.array([0.0], type=pa.float64()),
                "videos/observation.images.ego/to_timestamp": pa.array(
                    [float(ir.video.duration_s)], type=pa.float64()
                ),
                "robot_readiness_score": pa.array(
                    [float(ir.quality.robot_readiness_score)], type=pa.float32()
                ),
                "quality_verdict": pa.array([str(ir.quality.verdict)], type=pa.string()),
                "environment_tag": pa.array([str(ir.language.environment_tag)], type=pa.string()),
                "domain_taxonomy": pa.array([str(ir.language.domain_taxonomy)], type=pa.string()),
                "depth_provenance": pa.array(
                    [str(ir.scale_provenance.depth_provenance)], type=pa.string()
                ),
            }
        ),
        ep_meta_dir / "file-000.parquet",
    )

    if cfg.export.include_groot_modality_json:
        modality_json = {
            "state": {
                "left_wrist_pos": {"start": 0, "end": 3},
                "left_wrist_rot6d": {"start": 3, "end": 9},
                "left_finger_curls": {"start": 9, "end": 14},
                "left_gripper": {"start": 14, "end": 15},
                "right_wrist_pos": {"start": 15, "end": 18},
                "right_wrist_rot6d": {"start": 18, "end": 24},
                "right_finger_curls": {"start": 24, "end": 29},
                "right_gripper": {"start": 29, "end": 30},
            },
            "action": {
                "left_wrist_delta": {"start": 0, "end": 15, "absolute": False},
                "right_wrist_delta": {"start": 15, "end": 30, "absolute": False},
                "canonical_bimanual_se3_delta": {
                    "original_key": "action.canonical_delta",
                    "start": 0,
                    "end": 16,
                    "absolute": False,
                },
            },
            "tactile": {
                "left_hand_9d": {
                    "start": 0,
                    "end": 9,
                    "source": ir.frames[0].tactile_left.source if ir.frames else "derived_rgb_proxy",
                },
                "right_hand_9d": {
                    "start": 9,
                    "end": 18,
                    "source": ir.frames[0].tactile_right.source if ir.frames else "derived_rgb_proxy",
                },
            },
            "video": {"ego_view": {"original_key": "observation.images.ego"}},
            "annotation": {
                "human.action.task_description": {"original_key": "task_index"},
                "human.action.subtask_description": {"original_key": "subtask_index"},
            },
        }
        (meta_dir / "modality.json").write_text(json.dumps(modality_json, indent=2), encoding="utf-8")

    return {
        "root": str(lr_root),
        "parquet": str(parquet_path),
        "video": str(mp4_path),
        "tasks_parquet": str(meta_dir / "tasks.parquet"),
        "subtasks_parquet": str(meta_dir / "subtasks.parquet"),
        "episodes_parquet": str(ep_meta_dir / "file-000.parquet"),
        "frames": n,
        "privacy": privacy_report,
    }


def export_rlds_oxe(
    ir: EpisodeIR,
    out_dir: Path,
) -> dict[str, Any]:
    """Export Open X-Embodiment / RLDS-compatible episode tensors and schema descriptors."""
    rlds_dir = out_dir / "rlds"
    rlds_dir.mkdir(parents=True, exist_ok=True)
    arrays = _build_embodiment_agnostic_arrays(ir)
    n = len(ir.frames)

    is_first = np.zeros(n, dtype=bool)
    is_last = np.zeros(n, dtype=bool)
    is_terminal = np.zeros(n, dtype=bool)
    if n > 0:
        is_first[0] = True
        is_last[-1] = True
        is_terminal[-1] = True

    step_instructions = [ir.language.task_goal] * n
    for seg in ir.segments:
        if seg.semantic_description:
            for fi in range(max(0, seg.start_frame), min(n, seg.end_frame + 1)):
                step_instructions[fi] = seg.semantic_description

    npz_path = rlds_dir / "episode_000000.npz"
    np.savez_compressed(
        npz_path,
        observation_state=arrays["observation.state"],
        observation_tactile=arrays["observation.tactile"],
        observation_camera_pose=arrays["observation.camera_pose_world"],
        observation_hand_valid=arrays["observation.hand_valid"],
        observation_proprio_valid=arrays["observation.proprio_valid"],
        action=arrays["action"],
        action_canonical_delta=arrays["action.canonical_delta"],
        timestamp=arrays["timestamp"],
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
    )

    manifest = {
        "format": "rlds_oxe_compatible",
        "episode_id": ir.episode_id,
        "num_steps": n,
        "fps": ir.video.processing_fps,
        "task_goal": ir.language.task_goal,
        "environment_tag": ir.language.environment_tag,
        "domain_taxonomy": ir.language.domain_taxonomy,
        "tactile_source": ir.frames[0].tactile_right.source if ir.frames else "derived_rgb_proxy",
        "scale_provenance": ir.scale_provenance.model_dump(),
        "files": {"episode_npz": npz_path.name, "instructions_json": "step_instructions.json"},
    }
    (rlds_dir / "dataset_info.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (rlds_dir / "step_instructions.json").write_text(json.dumps(step_instructions), encoding="utf-8")
    return {"root": str(rlds_dir), "npz": str(npz_path), "steps": n}


def export_canonical_hdf5(
    ir: EpisodeIR,
    retargeted: dict[str, EmbodimentTrajectory],
    out_dir: Path,
) -> dict[str, Any]:
    """Export gzip-compressed canonical HDF5 containing IR world streams + all retargeted embodiments."""
    h5_path = out_dir / "episode_canonical.h5"
    arrays = _build_embodiment_agnostic_arrays(ir)

    with h5py.File(h5_path, "w") as hf:
        hf.attrs["schema_version"] = ir.schema_version
        hf.attrs["episode_id"] = ir.episode_id
        hf.attrs["total_frames"] = len(ir.frames)
        hf.attrs["is_full_video_processed"] = bool(ir.video.is_full_video_processed)
        hf.attrs["fps"] = float(ir.video.processing_fps)
        hf.attrs["scale_source"] = ir.scale_provenance.scale_source
        hf.attrs["depth_provenance"] = ir.scale_provenance.depth_provenance
        hf.attrs["scale_uncertainty_pct"] = float(ir.scale_provenance.scale_uncertainty_pct)
        hf.attrs["is_calibrated_metric"] = bool(ir.scale_provenance.is_calibrated_metric)
        hf.attrs["tactile_source"] = (
            ir.frames[0].tactile_right.source if ir.frames else "derived_rgb_proxy"
        )
        hf.attrs["robot_readiness_score"] = float(ir.quality.robot_readiness_score)
        hf.attrs["task_goal"] = ir.language.task_goal

        g_world = hf.create_group("world")
        for key, arr in arrays.items():
            clean_key = key.replace(".", "_")
            g_world.create_dataset(clean_key, data=arr, compression="gzip", compression_opts=4)

        g_ret = hf.create_group("retargeted")
        for emb_name, traj in retargeted.items():
            ge = g_ret.create_group(emb_name)
            ge.attrs["embodiment_class"] = traj.embodiment_class
            ge.attrs["joint_names_json"] = json.dumps(traj.joint_names)
            ge.attrs["metrics_json"] = json.dumps(traj.metrics)
            ge.create_dataset("qpos", data=traj.qpos, compression="gzip", compression_opts=4)
            ge.create_dataset("actions", data=traj.actions, compression="gzip", compression_opts=4)
            ge.create_dataset(
                "fk_pos_error_mm",
                data=traj.fk_pos_error_mm,
                compression="gzip",
                compression_opts=4,
            )

    return {
        "hdf5": str(h5_path),
        "size_bytes": h5_path.stat().st_size,
        "embodiments": sorted(retargeted.keys()),
    }


def export_parquet_and_json_sidecar(
    ir: EpisodeIR,
    out_dir: Path,
) -> dict[str, Any]:
    """Export canonical frame-level Parquet table + full Pydantic-validated JSON IR sidecar."""
    pq_dir = out_dir / "parquet_json"
    pq_dir.mkdir(parents=True, exist_ok=True)
    arrays = _build_embodiment_agnostic_arrays(ir)
    table = pa.table(
        {
            "frame_idx": pa.array([f.frame_idx for f in ir.frames], type=pa.int64()),
            "timestamp_s": pa.array(arrays["timestamp"], type=pa.float64()),
            "state_30d": pa.array(
                arrays["observation.state"].tolist(), type=pa.list_(pa.float32())
            ),
            "action_30d": pa.array(arrays["action"].tolist(), type=pa.list_(pa.float32())),
            "canonical_delta_16d": pa.array(
                arrays["action.canonical_delta"].tolist(), type=pa.list_(pa.float32())
            ),
            "tactile_18d": pa.array(
                arrays["observation.tactile"].tolist(), type=pa.list_(pa.float32())
            ),
            "camera_pose_7d": pa.array(
                arrays["observation.camera_pose_world"].tolist(),
                type=pa.list_(pa.float32()),
            ),
            "hand_valid_2d": pa.array(
                arrays["observation.hand_valid"].tolist(), type=pa.list_(pa.int64())
            ),
            "proprio_valid": pa.array(
                arrays["observation.proprio_valid"].tolist(), type=pa.bool_()
            ),
        }
    )
    pq_path = pq_dir / "frames.parquet"
    pq.write_table(table, pq_path)

    ir_json_path = pq_dir / "episode_ir.json"
    ir_json_path.write_text(ir.model_dump_json(), encoding="utf-8")
    return {"parquet": str(pq_path), "ir_json": str(ir_json_path)}


def _mcap_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<I", len(b)) + b


def _mcap_record(opcode: int, body: bytes) -> bytes:
    return struct.pack("<BQ", opcode, len(body)) + body


def export_mcap_ros2(ir: EpisodeIR, out_dir: Path) -> dict[str, Any]:
    """Export standard MCAP v0 binary container for ROS 2 / Foxglove replay."""
    mcap_path = out_dir / f"{ir.episode_id}.mcap"
    buf = bytearray()
    buf.extend(_MCAP_MAGIC)

    # 0x01 Header: profile + library
    buf.extend(_mcap_record(0x01, _mcap_str("ros2") + _mcap_str("egoannotate-v4")))

    # 0x03 Schema (id=1, JSON schema)
    schema_bytes = json.dumps(
        {
            "type": "object",
            "properties": {
                "frame_idx": {"type": "integer"},
                "camera_pose_world": {"type": "array"},
                "canonical_delta_action": {"type": "array"},
                "contact_event_tags": {"type": "array"},
                "proprio_valid": {"type": "boolean"},
            },
        }
    ).encode("utf-8")
    schema_body = (
        struct.pack("<H", 1)
        + _mcap_str("egoannotate.FrameTelemetry")
        + _mcap_str("jsonschema")
        + struct.pack("<I", len(schema_bytes))
        + schema_bytes
    )
    buf.extend(_mcap_record(0x03, schema_body))

    # 0x04 Channel (id=1, schema_id=1, topic=/ego/telemetry)
    chan_body = (
        struct.pack("<HH", 1, 1)
        + _mcap_str("/ego/telemetry")
        + _mcap_str("json")
        + struct.pack("<I", 0)
    )
    buf.extend(_mcap_record(0x04, chan_body))

    # 0x05 Message records for all frames
    for seq, fr in enumerate(ir.frames):
        ts_ns = round(float(fr.timestamp_s) * 1e9)
        payload = json.dumps(
            {
                "frame_idx": fr.frame_idx,
                "camera_pose_world": fr.camera_pose.pos_world_m + fr.camera_pose.quat_xyzw_world,
                "left_wrist_world_m": fr.left_hand.wrist_pos_world_m,
                "right_wrist_world_m": fr.right_hand.wrist_pos_world_m,
                "canonical_delta_action": fr.canonical_delta_action,
                "contact_event_tags": fr.contact_event_tags,
                "proprio_valid": fr.proprio_valid,
            }
        ).encode("utf-8")
        msg_body = struct.pack("<HIQQ", 1, seq, ts_ns, ts_ns) + payload
        buf.extend(_mcap_record(0x05, msg_body))

    # 0x0F DataEnd (data_section_crc=0)
    buf.extend(_mcap_record(0x0F, struct.pack("<I", 0)))
    # 0x02 Footer (summary_start=0, summary_offset_start=0, summary_crc=0)
    buf.extend(_mcap_record(0x02, struct.pack("<QQI", 0, 0, 0)))
    buf.extend(_MCAP_MAGIC)

    mcap_path.write_bytes(bytes(buf))
    return {"mcap": str(mcap_path), "messages": len(ir.frames), "size_bytes": mcap_path.stat().st_size}


def export_dataset_card(ir: EpisodeIR, out_dir: Path) -> dict[str, Any]:
    """Generate commercial dataset_card.json + Apache-2.0/MIT license audit manifest."""
    for mod_name in (
        "src.egoannotate.geometry_world",
        "src.egoannotate.perception_hands",
        "src.egoannotate.perception_hands_rtmw",
        "src.egoannotate.perception_objects",
        "src.egoannotate.semantics",
        "src.egoannotate.gpu_backends_v4",
    ):
        importlib.import_module(mod_name)

    card = {
        "schema_version": ir.schema_version,
        "episode_id": ir.episode_id,
        "created_utc": ir.created_utc,
        "task_goal": ir.language.task_goal,
        "environment_tag": ir.language.environment_tag,
        "domain_taxonomy": ir.language.domain_taxonomy,
        "operator_ehs_metadata": ir.language.operator_ehs_metadata,
        "discovered_objects": ir.language.discovered_objects,
        "video": {
            "width": ir.video.width,
            "height": ir.video.height,
            "source_fps": ir.video.source_fps,
            "processing_fps": ir.video.processing_fps,
            "processed_frames": ir.video.processed_frames,
            "duration_s": ir.video.duration_s,
            "is_full_video_processed": ir.video.is_full_video_processed,
        },
        "depth_provenance": ir.scale_provenance.depth_provenance,
        "scale_provenance": ir.scale_provenance.model_dump(),
        "quality": {
            "robot_readiness_score": ir.quality.robot_readiness_score,
            "verdict": ir.quality.verdict,
            "flags": ir.quality.flags,
        },
        "privacy_audit": ir.privacy_audit,
        "retargeting_summary": ir.retargeting_summary,
        "commercial_license_manifest": REGISTRY.audit_table(),
    }
    card_path = out_dir / "dataset_card.json"
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    return {"dataset_card": str(card_path)}


def export_all_formats(
    ir: EpisodeIR,
    retargeted: dict[str, EmbodimentTrajectory],
    src_video: Path,
    out_dir: Path,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    """Run all configured exporters from the single embodiment-agnostic IR."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    if "lerobot_v3" in cfg.export.formats:
        lr = export_lerobot_v3(ir, retargeted, src_video, out_dir, cfg)
        summary["lerobot_v3"] = lr
        ir.privacy_audit = lr["privacy"]
    if "rlds" in cfg.export.formats:
        summary["rlds"] = export_rlds_oxe(ir, out_dir)
    if "hdf5" in cfg.export.formats:
        summary["hdf5"] = export_canonical_hdf5(ir, retargeted, out_dir)
    if "parquet_json" in cfg.export.formats:
        summary["parquet_json"] = export_parquet_and_json_sidecar(ir, out_dir)
    if "mcap" in cfg.export.formats:
        summary["mcap"] = export_mcap_ros2(ir, out_dir)
    if cfg.export.generate_dataset_card:
        summary["dataset_card"] = export_dataset_card(ir, out_dir)
    return summary
