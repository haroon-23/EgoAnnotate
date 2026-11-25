"""Multi-format dataset exporters: LeRobot v3 (+GR00T modality.json), RLDS/OXE, Canonical HDF5 & Parquet+JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.egoannotate.config import PipelineConfig
from src.egoannotate.privacy_v3 import export_sanitized_mp4
from src.egoannotate.retargeting_v3 import EmbodimentTrajectory
from src.egoannotate.schemas import EpisodeIR


def _build_embodiment_agnostic_arrays(ir: EpisodeIR) -> dict[str, np.ndarray]:
    """Pack embodiment-agnostic world-frame 30D state, 30D delta action, 18D tactile & 2D hand-valid tensors."""
    n = len(ir.frames)
    # 30D state: [left_wrist_xyz(3), left_rot6d(6), left_curls(5), left_ap(1),
    #             right_wrist_xyz(3), right_rot6d(6), right_curls(5), right_ap(1)]
    state = np.zeros((n, 30), dtype=np.float32)
    tactile = np.zeros((n, 18), dtype=np.float32)
    cam_pose = np.zeros((n, 7), dtype=np.float32)
    hand_valid = np.zeros((n, 2), dtype=np.int64)
    timestamps = np.zeros(n, dtype=np.float32)

    for i, fr in enumerate(ir.frames):
        timestamps[i] = float(fr.timestamp_s)
        cam_pose[i, :3] = fr.camera_pose.pos_world_m
        cam_pose[i, 3:7] = fr.camera_pose.quat_xyzw_world

        lh, rh = fr.left_hand, fr.right_hand
        hand_valid[i, 0] = 1 if lh.detected else 0
        hand_valid[i, 1] = 1 if rh.detected else 0

        state[i, 0:3] = lh.wrist_pos_world_m
        state[i, 3:9] = lh.wrist_rot6d_world
        state[i, 9:14] = lh.finger_curls_5d
        state[i, 14] = lh.gripper_aperture_m

        state[i, 15:18] = rh.wrist_pos_world_m
        state[i, 18:24] = rh.wrist_rot6d_world
        state[i, 24:29] = rh.finger_curls_5d
        state[i, 29] = rh.gripper_aperture_m

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
        "observation.tactile": tactile,
        "observation.camera_pose_world": cam_pose,
        "observation.hand_valid": hand_valid,
        "timestamp": timestamps,
    }


def _array_stats(arr: np.ndarray) -> dict[str, list[float]]:
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
    """Export LeRobot v3 chunked Parquet + packet-synced sanitized MP4 + meta/info, stats, tasks & GR00T modality.json."""
    lr_root = out_dir / "lerobot_v3"
    data_dir = lr_root / "data" / "chunk-000"
    vid_dir = lr_root / "videos" / "observation.images.ego" / "chunk-000"
    meta_dir = lr_root / "meta"
    for d in (data_dir, vid_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    arrays = _build_embodiment_agnostic_arrays(ir)
    n = len(ir.frames)

    # Map every frame index to its active atomic segment's task index
    seg_lookup = np.zeros(n, dtype=np.int64)
    tasks_list = [{"task_index": 0, "task": ir.language.task_goal}]
    for seg in ir.segments:
        t_idx = len(tasks_list)
        tasks_list.append({"task_index": t_idx, "task": seg.semantic_description or ir.language.task_goal})
        s0 = max(0, seg.start_frame)
        s1 = min(n - 1, seg.end_frame)
        seg_lookup[s0 : s1 + 1] = t_idx

    cols: dict[str, Any] = {
        "observation.state": pa.array(arrays["observation.state"].tolist(), type=pa.list_(pa.float32())),
        "action": pa.array(arrays["action"].tolist(), type=pa.list_(pa.float32())),
        "observation.tactile": pa.array(arrays["observation.tactile"].tolist(), type=pa.list_(pa.float32())),
        "observation.camera_pose_world": pa.array(
            arrays["observation.camera_pose_world"].tolist(), type=pa.list_(pa.float32())
        ),
        "observation.hand_valid": pa.array(arrays["observation.hand_valid"].tolist(), type=pa.list_(pa.int64())),
        "timestamp": pa.array(arrays["timestamp"], type=pa.float32()),
        "frame_index": pa.array(np.arange(n, dtype=np.int64), type=pa.int64()),
        "episode_index": pa.array(np.zeros(n, dtype=np.int64), type=pa.int64()),
        "index": pa.array(np.arange(n, dtype=np.int64), type=pa.int64()),
        "task_index": pa.array(seg_lookup, type=pa.int64()),
    }

    # Include primary retargeted robot joint states & actions if available
    primary_emb = next(iter(retargeted.values()), None)
    if primary_emb is not None and primary_emb.qpos.shape[1] > 0:
        cols[f"retargeted.{primary_emb.name}.qpos"] = pa.array(
            primary_emb.qpos.astype(np.float32).tolist(), type=pa.list_(pa.float32())
        )
        cols[f"retargeted.{primary_emb.name}.action"] = pa.array(
            primary_emb.actions.astype(np.float32).tolist(), type=pa.list_(pa.float32())
        )

    parquet_path = data_dir / "file-000.parquet"
    pq.write_table(pa.table(cols), parquet_path)

    # Export privacy-sanitized packet-synced MP4 across all frames 0..N-1
    mp4_path = vid_dir / "file-000.mp4"
    privacy_report = export_sanitized_mp4(
        src_video=src_video,
        dst_video=mp4_path,
        cfg=cfg.privacy,
        crf=cfg.export.video_crf,
    )

    features_meta: dict[str, Any] = {
        "observation.images.ego": {
            "dtype": "video",
            "shape": [ir.video.height, ir.video.width, 3],
            "names": ["height", "width", "channel"],
            "info": {"video.fps": ir.video.processing_fps, "video.codec": "h264", "has_audio": False},
        },
        "observation.state": {"dtype": "float32", "shape": [30]},
        "action": {"dtype": "float32", "shape": [30]},
        "observation.tactile": {
            "dtype": "float32",
            "shape": [18],
            "source": ir.frames[0].tactile_right.source if ir.frames else "derived_rgb_proxy",
        },
        "observation.camera_pose_world": {"dtype": "float32", "shape": [7]},
        "observation.hand_valid": {"dtype": "int64", "shape": [2]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }

    info_json = {
        "codebase_version": "v3.0",
        "robot_type": "embodiment_agnostic_human_ego",
        "total_episodes": 1,
        "total_frames": n,
        "total_tasks": len(tasks_list),
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
        "observation.tactile": _array_stats(arrays["observation.tactile"]),
        "observation.camera_pose_world": _array_stats(arrays["observation.camera_pose_world"]),
    }
    (meta_dir / "stats.json").write_text(json.dumps(stats_json, indent=2), encoding="utf-8")

    with open(meta_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row) + "\n" for row in tasks_list)

    ep_meta = {
        "episode_index": 0,
        "episode_id": ir.episode_id,
        "tasks": [ir.language.task_goal],
        "length": n,
        "robot_readiness_score": ir.quality.robot_readiness_score,
        "quality_verdict": ir.quality.verdict,
    }
    (meta_dir / "episodes.jsonl").write_text(json.dumps(ep_meta) + "\n", encoding="utf-8")

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
            },
            "tactile": {
                "left_hand_9d": {"start": 0, "end": 9, "source": ir.frames[0].tactile_left.source if ir.frames else "derived_rgb_proxy"},
                "right_hand_9d": {"start": 9, "end": 18, "source": ir.frames[0].tactile_right.source if ir.frames else "derived_rgb_proxy"},
            },
            "video": {"ego_view": {"original_key": "observation.images.ego"}},
            "annotation": {"human.action.task_description": {"original_key": "task_index"}},
        }
        (meta_dir / "modality.json").write_text(json.dumps(modality_json, indent=2), encoding="utf-8")

    return {
        "root": str(lr_root),
        "parquet": str(parquet_path),
        "video": str(mp4_path),
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
        action=arrays["action"],
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
        "tactile_source": ir.frames[0].tactile_right.source if ir.frames else "derived_rgb_proxy",
        "scale_provenance": ir.scale_provenance.model_dump(),
        "tfds_builder_note": "Standard OXE step structure; directly convertible to TFRecord when tensorflow_datasets is installed.",
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
        hf.attrs["scale_uncertainty_pct"] = float(ir.scale_provenance.scale_uncertainty_pct)
        hf.attrs["is_calibrated_metric"] = bool(ir.scale_provenance.is_calibrated_metric)
        hf.attrs["tactile_source"] = ir.frames[0].tactile_right.source if ir.frames else "derived_rgb_proxy"
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
            ge.create_dataset("fk_pos_error_mm", data=traj.fk_pos_error_mm, compression="gzip", compression_opts=4)

    return {"hdf5": str(h5_path), "size_bytes": h5_path.stat().st_size, "embodiments": sorted(retargeted.keys())}


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
            "timestamp_s": pa.array(arrays["timestamp"], type=pa.float32()),
            "state_30d": pa.array(arrays["observation.state"].tolist(), type=pa.list_(pa.float32())),
            "action_30d": pa.array(arrays["action"].tolist(), type=pa.list_(pa.float32())),
            "tactile_18d": pa.array(arrays["observation.tactile"].tolist(), type=pa.list_(pa.float32())),
            "camera_pose_7d": pa.array(arrays["observation.camera_pose_world"].tolist(), type=pa.list_(pa.float32())),
            "hand_valid_2d": pa.array(arrays["observation.hand_valid"].tolist(), type=pa.list_(pa.int64())),
        }
    )
    pq_path = pq_dir / "frames.parquet"
    pq.write_table(table, pq_path)

    ir_json_path = pq_dir / "episode_ir.json"
    ir_json_path.write_text(ir.model_dump_json(), encoding="utf-8")
    return {"parquet": str(pq_path), "ir_json": str(ir_json_path)}


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
    return summary
