"""Atomic multi-episode sharded LeRobotDataset v3.0 packer (Parquet + MP4 shards + offset tables)."""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.egoannotate.batch_orchestrator import BatchStateLedger
from src.egoannotate.exporters_v3 import _array_stats
from src.egoannotate.policy_gate import validate_lerobot_v3_dataset


def _concat_mp4_files(mp4_paths: list[Path], dst_mp4: Path) -> None:
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    if len(mp4_paths) == 1:
        shutil.copy2(mp4_paths[0], dst_mp4)
        return
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg binary required to concatenate LeRobot v3.0 video shards.")
    list_file = dst_mp4.with_suffix(".concat.txt")
    list_file.write_text(
        "\n".join(f"file '{p.resolve()}'" for p in mp4_paths) + "\n",
        encoding="utf-8",
    )
    try:
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            "-an",
            str(dst_mp4),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg shard concat failed: {res.stderr.strip()}")
    finally:
        list_file.unlink(missing_ok=True)


def pack_sharded_lerobot_v3(
    episode_dirs: list[Path],
    output_dir: Path,
    *,
    max_frames_per_shard: int = 50000,
    min_quality_score: float = 0.0,
    run_validation_gate: bool = True,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    """Atomically pack QA-verified episode runs into a multi-episode sharded LeRobot v3.0 dataset."""
    out_final = Path(output_dir)
    staging = out_final.parent / f".{out_final.name}.staging_{uuid.uuid4().hex[:8]}"
    data_dir = staging / "data" / "chunk-000"
    vid_dir = staging / "videos" / "observation.images.ego" / "chunk-000"
    meta_dir = staging / "meta"
    ep_meta_dir = meta_dir / "episodes" / "chunk-000"
    for d in (data_dir, vid_dir, meta_dir, ep_meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    global_tasks: dict[str, int] = {}
    global_subtasks: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    accepted_manifests: list[str] = []

    shard_idx = 0
    shard_tables: list[pa.Table] = []
    shard_videos: list[Path] = []
    shard_frames = 0
    shard_video_cursor_s = 0.0
    shard_video_spec: tuple[int, int, int] | None = None
    global_frame_cursor = 0
    base_info: dict[str, Any] | None = None

    stats_acc: dict[str, list[np.ndarray]] = {
        "observation.state": [],
        "action": [],
        "action.canonical_delta": [],
        "observation.tactile": [],
        "observation.camera_pose_world": [],
    }
    per_episode_stats: list[dict[str, Any]] = []

    def _flush_shard() -> None:
        nonlocal shard_idx, shard_tables, shard_videos, shard_frames, shard_video_cursor_s, shard_video_spec
        if not shard_tables:
            return
        combined = pa.concat_tables(shard_tables, promote_options="default")
        pq.write_table(combined, data_dir / f"file-{shard_idx:03d}.parquet")
        _concat_mp4_files(shard_videos, vid_dir / f"file-{shard_idx:03d}.mp4")
        shard_idx += 1
        shard_tables = []
        shard_videos = []
        shard_frames = 0
        shard_video_cursor_s = 0.0
        shard_video_spec = None

    try:
        for ep_dir in sorted(episode_dirs):
            rep_path = ep_dir / "run_report.json"
            lr_dir = ep_dir / "lerobot_v3"
            pq_src = lr_dir / "data" / "chunk-000" / "file-000.parquet"
            mp4_src = lr_dir / "videos" / "observation.images.ego" / "chunk-000" / "file-000.mp4"
            info_src = lr_dir / "meta" / "info.json"
            if not (rep_path.exists() and pq_src.exists() and mp4_src.exists() and info_src.exists()):
                skipped.append({"dir": str(ep_dir), "reason": "incomplete_episode_artifacts"})
                continue

            rep = json.loads(rep_path.read_text(encoding="utf-8"))
            q_score = float(rep.get("quality", {}).get("robot_readiness_score", 0.0))
            verdict = str(rep.get("quality", {}).get("verdict", "QUARANTINE"))
            priv = rep.get("privacy_audit", {})
            if not priv.get("privacy_pass", False) or int(priv.get("verified_unblurred_faces", 0)) > 0:
                skipped.append({"dir": str(ep_dir), "reason": "privacy_gate_failed"})
                continue
            if verdict == "QUARANTINE" or q_score < float(min_quality_score):
                skipped.append({"dir": str(ep_dir), "reason": f"quality_gate_failed:{verdict}:{q_score}"})
                continue

            ep_info = json.loads(info_src.read_text(encoding="utf-8"))
            if base_info is None:
                base_info = ep_info
            tbl = pq.read_table(pq_src)
            ep_len = tbl.num_rows
            if ep_len == 0:
                skipped.append({"dir": str(ep_dir), "reason": "empty_episode"})
                continue

            cur_spec = (
                int(rep.get("video", {}).get("width", 0)),
                int(rep.get("video", {}).get("height", 0)),
                round(float(rep.get("video", {}).get("processing_fps", 30.0))),
            )
            if shard_frames > 0 and (
                (shard_frames + ep_len) > max_frames_per_shard
                or (shard_video_spec is not None and cur_spec != shard_video_spec)
            ):
                _flush_shard()
            if shard_video_spec is None:
                shard_video_spec = cur_spec

            ep_idx = len(episode_rows)
            tasks_pq_src = lr_dir / "meta" / "tasks.parquet"
            local_task_map: dict[int, str] = {}
            if tasks_pq_src.exists():
                tt = pq.read_table(tasks_pq_src)
                for tid, ttxt in zip(tt["task_index"].to_pylist(), tt["task"].to_pylist()):
                    local_task_map[int(tid)] = str(ttxt)

            remapped_tids: list[int] = []
            for old_tid in tbl["task_index"].to_pylist():
                t_str = local_task_map.get(int(old_tid), str(rep.get("language", {}).get("task_goal", "manipulate")))
                if t_str not in global_tasks:
                    global_tasks[t_str] = len(global_tasks)
                remapped_tids.append(global_tasks[t_str])

            subtasks_pq_src = lr_dir / "meta" / "subtasks.parquet"
            subtask_offset = len(global_subtasks)
            if subtasks_pq_src.exists():
                stt = pq.read_table(subtasks_pq_src)
                for row_i in range(stt.num_rows):
                    global_subtasks.append(
                        {
                            "subtask_index": len(global_subtasks),
                            "episode_index": ep_idx,
                            "subtask": str(stt["subtask"][row_i].as_py()),
                            "start_frame": int(stt["start_frame"][row_i].as_py()),
                            "end_frame": int(stt["end_frame"][row_i].as_py()),
                            "start_s": float(stt["start_s"][row_i].as_py()),
                            "end_s": float(stt["end_s"][row_i].as_py()),
                        }
                    )

            new_cols: dict[str, Any] = {}
            for cname in tbl.column_names:
                if cname == "episode_index":
                    new_cols[cname] = pa.array(np.full(ep_len, ep_idx, dtype=np.int64), type=pa.int64())
                elif cname == "index":
                    new_cols[cname] = pa.array(
                        np.arange(global_frame_cursor, global_frame_cursor + ep_len, dtype=np.int64),
                        type=pa.int64(),
                    )
                elif cname == "task_index":
                    new_cols[cname] = pa.array(remapped_tids, type=pa.int64())
                elif cname == "subtask_index":
                    orig_st = np.asarray(tbl[cname].to_pylist(), dtype=np.int64)
                    new_cols[cname] = pa.array(orig_st + subtask_offset, type=pa.int64())
                else:
                    new_cols[cname] = tbl[cname]

            ep_stat_entry: dict[str, Any] = {"episode_index": ep_idx, "episode_id": rep["episode_id"]}
            for skey, acc_list in stats_acc.items():
                if skey in tbl.column_names:
                    arr = np.asarray(tbl[skey].to_pylist(), dtype=np.float32)
                    acc_list.append(arr)
                    ep_stat_entry[skey] = _array_stats(arr)
            per_episode_stats.append(ep_stat_entry)

            dur_s = float(rep["video"]["duration_s"])
            ep_goal = str(rep.get("language", {}).get("task_goal", "Manipulate object"))
            if ep_goal not in global_tasks:
                global_tasks[ep_goal] = len(global_tasks)

            episode_rows.append(
                {
                    "episode_index": ep_idx,
                    "episode_id": str(rep["episode_id"]),
                    "tasks": [ep_goal],
                    "length": ep_len,
                    "data/chunk_index": 0,
                    "data/file_index": shard_idx,
                    "dataset_from_index": global_frame_cursor,
                    "dataset_to_index": global_frame_cursor + ep_len,
                    "videos/observation.images.ego/chunk_index": 0,
                    "videos/observation.images.ego/file_index": shard_idx,
                    "videos/observation.images.ego/from_timestamp": round(shard_video_cursor_s, 6),
                    "videos/observation.images.ego/to_timestamp": round(shard_video_cursor_s + dur_s, 6),
                    "robot_readiness_score": q_score,
                    "quality_verdict": verdict,
                    "environment_tag": str(rep.get("language", {}).get("environment_tag", "general_workspace")),
                    "domain_taxonomy": str(rep.get("language", {}).get("domain_taxonomy", "general_manipulation")),
                    "depth_provenance": str(rep.get("depth_provenance", "estimated_metric_monocular")),
                }
            )
            vid_sha = str(rep.get("video", {}).get("video_sha256", ""))
            if vid_sha:
                accepted_manifests.append(vid_sha)

            shard_tables.append(pa.table(new_cols))
            shard_videos.append(mp4_src)
            shard_frames += ep_len
            shard_video_cursor_s += dur_s
            global_frame_cursor += ep_len

        _flush_shard()

        if not episode_rows or base_info is None:
            raise RuntimeError(f"No valid episodes passed QA/privacy gates to shard (skipped={len(skipped)}).")

        tasks_sorted = sorted(global_tasks.items(), key=lambda kv: kv[1])
        pq.write_table(
            pa.table(
                {
                    "task_index": pa.array([idx for _, idx in tasks_sorted], type=pa.int64()),
                    "task": pa.array([txt for txt, _ in tasks_sorted], type=pa.string()),
                }
            ),
            meta_dir / "tasks.parquet",
        )
        with (meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as fh:
            for txt, idx in tasks_sorted:
                fh.write(json.dumps({"task_index": idx, "task": txt}) + "\n")

        if not global_subtasks:
            global_subtasks.append(
                {
                    "subtask_index": 0,
                    "episode_index": 0,
                    "subtask": tasks_sorted[0][0],
                    "start_frame": 0,
                    "end_frame": max(0, global_frame_cursor - 1),
                    "start_s": 0.0,
                    "end_s": float(episode_rows[0]["videos/observation.images.ego/to_timestamp"]),
                }
            )
        pq.write_table(
            pa.table(
                {
                    "subtask_index": pa.array([r["subtask_index"] for r in global_subtasks], type=pa.int64()),
                    "episode_index": pa.array([r["episode_index"] for r in global_subtasks], type=pa.int64()),
                    "subtask": pa.array([r["subtask"] for r in global_subtasks], type=pa.string()),
                    "start_frame": pa.array([r["start_frame"] for r in global_subtasks], type=pa.int64()),
                    "end_frame": pa.array([r["end_frame"] for r in global_subtasks], type=pa.int64()),
                    "start_s": pa.array([r["start_s"] for r in global_subtasks], type=pa.float64()),
                    "end_s": pa.array([r["end_s"] for r in global_subtasks], type=pa.float64()),
                }
            ),
            meta_dir / "subtasks.parquet",
        )

        pq.write_table(
            pa.table(
                {
                    "episode_index": pa.array([r["episode_index"] for r in episode_rows], type=pa.int64()),
                    "episode_id": pa.array([r["episode_id"] for r in episode_rows], type=pa.string()),
                    "tasks": pa.array([r["tasks"] for r in episode_rows], type=pa.list_(pa.string())),
                    "length": pa.array([r["length"] for r in episode_rows], type=pa.int64()),
                    "data/chunk_index": pa.array([r["data/chunk_index"] for r in episode_rows], type=pa.int64()),
                    "data/file_index": pa.array([r["data/file_index"] for r in episode_rows], type=pa.int64()),
                    "dataset_from_index": pa.array(
                        [r["dataset_from_index"] for r in episode_rows], type=pa.int64()
                    ),
                    "dataset_to_index": pa.array(
                        [r["dataset_to_index"] for r in episode_rows], type=pa.int64()
                    ),
                    "videos/observation.images.ego/chunk_index": pa.array(
                        [r["videos/observation.images.ego/chunk_index"] for r in episode_rows], type=pa.int64()
                    ),
                    "videos/observation.images.ego/file_index": pa.array(
                        [r["videos/observation.images.ego/file_index"] for r in episode_rows], type=pa.int64()
                    ),
                    "videos/observation.images.ego/from_timestamp": pa.array(
                        [r["videos/observation.images.ego/from_timestamp"] for r in episode_rows],
                        type=pa.float64(),
                    ),
                    "videos/observation.images.ego/to_timestamp": pa.array(
                        [r["videos/observation.images.ego/to_timestamp"] for r in episode_rows],
                        type=pa.float64(),
                    ),
                    "robot_readiness_score": pa.array(
                        [r["robot_readiness_score"] for r in episode_rows], type=pa.float32()
                    ),
                    "quality_verdict": pa.array([r["quality_verdict"] for r in episode_rows], type=pa.string()),
                    "environment_tag": pa.array([r["environment_tag"] for r in episode_rows], type=pa.string()),
                    "domain_taxonomy": pa.array([r["domain_taxonomy"] for r in episode_rows], type=pa.string()),
                    "depth_provenance": pa.array(
                        [r["depth_provenance"] for r in episode_rows], type=pa.string()
                    ),
                }
            ),
            ep_meta_dir / "file-000.parquet",
        )
        with (meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as fh:
            for row in episode_rows:
                fh.write(json.dumps(row) + "\n")

        info_out = dict(base_info)
        info_out.update(
            {
                "codebase_version": "v3.0",
                "total_episodes": len(episode_rows),
                "total_frames": global_frame_cursor,
                "total_tasks": len(tasks_sorted),
                "total_subtasks": len(global_subtasks),
                "total_shards": shard_idx,
            }
        )
        (meta_dir / "info.json").write_text(json.dumps(info_out, indent=2), encoding="utf-8")

        global_stats: dict[str, Any] = {"per_episode": per_episode_stats}
        for skey, parts in stats_acc.items():
            if parts:
                global_stats[skey] = _array_stats(np.concatenate(parts, axis=0))
        (meta_dir / "stats.json").write_text(json.dumps(global_stats, indent=2), encoding="utf-8")

        first_mod = next((d / "lerobot_v3" / "meta" / "modality.json" for d in episode_dirs if (d / "lerobot_v3" / "meta" / "modality.json").exists()), None)
        if first_mod is not None:
            shutil.copy2(first_mod, meta_dir / "modality.json")

        batch_card = {
            "schema_version": "4.0.0",
            "format": "lerobot_v3_sharded",
            "total_episodes": len(episode_rows),
            "total_frames": global_frame_cursor,
            "total_shards": shard_idx,
            "skipped_episodes": skipped,
            "mean_robot_readiness_score": round(
                float(np.mean([r["robot_readiness_score"] for r in episode_rows])), 4
            ),
        }
        (staging / "dataset_card.json").write_text(json.dumps(batch_card, indent=2), encoding="utf-8")

        gate_report: dict[str, Any] = {"passed": True, "skipped": not run_validation_gate}
        if run_validation_gate:
            gate_report = validate_lerobot_v3_dataset(staging, run_smoke_train=True, smoke_epochs=4)
            if not gate_report["passed"]:
                raise RuntimeError(f"Atomic sharding aborted — Layer 7 gate failed: {gate_report['errors']}")

        if out_final.exists():
            shutil.rmtree(out_final)
        staging.rename(out_final)

        if ledger_path is not None and ledger_path.exists():
            ledger = BatchStateLedger(ledger_path)
            for mid in accepted_manifests:
                if ledger.get(mid) is not None:
                    ledger.transition(mid, "SHARDED")

        return {
            "output_dir": str(out_final),
            "total_episodes": len(episode_rows),
            "total_frames": global_frame_cursor,
            "total_shards": shard_idx,
            "skipped": skipped,
            "validation_gate": gate_report,
        }
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
