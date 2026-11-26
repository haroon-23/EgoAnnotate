"""Layer 7 automated policy-readiness gate: schema, sync, finite-diff, privacy & smoke training."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from src.egoannotate.policy_train_v3 import run_policy_training_smoke


def _probe_mp4_stream(mp4_path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe or not mp4_path.exists():
        return {"exists": mp4_path.exists(), "has_audio": False, "nb_frames": None}
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_streams",
        "-of",
        "json",
        str(mp4_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        return {"exists": True, "has_audio": False, "nb_frames": None}
    data = json.loads(res.stdout or "{}")
    streams = data.get("streams", [])
    v_streams = [s for s in streams if s.get("codec_type") == "video"]
    a_streams = [s for s in streams if s.get("codec_type") == "audio"]
    nb = None
    if v_streams:
        raw_nb = v_streams[0].get("nb_frames")
        if raw_nb and str(raw_nb).isdigit():
            nb = int(raw_nb)
    return {
        "exists": True,
        "has_audio": len(a_streams) > 0,
        "nb_frames": nb,
    }


def validate_lerobot_v3_dataset(
    lerobot_dir: Path,
    *,
    run_smoke_train: bool = True,
    smoke_epochs: int = 4,
    sync_tol_s: float = 1e-5,
) -> dict[str, Any]:
    """Run all Layer 7 policy-readiness checks on a single- or multi-episode LeRobot v3.0 dataset."""
    root = Path(lerobot_dir)
    meta_dir = root / "meta"
    errors: list[str] = []

    required_meta = [
        meta_dir / "info.json",
        meta_dir / "stats.json",
        meta_dir / "tasks.parquet",
        meta_dir / "subtasks.parquet",
        meta_dir / "episodes" / "chunk-000" / "file-000.parquet",
    ]
    for req in required_meta:
        if not req.exists():
            errors.append(f"missing_meta:{req.relative_to(root)}")

    if errors:
        return {"passed": False, "errors": errors}

    info = json.loads((meta_dir / "info.json").read_text(encoding="utf-8"))
    fps = float(info.get("fps", 30.0))
    expected_dt = 1.0 / max(fps, 1.0)

    parquet_files = sorted((root / "data").rglob("*.parquet"))
    if not parquet_files:
        errors.append("no_data_parquet_shards")
        return {"passed": False, "errors": errors}

    total_rows = 0
    max_ts_jitter = 0.0
    max_fd_err = 0.0
    nan_columns: list[str] = []

    for pf in parquet_files:
        table = pq.read_table(pf)
        total_rows += table.num_rows
        for col_name in (
            "observation.state",
            "action",
            "action.canonical_delta",
            "observation.tactile",
            "observation.camera_pose_world",
            "timestamp",
        ):
            if col_name not in table.column_names:
                continue
            arr = np.asarray(table[col_name].to_pylist(), dtype=np.float64)
            if not np.all(np.isfinite(arr)):
                nan_columns.append(col_name)

        ep_ids = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        ts_all = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
        state_all = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        act_all = np.asarray(table["action"].to_pylist(), dtype=np.float32)

        for ep in np.unique(ep_ids):
            mask = ep_ids == ep
            ts = ts_all[mask]
            if ts.size >= 2:
                dts = np.diff(ts)
                if not bool(np.all(dts > 0.0)):
                    errors.append(f"non_monotonic_timestamps:ep_{ep}")
                jitter = float(np.max(np.abs(dts - expected_dt)))
                max_ts_jitter = max(max_ts_jitter, jitter)
                if jitter > max(sync_tol_s, 1e-4):
                    errors.append(f"timestamp_jitter_exceeded:ep_{ep}:{jitter:.6f}s")

            st = state_all[mask]
            ac = act_all[mask]
            if st.shape[0] >= 2 and ac.shape == st.shape:
                fd_err = float(np.mean(np.abs(ac[:-1] - (st[1:] - st[:-1]))))
                max_fd_err = max(max_fd_err, fd_err)
                if fd_err > 1e-3:
                    errors.append(f"action_state_finite_diff_mismatch:ep_{ep}:{fd_err:.5f}")

    if nan_columns:
        errors.append(f"non_finite_values:{sorted(set(nan_columns))}")

    video_files = sorted((root / "videos").rglob("*.mp4"))
    total_video_frames = 0
    audio_detected = False
    for vf in video_files:
        probe = _probe_mp4_stream(vf)
        if probe["has_audio"]:
            audio_detected = True
        if probe["nb_frames"] is not None:
            total_video_frames += int(probe["nb_frames"])

    if audio_detected:
        errors.append("audio_stream_present_in_sanitized_mp4")
    if video_files and total_video_frames > 0 and abs(total_video_frames - total_rows) > max(3, len(video_files) * 2):
        errors.append(f"video_parquet_frame_mismatch:video={total_video_frames},parquet={total_rows}")

    smoke_report: dict[str, Any] | None = None
    if run_smoke_train and total_rows >= 10:
        smoke_report = run_policy_training_smoke(root, epochs=smoke_epochs)
        if not smoke_report.get("finite_predictions", False):
            errors.append("policy_smoke_non_finite_predictions")

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "total_rows": total_rows,
        "total_video_frames": total_video_frames,
        "max_timestamp_jitter_s": round(max_ts_jitter, 8),
        "max_action_state_fd_err": round(max_fd_err, 6),
        "audio_stripped": not audio_detected,
        "policy_smoke": smoke_report,
    }
