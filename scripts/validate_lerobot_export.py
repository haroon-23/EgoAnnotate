#!/usr/bin/env python3
"""L0 validation for a LeRobot v3 export root (see src/lerobot_exporter.py).

Checks (Track 6 R1-R8), extended for Phase-B merged datasets:
  L0.1  layout: meta/info.json, meta/stats.json, meta/tasks.parquet,
        meta/episodes/chunk-000/file-000.parquet, >=1 data/chunk-*/file-*.parquet
  L0.2  info.json: codebase_version == "v3.0", data_path/video_path are v3
        *templates* (not literal paths), splits present
  L0.3  stats.json keys match info.json feature keys exactly (a mismatch
        silently disables normalization downstream)
  L0.4  data parquet(s): required columns (timestamp/frame_index/episode_index/
        index/task_index), dtypes, globally monotonic index 0..M-1 across
        chunks, timestamp ~= frame_index / fps (per-episode fps when the
        episodes parquet carries fps/fps_source columns, else info.fps)
  L0.5  episodes parquet: required columns incl. per-episode stats columns and
        dataset_from_index continuity across rows; per-ROW video checks —
        an mp4 is required only for rows declaring a video
        (videos/<key>/file_index not NaN) and must match that row's length
  L0.6  every video feature has a v3 video info block in info.json

Usage:
    python scripts/validate_lerobot_export.py <lerobot_v3_root>

Exit code 0 = all checks pass, 1 = one or more failures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


class Check:
    def __init__(self):
        self.failures: list[str] = []
        self.passes: list[str] = []

    def ok(self, name: str, cond: bool, detail: str = ""):
        (self.passes if cond else self.failures).append(name if cond else f"{name}: {detail}")
        return cond

    def report(self) -> bool:
        for name in self.passes:
            print(f"  [PASS] {name}")
        for name in self.failures:
            print(f"  [FAIL] {name}")
        print(f"\n{len(self.passes)} passed, {len(self.failures)} failed")
        return not self.failures


def main(root: str) -> int:
    c = Check()
    root_p = Path(root)
    meta = root_p / "meta"
    print(f"Validating LeRobot v3 export: {root_p}")

    # L0.1 layout -----------------------------------------------------------
    info_p = meta / "info.json"
    stats_p = meta / "stats.json"
    tasks_p = meta / "tasks.parquet"
    ep_p = meta / "episodes" / "chunk-000" / "file-000.parquet"
    data_ps = sorted((root_p / "data").glob("chunk-*/file-*.parquet"))
    for p, label in [(info_p, "meta/info.json"), (stats_p, "meta/stats.json"),
                     (tasks_p, "meta/tasks.parquet"),
                     (ep_p, "meta/episodes/chunk-000/file-000.parquet")]:
        c.ok(f"layout:{label}", p.exists(), "missing")
    c.ok("layout:data/chunk-*/file-*.parquet", len(data_ps) > 0,
         "no data chunks found")
    if c.failures:
        c.report()
        return 1

    info = json.loads(info_p.read_text())
    stats = json.loads(stats_p.read_text())
    df = pd.concat([pd.read_parquet(p) for p in data_ps], ignore_index=True)
    ep_df = pd.read_parquet(ep_p)

    # L0.2 info.json ---------------------------------------------------------
    c.ok("info:codebase_version==v3.0", info.get("codebase_version") == "v3.0",
         f"got {info.get('codebase_version')!r}")
    c.ok("info:data_path is template",
         "{chunk_index:03d}" in str(info.get("data_path", "")),
         f"got {info.get('data_path')!r}")
    c.ok("info:video_path is template",
         "{video_key}" in str(info.get("video_path", "")),
         f"got {info.get('video_path')!r}")
    c.ok("info:splits present", isinstance(info.get("splits"), dict),
         "missing splits (v3 reader expects them)")
    features = info.get("features", {})
    c.ok("info:features non-empty", bool(features), "empty")

    # L0.3 stats.json keys match feature keys exactly ------------------------
    stats_keys = set(stats.keys())
    feature_keys = {k for k, v in features.items() if v.get("dtype") != "video"}
    c.ok("stats:keys match features", stats_keys == feature_keys,
         f"stats={sorted(stats_keys)} features={sorted(feature_keys)}")
    for key, s in stats.items():
        c.ok(f"stats:{key} has min/max/mean/std/count/q01/q99",
             all(k in s for k in ("min", "max", "mean", "std", "count", "q01", "q99")),
             f"keys={sorted(s.keys())}")

    # L0.4 data parquet ------------------------------------------------------
    required = ["timestamp", "frame_index", "episode_index", "index", "task_index",
                "observation.state", "action"]
    for col in required:
        c.ok(f"data:has column {col}", col in df.columns, "missing")
    # per-episode fps when the episodes parquet carries it (merged datasets),
    # else the dataset-global info.fps (single-episode exports)
    has_ep_fps = "fps" in ep_df.columns
    if all(col in df.columns for col in required):
        c.ok("data:timestamp float", str(df["timestamp"].dtype).startswith("float"),
             f"dtype={df['timestamp'].dtype}")
        idx = df["index"].to_numpy()
        c.ok("data:index monotonic 0..M-1",
             bool(np.array_equal(idx, np.arange(len(df)))), "not 0..M-1 monotonic")
        ts = df["timestamp"].to_numpy(dtype=np.float64)
        fi = df["frame_index"].to_numpy(dtype=np.float64)
        if has_ep_fps:
            ok_all, bad = True, []
            for _, erow in ep_df.iterrows():
                m = df["episode_index"].to_numpy() == int(erow["episode_index"])
                efps = float(erow["fps"])
                if not np.allclose(ts[m], fi[m] / efps, atol=1e-3):
                    ok_all = False
                    bad.append(int(erow["episode_index"]))
            c.ok("data:timestamp == frame_index/episode_fps (per row)",
                 ok_all, f"drift in episodes {bad}")
        else:
            fps = float(info.get("fps", 30))
            expected = fi / fps
            c.ok("data:timestamp == frame_index/fps",
                 bool(np.allclose(ts, expected, atol=1e-3)), "drift from video clock")
        c.ok("data:total_frames consistent",
             len(df) == int(info.get("total_frames", -1)),
             f"rows={len(df)} info={info.get('total_frames')}")
        c.ok("data:chunks cover info.total_chunks",
             len(data_ps) == int(info.get("total_chunks", -1)),
             f"chunks={len(data_ps)} info={info.get('total_chunks')}")

    # L0.5 episodes parquet + video lookup -----------------------------------
    ep_required = ["episode_index", "tasks", "length",
                   "data/chunk_index", "data/file_index",
                   "dataset_from_index", "dataset_to_index"]
    for col in ep_required:
        c.ok(f"episodes:has column {col}", col in ep_df.columns, "missing")
    for feat in sorted(feature_keys):
        for agg in ("min", "max", "mean", "std", "count"):
            c.ok(f"episodes:has stats/{feat}/{agg}",
                 f"stats/{feat}/{agg}" in ep_df.columns, "missing (v3 R1)")

    # dataset_from_index continuity across episode rows (multi-episode merge)
    ep_sorted = ep_df.sort_values("episode_index").reset_index(drop=True)
    c.ok("episodes:episode_index 0..K-1",
         bool(np.array_equal(ep_sorted["episode_index"].to_numpy(),
                             np.arange(len(ep_sorted)))),
         "episode_index not 0..K-1")
    exp_from = np.cumsum([0] + ep_sorted["length"].tolist()[:-1])
    c.ok("episodes:dataset_from_index continuity",
         bool(np.array_equal(ep_sorted["dataset_from_index"].to_numpy(), exp_from)),
         "gaps/overlaps in frame offsets")
    c.ok("episodes:dataset_to_index == from + length",
         bool(np.array_equal(
             ep_sorted["dataset_to_index"].to_numpy(),
             ep_sorted["dataset_from_index"].to_numpy() + ep_sorted["length"].to_numpy())),
         "mismatch")
    c.ok("episodes:last dataset_to_index == total rows",
         int(ep_sorted["dataset_to_index"].iloc[-1]) == len(df),
         f"to={ep_sorted['dataset_to_index'].iloc[-1]} rows={len(df)}")

    video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
    total_videos = int(info.get("total_videos", 0))
    c.ok("info:total_videos matches video features",
         (total_videos > 0) == bool(video_keys),
         f"total_videos={total_videos} video_features={video_keys}")
    n_declared = 0
    for vkey in video_keys:
        for suffix in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
            col = f"videos/{vkey}/{suffix}"
            c.ok(f"episodes:has {col}", col in ep_df.columns, "missing (v3 R3)")
        if f"videos/{vkey}/file_index" not in ep_df.columns:
            continue
        # Per-ROW checks: an mp4 is required only for rows declaring a video.
        for _, erow in ep_df.iterrows():
            fi_cell = erow[f"videos/{vkey}/file_index"]
            ep_i = int(erow["episode_index"])
            if pd.isna(fi_cell):
                continue
            n_declared += 1
            fi = int(fi_cell)
            vpath = root_p / "videos" / vkey / "chunk-000" / f"file-{fi:03d}.mp4"
            ok_mp4 = c.ok(f"video:{vkey} ep{ep_i} mp4 exists", vpath.exists(),
                           f"missing {vpath}")
            if ok_mp4:
                cap = cv2.VideoCapture(str(vpath))
                try:
                    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                finally:
                    cap.release()
                length = int(erow["length"])
                c.ok(f"video:{vkey} ep{ep_i} frame count matches length",
                     n == length, f"mp4={n} length={length}")
                row_fps = float(erow["fps"]) if "fps" in ep_df.columns \
                    else float(info.get("fps", 30))
                to_ts = float(erow[f"videos/{vkey}/to_timestamp"])
                c.ok(f"video:{vkey} ep{ep_i} to_timestamp ~= duration",
                     abs(to_ts - (length - 1) / row_fps) < 1.0,
                     f"to_ts={to_ts} len={length} fps={row_fps}")
    c.ok("info:total_videos == declared video rows",
         n_declared == total_videos,
         f"declared={n_declared} total_videos={total_videos}")

    # L0.6 video info block ---------------------------------------------------
    for vkey in video_keys:
        blk = features[vkey].get("info", {})
        need = {"video.fps", "video.codec", "video.pix_fmt",
                "video.height", "video.width", "video.channels", "video.is_depth_map"}
        c.ok(f"info:{vkey} video info block", need.issubset(blk.keys()),
             f"missing={[k for k in need if k not in blk]}")

    print()
    return 0 if c.report() else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
