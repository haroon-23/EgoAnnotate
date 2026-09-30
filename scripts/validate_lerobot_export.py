#!/usr/bin/env python3
"""L0 validation for a LeRobot v3 export root (see src/lerobot_exporter.py).

Checks (Track 6 R1-R8):
  L0.1  layout: meta/info.json, meta/stats.json, meta/tasks.parquet,
        meta/episodes/chunk-000/file-000.parquet, data/chunk-000/file-000.parquet
  L0.2  info.json: codebase_version == "v3.0", data_path/video_path are v3
        *templates* (not literal paths), splits present
  L0.3  stats.json keys match info.json feature keys exactly (a mismatch
        silently disables normalization downstream)
  L0.4  data parquet: required columns (timestamp/frame_index/episode_index/
        index/task_index), dtypes, globally monotonic index,
        timestamp ~= frame_index / fps
  L0.5  episodes parquet: required columns incl. per-episode stats columns;
        when total_videos > 0, video-lookup columns resolve to a real mp4 whose
        frame count matches total_frames
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
    data_p = root_p / "data" / "chunk-000" / "file-000.parquet"
    for p, label in [(info_p, "meta/info.json"), (stats_p, "meta/stats.json"),
                     (tasks_p, "meta/tasks.parquet"),
                     (ep_p, "meta/episodes/chunk-000/file-000.parquet"),
                     (data_p, "data/chunk-000/file-000.parquet")]:
        c.ok(f"layout:{label}", p.exists(), "missing")
    if c.failures:
        c.report()
        return 1

    info = json.loads(info_p.read_text())
    stats = json.loads(stats_p.read_text())
    df = pd.read_parquet(data_p)
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
    if all(col in df.columns for col in required):
        c.ok("data:timestamp float", str(df["timestamp"].dtype).startswith("float"),
             f"dtype={df['timestamp'].dtype}")
        idx = df["index"].to_numpy()
        c.ok("data:index monotonic 0..N-1",
             bool(np.array_equal(idx, np.arange(len(df)))), "not 0..N-1 monotonic")
        fps = float(info.get("fps", 30))
        ts = df["timestamp"].to_numpy(dtype=np.float64)
        expected = df["frame_index"].to_numpy(dtype=np.float64) / fps
        c.ok("data:timestamp == frame_index/fps",
             bool(np.allclose(ts, expected, atol=1e-3)), "drift from video clock")
        c.ok("data:total_frames consistent",
             len(df) == int(info.get("total_frames", -1)),
             f"rows={len(df)} info={info.get('total_frames')}")

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

    video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
    total_videos = int(info.get("total_videos", 0))
    c.ok("info:total_videos matches video features",
         (total_videos > 0) == bool(video_keys),
         f"total_videos={total_videos} video_features={video_keys}")
    for vkey in video_keys:
        for suffix in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
            col = f"videos/{vkey}/{suffix}"
            c.ok(f"episodes:has {col}", col in ep_df.columns, "missing (v3 R3)")
        vpath = root_p / "videos" / vkey / "chunk-000" / "file-000.mp4"
        c.ok(f"video:{vkey} mp4 exists", vpath.exists(), f"missing {vpath}")
        if vpath.exists():
            cap = cv2.VideoCapture(str(vpath))
            try:
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            finally:
                cap.release()
            c.ok(f"video:{vkey} frame count matches rows",
                 n == len(df), f"mp4={n} rows={len(df)}")
            to_ts = float(ep_df[f"videos/{vkey}/to_timestamp"].iloc[0])
            dur = n / float(info.get("fps", 30)) if n else 0.0
            c.ok(f"video:{vkey} to_timestamp ~= duration",
                 abs(to_ts - (len(df) - 1) / float(info.get("fps", 30))) < 1.0,
                 f"to_ts={to_ts} dur~={dur:.2f}")

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
