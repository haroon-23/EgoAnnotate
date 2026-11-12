#!/usr/bin/env python3
"""Gate 06: LeRobot v3 format strict compliance, dtype alignment, ffprobe packet sync, and DataLoader test."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def resolve_lerobot_dir(ep_or_ds_dir: Path) -> Path:
    if (ep_or_ds_dir / "lerobot_v3" / "meta" / "info.json").exists():
        return ep_or_ds_dir / "lerobot_v3"
    if (ep_or_ds_dir / "meta" / "info.json").exists():
        return ep_or_ds_dir
    raise FileNotFoundError(f"Could not find lerobot_v3/meta/info.json under {ep_or_ds_dir}")


def _ffprobe_packet_count(mp4_path: Path) -> int:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_packets", "-show_entries", "stream=nb_read_packets",
        "-of", "csv=p=0", str(mp4_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not res.stdout.strip():
        return -1
    return int(res.stdout.strip().split(",")[0])


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate 06: LeRobot v3 Format & DataLoader Verification")
    parser.add_argument("episode_dir", type=str)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    print("=== G06: LEROBOT V3 STRICT COMPLIANCE & DATALOADER GATE ===")
    lr_dir = resolve_lerobot_dir(Path(args.episode_dir))
    meta_dir = lr_dir / "meta"
    info = json.loads((meta_dir / "info.json").read_text(encoding="utf-8"))
    fps = float(info.get("fps", 30.0))

    failures: List[str] = []

    # 1. No leftover v2 files (episodes.jsonl, tasks.jsonl)
    for v2_leftover in ["episodes.jsonl", "tasks.jsonl"]:
        if (meta_dir / v2_leftover).exists():
            failures.append(f"Leftover LeRobot v2 file present: {meta_dir / v2_leftover}")

    # 2. Check v3 meta/tasks.parquet and meta/episodes/chunk-*/file-*.parquet exist
    if not (meta_dir / "tasks.parquet").exists():
        failures.append(f"Missing v3 {meta_dir / 'tasks.parquet'}")
    ep_meta_pqs = sorted((meta_dir / "episodes").rglob("*.parquet")) if (meta_dir / "episodes").exists() else []
    if not ep_meta_pqs:
        failures.append(f"Missing v3 chunked episode metadata under {meta_dir / 'episodes'}")

    # 3. Parquet schema dtype vs info.json features dtype check
    data_pqs = sorted((lr_dir / "data").rglob("*.parquet"))
    if not data_pqs:
        failures.append(f"No data parquet files found under {lr_dir / 'data'}")
        total_rows = 0
    else:
        table = pq.read_table(data_pqs[0])
        schema = table.schema
        features = info.get("features", {})
        for fname, fspec in features.items():
            if fspec.get("dtype") in ("video", "image"):
                continue
            if fname not in schema.names:
                failures.append(f"Feature `{fname}` in info.json missing from Parquet schema")
                continue
            field_type = schema.field(fname).type
            elem_type = field_type.value_type if pa.types.is_list(field_type) or pa.types.is_fixed_size_list(field_type) else field_type
            expected_dtype = str(fspec.get("dtype", "")).lower()
            actual_dtype = str(elem_type).lower()
            if expected_dtype in ("float32", "float") and actual_dtype in ("double", "float64"):
                failures.append(f"Dtype mismatch on `{fname}`: Parquet has {actual_dtype} but info.json declares {expected_dtype}")
            elif expected_dtype == "int64" and "int" not in actual_dtype:
                failures.append(f"Dtype mismatch on `{fname}`: Parquet has {actual_dtype} vs info.json {expected_dtype}")

        df = table.to_pandas()
        total_rows = sum(len(pq.read_table(p)) for p in data_pqs)
        ts = df["timestamp"].to_numpy(dtype=np.float64)
        f_idx = df["frame_index"].to_numpy(dtype=np.float64)
        max_ts_err = float(np.max(np.abs(ts - (f_idx / fps))))
        print(f"  Parquet rows={total_rows}, fps={fps}, max |timestamp - frame_index/fps| = {max_ts_err:.6f}s")
        if max_ts_err > (0.55 / fps):
            failures.append(f"Timestamp drift {max_ts_err:.5f}s exceeds 0.55/fps ({0.55/fps:.5f}s)")

    # 4. Video packet count == parquet frame count
    mp4_files = sorted((lr_dir / "videos").rglob("*.mp4")) if (lr_dir / "videos").exists() else []
    if not mp4_files:
        failures.append(f"No video files found under {lr_dir / 'videos'}")
    else:
        pkt_total = sum(_ffprobe_packet_count(m) for m in mp4_files)
        print(f"  Video files={len(mp4_files)}, total ffprobe video packets={pkt_total}, parquet rows={total_rows}")
        if pkt_total != total_rows:
            failures.append(f"Video packet count ({pkt_total}) != Parquet row count ({total_rows})")

    # 5. Load via LeRobotDataset and pull 1 DataLoader batch
    dataloader_ok = False
    try:
        import torch
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset(repo_id="local/verify_v3", root=str(lr_dir))
        loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        dataloader_ok = True
        print(f"  LeRobotDataset DataLoader batch keys: {sorted(batch.keys())}")
    except Exception as exc:
        failures.append(f"LeRobotDataset DataLoader load failed: {exc}")

    passed = len(failures) == 0
    for f_msg in failures:
        print(f"  [FAIL] {f_msg}")
    print(f"G06 VERDICT: {'PASS' if passed else 'FAIL'}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "gate": "g06_lerobot_v3",
            "status": "PASS" if passed else "FAIL",
            "total_rows": total_rows,
            "dataloader_ok": dataloader_ok,
            "failures": failures,
        }, indent=2) + "\n", encoding="utf-8")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
