#!/usr/bin/env python3
"""Merge per-episode LeRobot v3 exports into ONE training dataset.

Each input is a ``lerobot_v3/`` root as produced by
:func:`src.lerobot_exporter.export_to_lerobot` (or returned by
``export_all_lerobot``). The merge concatenates frame tables, renumbers
``index``/``episode_index``, dedupes tasks, repacks data into chunks, copies
videos (never re-encodes), and **recomputes** ``meta/stats.json`` pooled over
all frames via :func:`src.lerobot_exporter.compute_feature_stats` — stats are
never averaged across episodes (wrong for std/quantiles).

Validation gate (refuses loudly, never silently coerces):
  - mixed ``export_mode`` (24D human vs 8D robot)          -> ValueError
  - differing feature schema (keys/dtype/shape, excl. video);
    incl. human-mode ``observation.robot_*`` side-channels present in only
    some episodes                                        -> ValueError
  - differing ``robot_type``                              -> ValueError
  - differing video keys across video-having episodes     -> ValueError
  - mixed fps                                             -> warn + allow
    (per-episode fps/fps_source kept; merged info.fps = first episode's)
  - video in some episodes only                           -> allow
    (total_videos = count with video; lookup cols NaN for video-less rows)
  - zero episodes / duplicate dirs                         -> ValueError / dedupe

Usage:
    python scripts/merge_lerobot.py <ep1/lerobot_v3> [<ep2/lerobot_v3> ...] \\
        -o <out>/merged_lerobot_v3 [--chunks-size 1000]
    python scripts/merge_lerobot.py --scan data/output -o <out>/merged_lerobot_v3
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.lerobot_exporter import compute_feature_stats  # noqa: E402

logger = logging.getLogger(__name__)

# Pooled-stats concat is O(total_frames) RAM; warn past this soft cap.
SOFT_FRAME_CAP = 2_000_000


def find_lerobot_exports(output_dir: str | Path) -> List[Path]:
    """Scan ``<output_dir>/*/lerobot_v3`` and return sorted export roots."""
    root = Path(output_dir)
    found = sorted(p.resolve() for p in root.glob("*/lerobot_v3") if p.is_dir())
    logger.info("Found %d lerobot_v3 exports under %s", len(found), root)
    return found


def _as_list(cell) -> list:
    """Normalize a parquet list-cell (list/tuple/ndarray) to a Python list."""
    if isinstance(cell, np.ndarray):
        return cell.tolist()
    if isinstance(cell, (list, tuple)):
        return list(cell)
    return [cell]


def _check_schema(infos: List[dict]) -> Tuple[dict, Optional[str]]:
    """Validate identical non-video feature schemas; return (features, video_key).

    Raises ValueError on any mismatch.
    """
    ref = infos[0]["features"]
    ref_keys = {k: (v.get("dtype"), tuple(v.get("shape") or ()))
                for k, v in ref.items() if v.get("dtype") != "video"}
    ref_video_keys = [k for k, v in ref.items() if v.get("dtype") == "video"]

    for i, info in enumerate(infos[1:], start=1):
        feats = info["features"]
        keys = {k: (v.get("dtype"), tuple(v.get("shape") or ()))
                for k, v in feats.items() if v.get("dtype") != "video"}
        if keys != ref_keys:
            only_a = sorted(set(ref_keys) - set(keys))
            only_b = sorted(set(keys) - set(ref_keys))
            raise ValueError(
                f"Feature schema mismatch between episode 0 and episode {i}: "
                f"only in episode 0: {only_a}; only in episode {i}: {only_b}. "
                "Refusing to merge — silent schema mismatch poisons training."
            )
        vk = [k for k, v in feats.items() if v.get("dtype") == "video"]
        if vk and ref_video_keys and vk != ref_video_keys:
            raise ValueError(
                f"Video key mismatch: episode 0 uses {ref_video_keys}, "
                f"episode {i} uses {vk}. Refusing to merge."
            )
    video_key = ref_video_keys[0] if ref_video_keys else None
    return ref, video_key


def _load_episode(ep_dir: Path) -> dict:
    """Load one per-episode export into memory (frame table + metadata)."""
    info = json.loads((ep_dir / "meta" / "info.json").read_text())
    df = pd.read_parquet(ep_dir / "data" / "chunk-000" / "file-000.parquet")
    ep_df = pd.read_parquet(ep_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    if len(ep_df) != 1:
        raise ValueError(f"Expected single-row episodes parquet in {ep_dir}, got {len(ep_df)} rows")
    row = ep_df.iloc[0]
    tasks = _as_list(row["tasks"])
    task_desc = _as_list(tasks[0])[0] if tasks and isinstance(tasks[0], (list, tuple, np.ndarray)) else tasks[0]
    return {
        "dir": ep_dir,
        "info": info,
        "df": df,
        "task_desc": str(task_desc),
        "n_frames": len(df),
        "fps": float(info.get("fps", 30.0)),
        "fps_source": str(info.get("fps_source", "unknown")),
        "has_video": int(info.get("total_videos", 0)) > 0,
    }


def _frame_arrays(df: pd.DataFrame, feature_keys: List[str]) -> Dict[str, np.ndarray]:
    """Split a frame DataFrame into per-feature arrays for stats computation."""
    arrays: Dict[str, np.ndarray] = {}
    for key in feature_keys:
        col = df[key]
        first = col.iloc[0]
        if isinstance(first, (list, np.ndarray)):
            arrays[key] = np.stack(col.to_numpy()).astype(np.float64)
        else:
            arrays[key] = col.to_numpy()
    return arrays


def merge_lerobot_datasets(
    episode_dirs: List[str | Path],
    out_dir: str | Path,
    chunks_size: int = 1000,
) -> Path:
    """Merge per-episode ``lerobot_v3`` exports into one dataset at ``out_dir``.

    Returns the output root path. See module docstring for the validation gate.
    """
    # Dedupe by resolved path, preserving input order.
    seen: Dict[Path, None] = {}
    for d in episode_dirs:
        p = Path(d).resolve()
        seen.setdefault(p, None)
    dirs = list(seen.keys())
    if not dirs:
        raise ValueError("merge_lerobot_datasets: no episode directories given")

    eps = [_load_episode(d) for d in dirs]
    n_eps = len(eps)

    # ---- Validation gate -------------------------------------------------
    modes = {e["info"].get("export_mode") for e in eps}
    if len(modes) > 1:
        raise ValueError(
            f"Mixed export_mode across episodes: {sorted(modes)}. "
            "24D human and 8D robot dims are incompatible — refusing to merge."
        )
    export_mode = modes.pop()
    robot_types = {e["info"].get("robot_type") for e in eps}
    if len(robot_types) > 1:
        raise ValueError(
            f"Mixed robot_type across episodes: {sorted(robot_types)}. "
            "Refusing to merge (e.g. Franka-8D != G1-8D without embodiment conditioning)."
        )
    robot_type = robot_types.pop()
    action_convention = eps[0]["info"].get("action_convention", "")
    if any(e["info"].get("action_convention") != action_convention for e in eps):
        raise ValueError("Mixed action_convention across episodes — refusing to merge.")

    features, video_key = _check_schema([e["info"] for e in eps])
    feature_keys = [k for k, v in features.items() if v.get("dtype") != "video"]

    fpss = {e["fps"] for e in eps}
    if len(fpss) > 1:
        logger.warning(
            "Mixed fps across episodes %s — allowing; per-episode fps kept in "
            "episodes rows, merged info.fps = first episode's (%.3f).",
            sorted(fpss), eps[0]["fps"],
        )

    total_frames = sum(e["n_frames"] for e in eps)
    if total_frames > SOFT_FRAME_CAP:
        logger.warning(
            "Merging %d frames (> %d soft cap): pooled stats concat is O(M) RAM.",
            total_frames, SOFT_FRAME_CAP,
        )

    # ---- Task dedupe ------------------------------------------------------
    task_descs: List[str] = []
    for e in eps:
        if e["task_desc"] not in task_descs:
            task_descs.append(e["task_desc"])
    task_to_idx = {t: i for i, t in enumerate(task_descs)}

    # ---- Renumber frame tables -------------------------------------------
    frames: List[pd.DataFrame] = []
    offset = 0
    for i, e in enumerate(eps):
        df = e["df"].copy()
        df["episode_index"] = i
        df["index"] = np.arange(offset, offset + e["n_frames"])
        df["task_index"] = task_to_idx[e["task_desc"]]
        # frame_index resets per episode (already 0..N-1 from export); timestamp kept as-is.
        frames.append(df)
        offset += e["n_frames"]
    full = pd.concat(frames, ignore_index=True)

    # ---- Greedy chunk packing (no episode spans a chunk boundary unless it
    #      alone exceeds chunks_size) --------------------------------------
    # chunks: list of list of (ep_idx, start, end) slices into `frames`
    chunks: List[List[Tuple[int, int, int]]] = []
    cur: List[Tuple[int, int, int]] = []
    cur_frames = 0
    ep_first_chunk: List[int] = []
    for i, e in enumerate(eps):
        n = e["n_frames"]
        if n > chunks_size:
            # Oversized episode: gets its own chunk(s), split across them.
            if cur:
                chunks.append(cur)
                cur, cur_frames = [], []
            start = 0
            first = len(chunks)
            while start < n:
                end = min(n, start + chunks_size)
                chunks.append([(i, start, end)])
                start = end
            ep_first_chunk.append(first)
            logger.info("Episode %d (%d frames) exceeds chunks_size=%d: split across chunks %d..%d",
                        i, n, chunks_size, first, len(chunks) - 1)
        elif cur_frames + n > chunks_size and cur:
            chunks.append(cur)
            cur, cur_frames = [], []
            ep_first_chunk.append(len(chunks))
            cur = [(i, 0, n)]
            cur_frames = n
        else:
            # Episode lands in the chunk currently being filled, whose index
            # is len(chunks) (it hasn't been appended yet).
            ep_first_chunk.append(len(chunks))
            cur.append((i, 0, n))
            cur_frames += n
    if cur:
        chunks.append(cur)
    n_chunks = len(chunks)

    out = Path(out_dir)
    data_root = out / "data"
    for c in range(n_chunks):
        cdir = data_root / f"chunk-{c:03d}"
        cdir.mkdir(parents=True, exist_ok=True)
        parts = []
        for ep_idx, s, e_ in chunks[c]:
            parts.append(frames[ep_idx].iloc[s:e_])
        pd.concat(parts, ignore_index=True).to_parquet(cdir / "file-000.parquet", index=False)

    # ---- Videos: copy (never re-encode) -----------------------------------
    videos_root = out / "videos" / video_key if video_key else None
    if videos_root:
        (videos_root / "chunk-000").mkdir(parents=True, exist_ok=True)
    v = 0
    ep_video_file: List[Optional[int]] = []
    video_sizes = 0
    for e in eps:
        if e["has_video"] and video_key:
            src = e["dir"] / "videos" / video_key / "chunk-000" / "file-000.mp4"
            if not src.exists():
                raise FileNotFoundError(
                    f"Episode {e['dir']} declares total_videos>0 but {src} is missing")
            dst = videos_root / "chunk-000" / f"file-{v:03d}.mp4"
            shutil.copy2(src, dst)
            video_sizes += dst.stat().st_size
            ep_video_file.append(v)
            v += 1
        else:
            ep_video_file.append(None)
    total_videos = v

    # ---- Episodes parquet --------------------------------------------------
    ep_rows: List[dict] = []
    offset = 0
    for i, e in enumerate(eps):
        n = e["n_frames"]
        ep_stats = compute_feature_stats(_frame_arrays(frames[i], feature_keys))
        row: dict = {
            "episode_index": i,
            "tasks": [[e["task_desc"]]],
            "length": n,
            "data/chunk_index": ep_first_chunk[i],
            "data/file_index": 0,
            "dataset_from_index": offset,
            "dataset_to_index": offset + n,
            "export_mode": export_mode,
            "action_convention": action_convention,
            "robot_type": robot_type,
            "fps": e["fps"],
            "fps_source": e["fps_source"],
        }
        for feat_key, s in ep_stats.items():
            for agg in ("min", "max", "mean", "std", "count"):
                row[f"stats/{feat_key}/{agg}"] = s[agg]
        if video_key:
            vf = ep_video_file[i]
            row[f"videos/{video_key}/chunk_index"] = 0 if vf is not None else float("nan")
            row[f"videos/{video_key}/file_index"] = vf if vf is not None else float("nan")
            row[f"videos/{video_key}/from_timestamp"] = 0.0 if vf is not None else float("nan")
            row[f"videos/{video_key}/to_timestamp"] = (
                float((n - 1) / e["fps"]) if vf is not None else float("nan"))
        ep_rows.append(row)
        offset += n
    ep_meta_dir = out / "meta" / "episodes" / "chunk-000"
    ep_meta_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ep_rows).to_parquet(ep_meta_dir / "file-000.parquet", index=False)

    # ---- stats.json: pooled recompute over ALL frames ----------------------
    stats = compute_feature_stats(_frame_arrays(full, feature_keys))
    meta_dir = out / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ---- tasks.parquet ------------------------------------------------------
    pd.DataFrame([{"task_index": i, "task": t} for i, t in enumerate(task_descs)]
                 ).to_parquet(meta_dir / "tasks.parquet", index=False)

    # ---- info.json -----------------------------------------------------------
    data_sizes = sum(
        (data_root / f"chunk-{c:03d}" / "file-000.parquet").stat().st_size
        for c in range(n_chunks))
    info = {
        "codebase_version": "v3.0",
        "robot_type": robot_type,
        "export_mode": export_mode,
        "action_convention": action_convention,
        "fps": eps[0]["fps"],
        "fps_source": eps[0]["fps_source"],
        "total_episodes": n_eps,
        "total_frames": total_frames,
        "total_tasks": len(task_descs),
        "total_videos": total_videos,
        "total_chunks": n_chunks,
        "chunks_size": chunks_size,
        "splits": {"train": f"0:{n_eps}"},
        "data_files_size_in_mb": round(data_sizes / 1e6, 3),
        "video_files_size_in_mb": round(video_sizes / 1e6, 3),
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    logger.info("Merged %d episodes (%d frames, %d videos, %d chunks) -> %s",
                n_eps, total_frames, total_videos, n_chunks, out)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Merge per-episode LeRobot v3 exports into one training dataset.")
    ap.add_argument("episode_dirs", nargs="*",
                    help="lerobot_v3/ roots to merge (as returned by export_all_lerobot)")
    ap.add_argument("--scan", metavar="OUTPUT_DIR",
                    help="instead of listing dirs, scan <OUTPUT_DIR>/*/lerobot_v3")
    ap.add_argument("-o", "--output", required=True,
                    help="output root for the merged dataset")
    ap.add_argument("--chunks-size", type=int, default=1000,
                    help="frames per data chunk (default: 1000)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    if args.scan:
        dirs = find_lerobot_exports(args.scan)
    else:
        dirs = [Path(d) for d in args.episode_dirs]
    if not dirs:
        print("No episode exports to merge.", file=sys.stderr)
        return 2
    try:
        out = merge_lerobot_datasets(dirs, args.output, chunks_size=args.chunks_size)
    except (ValueError, FileNotFoundError) as e:
        print(f"Merge refused: {e}", file=sys.stderr)
        return 1
    print(f"Merged dataset written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
