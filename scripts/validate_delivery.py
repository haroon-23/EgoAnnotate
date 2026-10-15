#!/usr/bin/env python3
"""Comprehensive delivery validator for EgoAnnotate v1.0.0 episodes."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import h5py

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.mcap_exporter import verify_mcap_file

BANNED_PLACEHOLDERS = {"manipulating object", "todo", "placeholder", "canned_default"}


def ffprobe_info(video_path: Path) -> Dict[str, Any]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_frames,codec_name,duration",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {video_path}: {res.stderr}")
    data = json.loads(res.stdout)
    s = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    dur = float(s.get("duration") or fmt.get("duration") or 0.0)
    nb = s.get("nb_frames", "0")
    return {
        "width": int(s.get("width", 0)),
        "height": int(s.get("height", 0)),
        "nb_frames": int(nb) if str(nb).isdigit() else 0,
        "duration_s": round(dur, 3),
        "codec": str(s.get("codec_name", "unknown")),
    }


def validate_episode_dir(episode_dir: Path, source_video: Path | None = None) -> Dict[str, Any]:
    """Validate all v1.0 deliverables in an episode output directory. Raises RuntimeError on failure."""
    failures: List[str] = []
    report: Dict[str, Any] = {"episode_dir": str(episode_dir), "artifacts": {}}

    # 1. annotations.json & summary.json (and ensure MCAP exists)
    ann_path = episode_dir / "annotations.json"
    sum_path = episode_dir / "summary.json"
    mcap_path = episode_dir / "episode.mcap"

    if not ann_path.exists():
        failures.append("annotations.json: MISSING")
    else:
        ann_data = json.loads(ann_path.read_text(encoding="utf-8"))
        task_desc = str(ann_data.get("task_description", "")).strip().lower()
        if not task_desc or task_desc in BANNED_PLACEHOLDERS:
            failures.append(f"annotations.json: invalid/canned task_description={task_desc!r}")
        report["artifacts"]["annotations.json"] = {
            "num_frames": ann_data.get("num_frames"),
            "task_description": ann_data.get("task_description"),
            "target_robot": ann_data.get("target_robot"),
        }
        # Auto-backfill episode.mcap if validating an older episode directory
        if not mcap_path.exists():
            from src.datatypes import AnnotatedEpisode
            from src.mcap_exporter import export_episode_to_mcap
            ep_obj = AnnotatedEpisode.load_json(str(ann_path))
            export_episode_to_mcap(ep_obj, mcap_path)

    if not sum_path.exists():
        failures.append("summary.json: MISSING")
    else:
        report["artifacts"]["summary.json"] = json.loads(sum_path.read_text(encoding="utf-8"))

    # 2. overlay_annotated.mp4 & side_by_side.mp4
    src_info = ffprobe_info(source_video) if (source_video and source_video.exists()) else None
    for vname in ("overlay_annotated.mp4", "side_by_side.mp4"):
        vp = episode_dir / vname
        if not vp.exists() or vp.stat().st_size == 0:
            failures.append(f"{vname}: MISSING or 0 bytes")
            continue
        vinfo = ffprobe_info(vp)
        report["artifacts"][vname] = vinfo
        if vinfo["codec"] not in ("h264", "avc1"):
            failures.append(f"{vname}: codec={vinfo['codec']} (expected h264/avc1)")
        if vname == "overlay_annotated.mp4" and vinfo["width"] <= 224 and vinfo["height"] <= 224:
            failures.append(f"{vname}: resolution {vinfo['width']}x{vinfo['height']} looks like 224x224 debug HUD")
        if src_info and vname == "overlay_annotated.mp4":
            if (vinfo["width"], vinfo["height"]) != (src_info["width"], src_info["height"]):
                failures.append(
                    f"{vname}: resolution {vinfo['width']}x{vinfo['height']} != source {src_info['width']}x{src_info['height']}"
                )

    # 3. episode_rlds.hdf5
    rlds_path = episode_dir / "episode_rlds.hdf5"
    if not rlds_path.exists():
        failures.append("episode_rlds.hdf5: MISSING")
    else:
        with h5py.File(rlds_path, "r") as hf:
            keys = list(hf.keys())
            if not keys:
                failures.append("episode_rlds.hdf5: empty top-level keys")
            else:
                grp = hf[keys[0]]
                for req in ("steps/action", "steps/observation/language_instruction"):
                    if req not in grp:
                        failures.append(f"episode_rlds.hdf5: missing {req}")
                report["artifacts"]["episode_rlds.hdf5"] = {
                    "episode_key": keys[0],
                    "action_shape": list(grp["steps/action"].shape) if "steps/action" in grp else None,
                }

    # 4. lerobot_v3/
    lr_dir = episode_dir / "lerobot_v3"
    info_json = lr_dir / "meta" / "info.json"
    pqs = list((lr_dir / "data").rglob("*.parquet")) if (lr_dir / "data").exists() else []
    if not info_json.exists() or not pqs:
        failures.append("lerobot_v3: missing meta/info.json or data/*.parquet")
    else:
        import pyarrow.parquet as pq
        tbl = pq.read_table(pqs[0])
        report["artifacts"]["lerobot_v3"] = {
            "num_rows": tbl.num_rows,
            "columns": tbl.column_names,
        }

    # 5. episode.mcap
    if not mcap_path.exists():
        failures.append("episode.mcap: MISSING")
    else:
        report["artifacts"]["episode.mcap"] = verify_mcap_file(mcap_path)

    report["passed"] = len(failures) == 0
    report["failures"] = failures
    if failures:
        raise RuntimeError("Delivery validation failed:\n  - " + "\n  - ".join(failures))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate EgoAnnotate episode deliverables.")
    ap.add_argument("episode_dir", type=Path, help="Path to data/output/<episode_id>")
    ap.add_argument("--source-video", type=Path, default=None, help="Optional source video for ffprobe comparison")
    args = ap.parse_args()
    rep = validate_episode_dir(args.episode_dir, args.source_video)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
