#!/usr/bin/env python3
"""Gate 07: HDF5 naming honesty, layout consistency, gzip compression, and storage budget (< 500 MB/hr)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List
import h5py


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate 07: HDF5 Honesty & Compression Audit")
    parser.add_argument("episode_dir", type=str)
    parser.add_argument("--max-mb-per-hour", type=float, default=500.0)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    print("=== G07: HDF5 HONESTY, LAYOUT & COMPRESSION GATE ===")
    ep_dir = Path(args.episode_dir)
    h5_files = sorted(list(ep_dir.glob("*.hdf5")) + list(ep_dir.glob("*.h5")))
    failures: List[str] = []
    tfds_status = "NOT VERIFIED"

    if not h5_files:
        failures.append(f"No .hdf5/.h5 file found in {ep_dir}")
        print(f"  [FAIL] {failures[-1]}")
        return 1

    # Check if any file is named *rlds*
    for hf_path in h5_files:
        is_named_rlds = "rlds" in hf_path.name.lower()
        if is_named_rlds:
            try:
                import tensorflow_datasets as tfds  # type: ignore
                builder = tfds.builder_from_directory(str(ep_dir))
                _ = builder.as_dataset()
                tfds_status = "REAL"
            except ImportError:
                failures.append(
                    f"File `{hf_path.name}` claims RLDS in filename, but `tensorflow_datasets` is not installed "
                    f"and raw h5py files are not TFDS RLDS datasets. Rename to `episode_hdf5.h5` / `episode.hdf5` "
                    f"and remove false RLDS claims."
                )
            except Exception as exc:
                failures.append(f"File `{hf_path.name}` named RLDS failed `tfds` loading: {exc}")

        # Inspect internal HDF5 layout, compression, and size per hour
        size_mb = hf_path.stat().st_size / (1024.0 * 1024.0)
        with h5py.File(hf_path, "r") as hf:
            top_keys = list(hf.keys())
            has_obs_root = "obs" in top_keys
            ep_grp = hf[top_keys[0]] if top_keys else hf
            has_steps_obs = "steps" in ep_grp and "observation" in ep_grp["steps"]
            if has_obs_root and has_steps_obs:
                failures.append(f"Ambiguous dual HDF5 layout in {hf_path.name}: both `obs/*` and `steps/observation/*` present")

            # Find image dataset & check compression
            img_ds = None
            if has_steps_obs and "image" in ep_grp["steps"]["observation"]:
                img_ds = ep_grp["steps"]["observation"]["image"]
            elif "obs" in hf and "image" in hf["obs"]:
                img_ds = hf["obs"]["image"]

            if img_ds is not None:
                comp = img_ds.compression
                n_frames = int(img_ds.shape[0])
                dur_hr = max(n_frames / 30.0 / 3600.0, 1e-6)
                mb_per_hr = size_mb / dur_hr
                print(
                    f"  File `{hf_path.name}`: size={size_mb:.2f} MB, frames={n_frames}, "
                    f"compression={comp!r}, projected={mb_per_hr:.1f} MB/hour"
                )
                if comp != "gzip":
                    failures.append(f"Image dataset in `{hf_path.name}` is not gzip-compressed (compression={comp!r})")
                if mb_per_hr > args.max_mb_per_hour:
                    failures.append(
                        f"Projected HDF5 size {mb_per_hr:.1f} MB/hour exceeds limit {args.max_mb_per_hour:.1f} MB/hour"
                    )
            else:
                print(f"  File `{hf_path.name}`: size={size_mb:.2f} MB (no image dataset found)")

    passed = len(failures) == 0
    for f_msg in failures:
        print(f"  [FAIL] {f_msg}")
    print(f"G07 VERDICT: {'PASS' if passed else 'FAIL'} (TFDS_STATUS={tfds_status})")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "gate": "g07_hdf5_honesty",
            "status": "PASS" if passed else "FAIL",
            "tfds_status": tfds_status,
            "files": [str(p) for p in h5_files],
            "failures": failures,
        }, indent=2) + "\n", encoding="utf-8")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
