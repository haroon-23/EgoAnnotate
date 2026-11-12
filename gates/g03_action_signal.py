#!/usr/bin/env python3
"""Gate 03: Per-dimension action & state variance, absent_dims honesty, and observation.hand_valid mask."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd


def resolve_lerobot_dir(ep_or_ds_dir: Path) -> Path:
    if (ep_or_ds_dir / "lerobot_v3" / "meta" / "info.json").exists():
        return ep_or_ds_dir / "lerobot_v3"
    if (ep_or_ds_dir / "meta" / "info.json").exists():
        return ep_or_ds_dir
    raise FileNotFoundError(f"Could not find lerobot_v3/meta/info.json under {ep_or_ds_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate 03: Action & State Signal Variance Audit")
    parser.add_argument("episode_dir", type=str, help="Path to episode output dir or LeRobot dataset dir")
    parser.add_argument("--eps", type=float, default=1e-6, help="Minimum variance threshold per active dimension")
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    print("=== G03: ACTION & STATE SIGNAL VARIANCE AUDIT ===")
    try:
        lr_dir = resolve_lerobot_dir(Path(args.episode_dir))
    except FileNotFoundError as exc:
        print(f"  [FAIL] {exc}")
        return 1

    info_path = lr_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    pq_files = sorted((lr_dir / "data").rglob("*.parquet"))
    if not pq_files:
        print(f"  [FAIL] No parquet files found in {lr_dir / 'data'}")
        return 1

    df = pd.concat([pd.read_parquet(p) for p in pq_files], ignore_index=True)
    n_rows = len(df)

    ego_meta = info.get("ego_annotate", {})
    absent_action_dims = set(ego_meta.get("absent_action_dims", ego_meta.get("absent_dims", [])))
    absent_state_dims = set(ego_meta.get("absent_state_dims", ego_meta.get("absent_dims", [])))

    failures: List[str] = []

    # 1. Verify observation.hand_valid mask exists in Parquet and info.json
    has_mask_col = "observation.hand_valid" in df.columns
    has_mask_info = "observation.hand_valid" in info.get("features", {})
    if not has_mask_col or not has_mask_info:
        failures.append(
            f"Missing `observation.hand_valid` validity mask (in_parquet={has_mask_col}, in_info_json={has_mask_info})"
        )
        valid_frame_mask = np.ones(n_rows, dtype=bool)
        valid_pct = 0.0
    else:
        mask_arr = np.stack(df["observation.hand_valid"].to_numpy())
        if mask_arr.ndim == 1:
            valid_frame_mask = mask_arr > 0
        else:
            valid_frame_mask = np.any(mask_arr > 0, axis=1)
        valid_pct = float(np.mean(valid_frame_mask) * 100.0)
        print(f"  Valid hand frames (`observation.hand_valid`): {int(np.sum(valid_frame_mask))}/{n_rows} ({valid_pct:.2f}%)")
        if valid_pct <= 0.0:
            failures.append("0% valid frames in observation.hand_valid")

    # 2. Per-dimension variance audit for observation.state and action
    dim_reports: Dict[str, List[Dict[str, Any]]] = {}
    for col_key, absent_set in [("observation.state", absent_state_dims), ("action", absent_action_dims)]:
        if col_key not in df.columns:
            failures.append(f"Missing column `{col_key}` in parquet")
            continue
        arr = np.stack(df[col_key].to_numpy()).astype(np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        arr_valid = arr[valid_frame_mask] if np.any(valid_frame_mask) else arr
        names = info.get("features", {}).get(col_key, {}).get("names", [f"dim_{i}" for i in range(arr.shape[1])])
        if isinstance(names, dict):
            names = list(names.values())[0] if names else [f"dim_{i}" for i in range(arr.shape[1])]

        col_rep = []
        dead_dims = []
        for d_idx in range(arr.shape[1]):
            d_name = names[d_idx] if d_idx < len(names) else f"dim_{d_idx}"
            var_all = float(np.var(arr[:, d_idx]))
            var_val = float(np.var(arr_valid[:, d_idx]))
            mean_val = float(np.mean(arr_valid[:, d_idx]))
            is_absent_declared = (d_idx in absent_set) or (d_name in absent_set)
            passed_dim = (var_val >= args.eps) or is_absent_declared
            if not passed_dim:
                dead_dims.append(f"{d_idx}:{d_name}(var={var_val:.3e})")
            col_rep.append({
                "index": d_idx,
                "name": d_name,
                "mean_valid": round(mean_val, 6),
                "var_valid": var_val,
                "var_all": var_all,
                "declared_absent": is_absent_declared,
                "passed": passed_dim,
            })
        dim_reports[col_key] = col_rep
        print(f"  Column `{col_key}`: shape={arr.shape}, dead_undeclared_dims={len(dead_dims)}/{arr.shape[1]}")
        if dead_dims:
            print(f"    [FAIL] Dead dimensions (< {args.eps:.1e}): {', '.join(dead_dims)}")
            failures.append(f"{col_key} has {len(dead_dims)} dead undeclared dims: {', '.join(dead_dims[:8])}")

    passed = len(failures) == 0
    for f_msg in failures:
        print(f"  [FAIL] {f_msg}")
    print(f"G03 VERDICT: {'PASS' if passed else 'FAIL'}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "gate": "g03_action_signal",
            "status": "PASS" if passed else "FAIL",
            "total_frames": n_rows,
            "valid_frames_pct": round(valid_pct, 2),
            "failures": failures,
            "dimensions": dim_reports,
        }, indent=2) + "\n", encoding="utf-8")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
