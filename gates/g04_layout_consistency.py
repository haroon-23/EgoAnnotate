#!/usr/bin/env python3
"""Gate 04: State & Action layout consistency, delta relationship a_t ≈ delta(s_t, s_{t+1}), and units/frame schema."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd


def resolve_lerobot_dir(ep_or_ds_dir: Path) -> Path:
    if (ep_or_ds_dir / "lerobot_v3" / "meta" / "info.json").exists():
        return ep_or_ds_dir / "lerobot_v3"
    if (ep_or_ds_dir / "meta" / "info.json").exists():
        return ep_or_ds_dir
    raise FileNotFoundError(f"Could not find lerobot_v3/meta/info.json under {ep_or_ds_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate 04: State/Action Layout & Delta Consistency")
    parser.add_argument("episode_dir", type=str)
    parser.add_argument("--tol", type=float, default=1e-2, help="Max mean absolute residual for delta check")
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    print("=== G04: STATE/ACTION LAYOUT & DELTA CONSISTENCY ===")
    lr_dir = resolve_lerobot_dir(Path(args.episode_dir))
    info = json.loads((lr_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    pq_files = sorted((lr_dir / "data").rglob("*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in pq_files], ignore_index=True)

    failures: List[str] = []

    # 1. Check info.json["ego_annotate"] metadata contract
    ego_meta = info.get("ego_annotate", {})
    required_meta_keys = ["units", "frame", "scale_source", "is_metric"]
    missing_meta = [k for k in required_meta_keys if k not in ego_meta]
    if missing_meta:
        failures.append(f"info.json['ego_annotate'] missing required keys: {missing_meta}")
    else:
        print(
            f"  ego_annotate meta: frame={ego_meta['frame']!r}, scale_source={ego_meta['scale_source']!r}, "
            f"is_metric={ego_meta['is_metric']}, units={ego_meta['units']}"
        )

    # 2. Check state and action feature names alignment
    s_feat = info.get("features", {}).get("observation.state", {})
    a_feat = info.get("features", {}).get("action", {})
    s_names = s_feat.get("names", [])
    a_names = a_feat.get("names", [])
    if isinstance(s_names, dict):
        s_names = list(s_names.values())[0]
    if isinstance(a_names, dict):
        a_names = list(a_names.values())[0]

    print(f"  observation.state dim={len(s_names)}, names={s_names}")
    print(f"  action            dim={len(a_names)}, names={a_names}")

    if not s_names or not a_names:
        failures.append("Missing feature names list on observation.state or action in info.json")
    elif len(s_names) != len(a_names):
        failures.append(f"Layout length mismatch: len(state)={len(s_names)} != len(action)={len(a_names)}")
    else:
        for idx, (sn, an) in enumerate(zip(s_names, a_names)):
            # Strip prefix like `delta_` or `d_` to verify semantic slot alignment
            sn_clean = sn.replace("state_", "").replace("pos_", "")
            an_clean = an.replace("delta_", "").replace("d_", "").replace("action_", "").replace("rel_", "")
            if sn_clean != an_clean:
                failures.append(f"Slot {idx} semantic mismatch: state[{idx}]={sn!r} vs action[{idx}]={an!r}")

    # 3. Check numerical relationship between state[t], state[t+1], and action[t] on valid consecutive frames
    states = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    actions = np.stack(df["action"].to_numpy()).astype(np.float64)
    if "observation.hand_valid" in df.columns:
        vmask = np.stack(df["observation.hand_valid"].to_numpy())
        valid_1d = (vmask > 0) if vmask.ndim == 1 else np.all(vmask > 0, axis=1)
    else:
        valid_1d = np.ones(len(df), dtype=bool)

    pair_valid = valid_1d[:-1] & valid_1d[1:]
    if "episode_index" in df.columns:
        ep_idx = df["episode_index"].to_numpy()
        pair_valid = pair_valid & (ep_idx[:-1] == ep_idx[1:])

    n_valid_pairs = int(np.sum(pair_valid))
    print(f"  Consecutive valid frame pairs: {n_valid_pairs}/{max(len(df) - 1, 1)}")

    # Check wrist_z non-zero if wrist_z is in state names
    for idx, sn in enumerate(s_names):
        if "wrist_z" in sn:
            wz = states[valid_1d, idx] if np.any(valid_1d) else states[:, idx]
            wz_mean = float(np.mean(np.abs(wz)))
            wz_std = float(np.std(wz))
            print(f"  State `{sn}`: mean_abs={wz_mean:.4f}m, std={wz_std:.4f}m")
            if wz_mean < 0.05 or wz_std < 1e-4:
                failures.append(f"`{sn}` is near-zero or constant (mean_abs={wz_mean:.4f}, std={wz_std:.4e})")

    # Check translation/joint delta consistency on position/translation slots
    delta_indices = [
        i for i, an in enumerate(a_names)
        if any(k in an for k in ("dx", "dy", "dz", "wrist_x", "wrist_y", "wrist_z", "joint_", "f0", "f1", "f2"))
        and "rot6d" not in an
    ]
    mean_delta_err = 0.0
    if delta_indices and n_valid_pairs > 0 and states.shape[1] == actions.shape[1]:
        s_diff = states[1:][pair_valid][:, delta_indices] - states[:-1][pair_valid][:, delta_indices]
        a_slice = actions[:-1][pair_valid][:, delta_indices]
        # Allow either direct delta (a_t ≈ s_{t+1} - s_t) or next-step target (a_t ≈ s_{t+1}) if documented
        err_as_delta = float(np.mean(np.abs(a_slice - s_diff)))
        mean_delta_err = err_as_delta
        print(f"  Mean |action[t] - (state[t+1] - state[t])| on {len(delta_indices)} delta dims: {err_as_delta:.6f}")
        if err_as_delta > args.tol:
            failures.append(
                f"Action delta mismatch: mean |a_t - (s_{{t+1}} - s_t)| = {err_as_delta:.5f} > tol {args.tol}"
            )

    passed = len(failures) == 0
    for f_msg in failures:
        print(f"  [FAIL] {f_msg}")
    print(f"G04 VERDICT: {'PASS' if passed else 'FAIL'}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "gate": "g04_layout_consistency",
            "status": "PASS" if passed else "FAIL",
            "valid_pairs": n_valid_pairs,
            "mean_delta_error": mean_delta_err,
            "failures": failures,
        }, indent=2) + "\n", encoding="utf-8")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
