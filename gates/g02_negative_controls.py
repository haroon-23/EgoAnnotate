#!/usr/bin/env python3
"""Gate 02: Negative controls (zero hands, static hands, random white-noise hands).

Builds synthetic degenerate episodes in a temporary directory and runs the repository's
retargeting / verification / delivery checks against them. A verifier that passes
degenerate garbage MUST fail this gate.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_hand_dict(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, side: str) -> Dict[str, Any]:
    landmarks = [
        {"x": float(xs[k]), "y": float(ys[k]), "z": float(zs[k])}
        for k in range(21)
    ]
    return {
        "handedness": side,
        "confidence": 0.95,
        "is_interpolated": False,
        "landmarks": landmarks,
    }


def _build_synthetic_annotations(mode: str, num_frames: int = 60, fps: float = 30.0) -> Dict[str, Any]:
    rng = np.random.default_rng(12345)
    frames = []
    # Base canonical hand template in normalized coordinates
    base_x = np.linspace(0.45, 0.55, 21, dtype=np.float32)
    base_y = np.linspace(0.65, 0.45, 21, dtype=np.float32)
    base_z = np.zeros(21, dtype=np.float32)

    for i in range(num_frames):
        ts = round(i / fps, 4)
        if mode == "zero_hands":
            lh, rh = None, None
        elif mode == "static_hands":
            lh = _make_hand_dict(base_x - 0.15, base_y, base_z, "Left")
            rh = _make_hand_dict(base_x + 0.15, base_y, base_z, "Right")
        elif mode == "random_noise":
            lh = _make_hand_dict(rng.uniform(0.0, 1.0, 21), rng.uniform(0.0, 1.0, 21), rng.uniform(-0.5, 0.5, 21), "Left")
            rh = _make_hand_dict(rng.uniform(0.0, 1.0, 21), rng.uniform(0.0, 1.0, 21), rng.uniform(-0.5, 0.5, 21), "Right")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        frames.append({
            "frame_idx": i,
            "timestamp": ts,
            "left_hand": lh,
            "right_hand": rh,
            "objects": [],
            "action_segment": "idle",
            "frame_description": f"synthetic_{mode}",
        })

    return {
        "metadata": {
            "episode_id": f"neg_{mode}",
            "video_path": f"neg_{mode}.mp4",
            "task_description": f"Negative control {mode}",
            "num_frames": num_frames,
            "duration_seconds": round((num_frames - 1) / fps, 4),
            "target_robot": "panda",
        },
        "segments": [
            {
                "name": "idle",
                "start_time": 0.0,
                "end_time": round((num_frames - 1) / fps, 4),
                "object_name": "none",
                "hand_used": "none",
                "hands": [],
                "description": "negative control",
            }
        ],
        "frames": frames,
    }


def _evaluate_negative_case(mode: str, tmp_root: Path) -> Dict[str, Any]:
    ep_dir = tmp_root / f"neg_{mode}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    ann_path = ep_dir / "annotations.json"
    ann_data = _build_synthetic_annotations(mode)
    ann_path.write_text(json.dumps(ann_data, indent=2), encoding="utf-8")

    results: Dict[str, Any] = {"mode": mode, "checks_run": 0, "falsely_accepted_by": []}

    # 1. Test AnnotatedEpisode + multi-embodiment retargeter / hardware audit if present
    try:
        mod_umr = importlib.import_module("src.retargeting.universal_multi_retargeter")
        if hasattr(mod_umr, "compile_all_embodiments_for_episode"):
            results["checks_run"] += 1
            hw_rep = mod_umr.compile_all_embodiments_for_episode(ann_path, ep_dir, render_multi_video=False)
            for rname, rinfo in (hw_rep or {}).items():
                audit = rinfo.get("hardware_audit", {})
                verdict = str(audit.get("verdict", ""))
                score = float(audit.get("hardware_readiness_score", 0.0))
                ik_conv = float(audit.get("ik_convergence_pct", audit.get("reachable_pct", 100.0)))
                if "PRODUCTION_READY" in verdict or "PASS" in verdict or score >= 80.0:
                    results["falsely_accepted_by"].append(
                        f"universal_multi_retargeter[{rname}]: score={score}, verdict={verdict!r}, ik_conv={ik_conv}"
                    )
    except Exception as exc:
        results["umr_error"] = str(exc)

    # 2. Test LeRobot exporter + action signal validity on negative episode
    try:
        from src.datatypes import AnnotatedEpisode
        from src.lerobot_exporter import export_to_lerobot
        ep_obj = AnnotatedEpisode.load_json(str(ann_path))
        ep_obj.save_json(str(ann_path))
        export_to_lerobot(f"neg_{mode}", str(tmp_root), export_mode="human")
        pq_files = list((ep_dir / "lerobot_v3" / "data").rglob("*.parquet"))
        if pq_files:
            import pandas as pd
            df = pd.read_parquet(pq_files[0])
            actions = np.stack(df["action"].to_numpy()).astype(np.float64)
            act_var = float(np.max(np.var(actions, axis=0)))
            has_valid_col = "observation.hand_valid" in df.columns
            if mode == "zero_hands":
                results["checks_run"] += 1
                if not has_valid_col:
                    results["falsely_accepted_by"].append(
                        "lerobot_exporter[zero_hands]: missing observation.hand_valid mask on zero-hand episode"
                    )
                else:
                    valid_vals = np.stack(df["observation.hand_valid"].to_numpy())
                    if np.any(valid_vals != 0):
                        results["falsely_accepted_by"].append(
                            f"lerobot_exporter[zero_hands]: observation.hand_valid has non-zero entries ({np.sum(valid_vals)})"
                        )
            elif mode == "static_hands":
                results["checks_run"] += 1
                if act_var > 1e-5:
                    results["falsely_accepted_by"].append(
                        f"lerobot_exporter[static_hands]: non-zero action variance ({act_var:.6e}) on completely static hands"
                    )
    except Exception as exc:
        results["exporter_note"] = str(exc)

    # 3. Test centific_roboverify_engine if present
    try:
        mod_ver = importlib.import_module("src.verification.centific_roboverify_engine")
        if hasattr(mod_ver, "certify_episode_5_layer_funnel"):
            from src.datatypes import AnnotatedEpisode
            ep_obj = AnnotatedEpisode.load_json(str(ann_path))
            if hasattr( importlib.import_module("src.wam_engine"), "compile_wam_for_episode"):
                importlib.import_module("src.wam_engine").compile_wam_for_episode(ep_obj, ep_dir)
            results["checks_run"] += 1
            cert = mod_ver.certify_episode_5_layer_funnel(ep_obj, ep_dir)
            verdict = str(cert.get("overall_qc_verdict", ""))
            if "GOLD" in verdict or "PASS" in verdict or "CERTIFIED" in verdict:
                results["falsely_accepted_by"].append(
                    f"centific_roboverify_engine: overall_qc_verdict={verdict!r} on {mode}"
                )
    except Exception as exc:
        results["verifier_note"] = str(exc)

    results["passed"] = (results["checks_run"] > 0) and (len(results["falsely_accepted_by"]) == 0)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate 02: Negative Controls Verification")
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    print("=== G02: NEGATIVE CONTROLS (ZERO HANDS, STATIC HANDS, RANDOM NOISE) ===")
    modes = ["zero_hands", "static_hands", "random_noise"]
    reports = []

    with tempfile.TemporaryDirectory(prefix="ego_g02_neg_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for m in modes:
            rep = _evaluate_negative_case(m, tmp_root)
            reports.append(rep)
            status = "PASS" if rep["passed"] else "FAIL"
            print(f"  [{status}] Negative Control `{m}` (checks_run={rep['checks_run']}):")
            for fa in rep["falsely_accepted_by"]:
                print(f"      -> FALSE ACCEPTANCE: {fa}")

    all_passed = all(r["passed"] for r in reports)
    print(f"G02 VERDICT: {'PASS' if all_passed else 'FAIL'}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "gate": "g02_negative_controls",
            "status": "PASS" if all_passed else "FAIL",
            "cases": reports,
        }, indent=2) + "\n", encoding="utf-8")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
