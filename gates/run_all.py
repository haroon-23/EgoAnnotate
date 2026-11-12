#!/usr/bin/env python3
"""Master Gate Runner: executes g01..g12, hashes all inputs, writes gates_report.json, and exits non-zero on any FAIL."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_input_hashes(ep_dir: Path, gates_dir: Path, extra_files: List[Path]) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for p in sorted(gates_dir.glob("*.py")):
        hashes[str(p)] = sha256_file(p)
    if ep_dir.exists():
        for p in sorted(ep_dir.rglob("*")):
            if p.is_file() and p.name != "gates_report.json":
                hashes[str(p)] = sha256_file(p)
    for p in extra_files:
        if p.exists() and p.is_file():
            hashes[str(p)] = sha256_file(p)
    return hashes


def get_git_commit() -> str:
    res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return res.stdout.strip() if res.returncode == 0 else "UNKNOWN"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all 12 verification gates and write gates_report.json")
    parser.add_argument("episode_dir", type=str, help="Target episode output dir (e.g. data/output/whatsapp_video)")
    parser.add_argument("--labels-csv", type=str, default="data/chirality_labels.csv")
    parser.add_argument("--max-template-fraction", type=float, default=1.0)
    parser.add_argument("--min-segment-sec", type=float, default=0.50)
    parser.add_argument("--report-out", type=str, default="gates_report.json")
    args = parser.parse_args()

    gates_dir = Path(__file__).resolve().parent
    ep_dir = Path(args.episode_dir)

    gate_specs = [
        ("g01_no_hardcoded_verdicts", [str(gates_dir / "g01_no_hardcoded_verdicts.py")]),
        ("g02_negative_controls", [str(gates_dir / "g02_negative_controls.py")]),
        ("g03_action_signal", [str(gates_dir / "g03_action_signal.py"), str(ep_dir)]),
        ("g04_layout_consistency", [str(gates_dir / "g04_layout_consistency.py"), str(ep_dir)]),
        ("g05_chirality", [str(gates_dir / "g05_chirality.py"), str(ep_dir), "--labels-csv", args.labels_csv]),
        ("g06_lerobot_v3", [str(gates_dir / "g06_lerobot_v3.py"), str(ep_dir)]),
        ("g07_hdf5_honesty", [str(gates_dir / "g07_hdf5_honesty.py"), str(ep_dir)]),
        ("g08_retarget_independent", [str(gates_dir / "g08_retarget_independent.py"), str(ep_dir)]),
        ("g09_physics_replay", [str(gates_dir / "g09_physics_replay.py"), str(ep_dir)]),
        ("g10_privacy", [str(gates_dir / "g10_privacy.py"), str(ep_dir)]),
        ("g11_captions", [
            str(gates_dir / "g11_captions.py"), str(ep_dir),
            "--max-template-fraction", str(args.max_template_fraction),
            "--min-segment-sec", str(args.min_segment_sec),
        ]),
        ("g12_tests_hygiene", [str(gates_dir / "g12_tests_hygiene.py")]),
    ]

    gate_results: Dict[str, Any] = {}
    any_failed = False

    with tempfile.TemporaryDirectory(prefix="ego_gates_json_") as tmp_jdir:
        for g_name, cmd_args in gate_specs:
            j_out = Path(tmp_jdir) / f"{g_name}.json"
            full_cmd = [sys.executable] + cmd_args + ["--json-out", str(j_out)]
            res = subprocess.run(full_cmd, capture_output=True, text=True)
            print(res.stdout)
            if res.stderr:
                print(res.stderr, file=sys.stderr)

            g_payload = json.loads(j_out.read_text(encoding="utf-8")) if j_out.exists() else {}
            status = "PASS" if res.returncode == 0 else "FAIL"
            if status != "PASS":
                any_failed = True
            gate_results[g_name] = {
                "status": status,
                "returncode": res.returncode,
                "details": g_payload,
            }

    input_hashes = collect_input_hashes(ep_dir, gates_dir, [Path(args.labels_csv)])
    overall_status = "PASS" if not any_failed else "FAIL"

    report_doc = {
        "schema_version": "ego_annotate_gates_v1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "episode_dir": str(ep_dir),
        "overall_status": overall_status,
        "gates": gate_results,
        "input_sha256": input_hashes,
    }

    out_path = Path(args.report_out)
    out_path.write_text(json.dumps(report_doc, indent=2) + "\n", encoding="utf-8")
    if ep_dir.exists():
        (ep_dir / "gates_report.json").write_text(json.dumps(report_doc, indent=2) + "\n", encoding="utf-8")

    print("\n============================================================")
    print(f"  GATES SUMMARY FOR: {ep_dir}")
    print("============================================================")
    for g_name, g_info in gate_results.items():
        print(f"  {g_name:28s} : {g_info['status']}")
    print("------------------------------------------------------------")
    print(f"  OVERALL GATES VERDICT        : {overall_status}")
    print(f"  REPORT WRITTEN TO            : {out_path}")
    print("============================================================")

    return 0 if not any_failed else 1


if __name__ == "__main__":
    sys.exit(main())
