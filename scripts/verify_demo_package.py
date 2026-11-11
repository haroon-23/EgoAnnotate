from __future__ import annotations
#!/usr/bin/env python3
"""Verify Demo Package Integrity strictly from `gates_report.json` (Rule R2 & Mission 1h).

Never prints unconditional pass strings; runs or loads `gates_report.json` and exits
non-zero if any gate in `g01..g12` did not pass.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def verify_from_gates_report(episode_dir: Path, run_gates: bool = False) -> int:
    report_path = episode_dir / "gates_report.json"
    if not report_path.exists() and Path("gates_report.json").exists():
        report_path = Path("gates_report.json")

    if run_gates or not report_path.exists():
        res = subprocess.run(
            [sys.executable, "gates/run_all.py", str(episode_dir), "--report-out", str(episode_dir / "gates_report.json")],
            capture_output=True, text=True,
        )
        print(res.stdout)
        report_path = episode_dir / "gates_report.json"

    if not report_path.exists():
        print(f"[FAIL] Missing {report_path}")
        return 1

    rep = json.loads(report_path.read_text(encoding="utf-8"))
    gates = rep.get("gates", {})
    failed_gates = [k for k, v in gates.items() if v.get("status") != "PASS"]
    overall = rep.get("overall_status", "FAIL")

    print(f"Demo Package Verification for `{episode_dir}` (Report: {report_path}):")
    for k, v in gates.items():
        print(f"  - {k:28s}: {v.get('status')}")

    if overall == "PASS" and not failed_gates and len(gates) == 12:
        print("DEMO PACKAGE VERDICT: PASS (12/12 Read-Only Gates Verified)")
        return 0

    print(f"DEMO PACKAGE VERDICT: FAIL (Failed gates: {failed_gates})")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify EgoAnnotate demo package from gates_report.json")
    parser.add_argument("episode_dir", nargs="?", default="data/output/test_10s")
    parser.add_argument("--run-gates", action="store_true", help="Re-run gates/run_all.py before checking")
    args = parser.parse_args()
    return verify_from_gates_report(Path(args.episode_dir), run_gates=args.run_gates)


if __name__ == "__main__":
    sys.exit(main())
