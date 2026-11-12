#!/usr/bin/env python3
"""Gate 12: Pytest full suite execution (0 failures allowed) & signal-level test count audit."""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import List

SIGNAL_ASSERTION_KEYWORDS = re.compile(
    r"\b(var|std|residual|hand_valid|reproj|convergence|negative|chirality|delta|tolerance)\b",
    re.IGNORECASE,
)


def _count_signal_tests(tests_dir: Path) -> int:
    count = 0
    for py_f in sorted(tests_dir.rglob("test_*.py")):
        txt = py_f.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(txt, filename=str(py_f))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                fn_src = ast.get_source_segment(txt, node) or ""
                has_assert = any(isinstance(sub, ast.Assert) for sub in ast.walk(node))
                if has_assert and SIGNAL_ASSERTION_KEYWORDS.search(fn_src):
                    count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate 12: Pytest Execution & Signal Test Hygiene")
    parser.add_argument("--min-signal-tests", type=int, default=12)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    print("=== G12: PYTEST SUITE & SIGNAL-LEVEL TEST HYGIENE GATE ===")
    failures: List[str] = []

    res = subprocess.run([sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True)
    print(res.stdout.strip())
    if res.stderr.strip():
        print(res.stderr.strip())

    if res.returncode != 0:
        failures.append(f"pytest exited with non-zero code {res.returncode}")

    sig_count = _count_signal_tests(Path("tests"))
    print(f"  Signal-level test functions asserting on variance/validity/residual/delta: {sig_count} (min={args.min_signal_tests})")
    if sig_count < args.min_signal_tests:
        failures.append(f"Signal-level test count {sig_count} < required minimum {args.min_signal_tests}")

    passed = len(failures) == 0
    for f_msg in failures:
        print(f"  [FAIL] {f_msg}")
    print(f"G12 VERDICT: {'PASS' if passed else 'FAIL'}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "gate": "g12_tests_hygiene",
            "status": "PASS" if passed else "FAIL",
            "pytest_returncode": res.returncode,
            "signal_test_count": sig_count,
            "failures": failures,
        }, indent=2) + "\n", encoding="utf-8")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
