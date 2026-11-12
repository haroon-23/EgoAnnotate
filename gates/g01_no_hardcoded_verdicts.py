#!/usr/bin/env python3
"""Gate 01: Static AST + Regex scanner for hardcoded verdicts, magic scores, and metric overwrites."""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

FORBIDDEN_STRING_PATTERNS = [
    (re.compile(r"\bPRODUCTION_READY\b"), "Hardcoded PRODUCTION_READY string"),
    (re.compile(r"\bCERTIFIED_PRODUCTION_GOLD\b"), "Hardcoded CERTIFIED_PRODUCTION_GOLD string"),
    (re.compile(r"\bCERTIFIED_GOLD\b"), "Hardcoded CERTIFIED_GOLD string"),
    (re.compile(r"\bREADY\s+FOR\s+DOOZY\b", re.IGNORECASE), "Hardcoded READY FOR DOOZY string"),
    (re.compile(r"\bALL\s+CHECKS\s+PASSED\b", re.IGNORECASE), "Unconditional ALL CHECKS PASSED string"),
    (re.compile(r"\bCERTIFIED_TOP_INFLUENCE\b"), "Hardcoded CERTIFIED_TOP_INFLUENCE string"),
]

FORBIDDEN_MAGIC_NUMBERS = [
    (re.compile(r"(?<![0-9.])99\.8(?![0-9])"), "Hardcoded magic score 99.8"),
    (re.compile(r"\+\s*12\.4(?![0-9])"), "Hardcoded +12.4 px reprojection offset"),
    (re.compile(r"max\(\s*0\.84\s*,"), "Hardcoded max(0.84, ...) correlation clamp"),
    (re.compile(r"max\([^)]*,\s*0\.912\s*\)"), "Hardcoded 0.912 IoU clamp"),
    (re.compile(r"max\(\s*r2_idm\s*,\s*0\.82\s*\)"), "Hardcoded 0.82 R2 clamp"),
    (re.compile(r"weld_slip_mm['\"]?\s*:\s*0\.\d+"), "Hardcoded weld_slip_mm constant in dict"),
]

METRIC_KEYWORDS = {
    "verdict", "score", "readiness", "passed", "self_collision",
    "teleport", "slip", "reachable", "ik_success", "status",
}


def _extract_names(node: ast.AST) -> Set[str]:
    names: Set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            names.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            names.add(sub.attr)
        elif isinstance(sub, ast.Subscript):
            if isinstance(sub.slice, ast.Constant) and isinstance(sub.slice.value, str):
                names.add(sub.slice.value)
    return names


def _is_constant_literal(node: ast.AST) -> Tuple[bool, str]:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) and node.value is True:
            return True, "True"
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            if float(node.value) in (0.0, 99.8, 99.9, 100.0, 0.38, 0.42):
                return True, repr(node.value)
        if isinstance(node.value, str):
            up = node.value.upper()
            if any(w in up for w in ("READY", "PASS", "GOLD", "CERTIFIED", "GOOD")):
                return True, repr(node.value)
    return False, ""


def scan_file(py_path: Path) -> List[Dict[str, object]]:
    hits: List[Dict[str, object]] = []
    text = py_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for regex, reason in FORBIDDEN_STRING_PATTERNS + FORBIDDEN_MAGIC_NUMBERS:
            if regex.search(line):
                hits.append({
                    "file": str(py_path),
                    "line": line_no,
                    "rule": "REGEX_FORBIDDEN_LITERAL",
                    "detail": f"{reason}: `{stripped}`",
                })

    try:
        tree = ast.parse(text, filename=str(py_path))
    except SyntaxError as exc:
        hits.append({
            "file": str(py_path),
            "line": exc.lineno or 1,
            "rule": "SYNTAX_ERROR",
            "detail": str(exc),
        })
        return hits

    for node in ast.walk(tree):
        # 1. Detect `if` statements that test a metric X and overwrite that same metric X inside the body
        if isinstance(node, ast.If):
            cond_names = {n.lower() for n in _extract_names(node.test)}
            cond_metrics = {
                m for m in cond_names
                if any(k in m for k in ("collision", "slip", "teleport", "err", "res", "score", "reach"))
            }
            if cond_metrics:
                for stmt in ast.walk(node):
                    if isinstance(stmt, ast.Assign):
                        for tgt in stmt.targets:
                            tgt_names = {n.lower() for n in _extract_names(tgt)}
                            overlap = cond_metrics.intersection(tgt_names)
                            if overlap:
                                hits.append({
                                    "file": str(py_path),
                                    "line": getattr(stmt, "lineno", node.lineno),
                                    "rule": "AST_METRIC_SELF_OVERWRITE_IN_IF",
                                    "detail": f"Metric(s) {sorted(overlap)} tested in `if` condition and overwritten in body",
                                })

        # 2. Detect direct constant assignments to dict keys or attributes representing verdicts/scores/collisions
        if isinstance(node, ast.Assign):
            is_const, const_repr = _is_constant_literal(node.value)
            if not is_const:
                continue
            for tgt in node.targets:
                tgt_names = _extract_names(tgt)
                for tname in tgt_names:
                    tl = tname.lower()
                    if any(k in tl for k in ("self_collision_events", "self_collision_frames", "teleport_jumps_detected", "hardware_readiness_score", "overall_qc_verdict")):
                        hits.append({
                            "file": str(py_path),
                            "line": getattr(node, "lineno", 1),
                            "rule": "AST_HARDCODED_METRIC_ASSIGNMENT",
                            "detail": f"Target `{tname}` assigned constant literal {const_repr}",
                        })

    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate 01: Scan for hardcoded verdicts and metric overwrites.")
    parser.add_argument("--roots", nargs="+", default=["src", "scripts"], help="Directories to scan")
    parser.add_argument("--json-out", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    all_hits: List[Dict[str, object]] = []
    scanned_files = 0

    for r in args.roots:
        rp = Path(r)
        if not rp.exists():
            continue
        for py_file in sorted(rp.rglob("*.py")):
            scanned_files += 1
            all_hits.extend(scan_file(py_file))

    print(f"=== G01: NO HARDCODED VERDICTS SCAN ===")
    print(f"Scanned Python files: {scanned_files}")
    print(f"Total violations found: {len(all_hits)}")
    for h in all_hits:
        print(f"  [HIT] {h['file']}:{h['line']} [{h['rule']}] -> {h['detail']}")

    passed = len(all_hits) == 0
    print(f"G01 VERDICT: {'PASS' if passed else 'FAIL'}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "gate": "g01_no_hardcoded_verdicts",
            "status": "PASS" if passed else "FAIL",
            "scanned_files": scanned_files,
            "violation_count": len(all_hits),
            "violations": all_hits,
        }, indent=2) + "\n", encoding="utf-8")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
