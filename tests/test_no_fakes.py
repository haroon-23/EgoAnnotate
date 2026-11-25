"""CI hard gate: fail if mock/stub/placeholder/paid-API/non-commercial leaks appear in src/egoannotate/."""

from __future__ import annotations

import re
from pathlib import Path

V3_SRC = Path("src/egoannotate")

FORBIDDEN_PATTERNS = [
    (re.compile(r"\bgoogle\.genai\b"), "Paid Gemini API import forbidden in v3 core"),
    (re.compile(r"\bGEMINI_API_KEY\b"), "Paid Gemini API key reference forbidden in v3 core"),
    (re.compile(r"\bOPENAI_API_KEY\b"), "Paid OpenAI API key reference forbidden in v3 core"),
    (re.compile(r"\bANTHROPIC_API_KEY\b"), "Paid Anthropic API key reference forbidden in v3 core"),
    (re.compile(r"\bNotImplementedError\b"), "Stub NotImplementedError forbidden in shipped v3 core"),
    (re.compile(r"\b(TODO|FIXME|HACK|XXX)\b"), "Unresolved TODO/FIXME forbidden in shipped v3 core"),
    (re.compile(r"\bfake_trajectory\b|\bmock_output\b|\bdummy_pose\b", re.IGNORECASE), "Fabricated output helper forbidden"),
]


def test_no_fakes_or_paid_apis_in_v3_core() -> None:
    py_files = sorted(V3_SRC.rglob("*.py"))
    assert py_files, "No python modules found in src/egoannotate/"
    violations = []
    for py in py_files:
        text = py.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pat, reason in FORBIDDEN_PATTERNS:
                if pat.search(line):
                    violations.append(f"{py}:{lineno}: {reason} -> {line.strip()}")
    assert not violations, "Anti-fake/commercial violations:\n" + "\n".join(violations)
