"""Hermetic test for scripts/hero_clip.py's refusal gate.

The refusal path runs before any heavy import, so it is testable via
subprocess without mujoco/pybullet/scipy.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "hero_clip.py"


def _run(report_payload, tmp_path):
    rep = tmp_path / "physics_verification.json"
    rep.write_text(json.dumps(report_payload))
    return subprocess.run(
        [sys.executable, str(SCRIPT),
         "--hdf5", str(tmp_path / "x.hdf5"),
         "--report", str(rep),
         "--video", str(tmp_path / "v.mp4"),
         "--urdf", str(tmp_path / "r.urdf"),
         "--out", str(tmp_path / "hero.mp4")],
        capture_output=True, text=True, timeout=60,
    )


def test_refuses_when_report_not_passed(tmp_path):
    r = _run({"episode_id": "ep", "passed": False,
              "checks": [{"name": "m1_tracking_err_rad", "passed": False,
                           "value": 0.2, "threshold": 0.15, "detail": ""}],
              "windows": [], "metrics": {}}, tmp_path)
    assert r.returncode == 2
    assert "REFUSES" in r.stderr
    assert "NOT physics-verified" in r.stderr


def test_refuses_when_report_missing(tmp_path):
    r = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--hdf5", "x", "--report", str(tmp_path / "nope.json"),
         "--video", "x", "--urdf", "x", "--out", str(tmp_path / "h.mp4")],
        capture_output=True, text=True, timeout=60)
    assert r.returncode == 2
    assert "REFUSES" in r.stderr
