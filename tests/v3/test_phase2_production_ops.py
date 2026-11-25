"""Phase 2 verification: CLI, Python API, SQLite queue resume, FastAPI worker, offline model check & full-video QA."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.egoannotate.api import export_qa_artifacts
from src.egoannotate.queue_worker import JobQueueStore, create_worker_app

FULL_SAMPLE_VIDEO = Path("data/sample_raw/ad521ca5-0017-475b-9dd8-1fdb423a66b7.mp4")
CERTIFIED_RUN_DIR = Path("data/v3_certified_full_run")


def test_cli_audit_licenses_and_offline_model_manifest(tmp_path: Path) -> None:
    res = subprocess.run(
        [sys.executable, "-m", "src.egoannotate.cli", "audit-licenses"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(res.stdout)
    backends = data["backends"]
    assert len(backends) >= 8
    default_ok = [b for b in backends if b["name"] != "hamer_mano_optional"]
    assert all(b["commercial_ok"] is True and b["is_paid_api"] is False for b in default_ok)

    dl = subprocess.run(
        [
            sys.executable,
            "scripts/download_models.py",
            "--profile",
            "cpu_smoke",
            "--cache-dir",
            str(tmp_path / "models"),
            "--offline",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    dl_json = json.loads(dl.stdout)
    assert dl_json["profile"] == "cpu_smoke"
    assert all(m["status"] == "bundled_ready" for m in dl_json["models"])


def test_sqlite_job_queue_dedup_and_fastapi_worker_api(tmp_path: Path) -> None:
    if not FULL_SAMPLE_VIDEO.exists():
        pytest.skip("Full sample video not available locally")

    db_path = tmp_path / "jobs.db"
    store = JobQueueStore(db_path)

    j1 = store.enqueue(FULL_SAMPLE_VIDEO, tmp_path / "out1", profile="cpu_smoke")
    assert j1["status"] == "PENDING"
    assert j1["deduplicated"] is False

    j2 = store.enqueue(FULL_SAMPLE_VIDEO, tmp_path / "out2", profile="cpu_smoke")
    assert j2["deduplicated"] is True
    assert j2["job_id"] == j1["job_id"]

    app = create_worker_app(db_path)
    client = TestClient(app)

    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    r_lic = client.get("/licenses")
    assert r_lic.status_code == 200
    assert len(r_lic.json()["backends"]) >= 8

    r_list = client.get("/jobs")
    assert r_list.status_code == 200
    assert len(r_list.json()["jobs"]) == 1

    r_get = client.get(f"/jobs/{j1['job_id']}")
    assert r_get.status_code == 200
    assert r_get.json()["video_sha256"] == j1["video_sha256"]


def test_full_5400_frame_qa_report_3d_viewer_and_overlay() -> None:
    if not (CERTIFIED_RUN_DIR / "parquet_json" / "episode_ir.json").exists():
        pytest.skip("Phase 1D certified 5400-frame run directory not present")

    qa = export_qa_artifacts(CERTIFIED_RUN_DIR, render_overlay=True)
    qa_html = Path(qa["qa_report_html"])
    viewer_html = Path(qa["viewer_3d_html"])
    overlay_mp4 = Path(qa["overlay"]["qa_overlay_mp4"])

    assert qa_html.exists() and "Robot Readiness" in qa_html.read_text(encoding="utf-8")
    assert viewer_html.exists() and "EgoAnnotate v3 World Frame" in viewer_html.read_text(encoding="utf-8")
    assert overlay_mp4.exists() and overlay_mp4.stat().st_size > 100_000
    assert qa["overlay"]["frames_rendered"] >= 5398

    print(
        f"\n[PHASE 2 FULL-VIDEO QA ARTIFACTS] "
        f"qa_report={qa_html.stat().st_size/1024:.1f}KB, "
        f"viewer_3d={viewer_html.stat().st_size/1024:.1f}KB, "
        f"qa_overlay={overlay_mp4.stat().st_size/1e6:.2f}MB ({qa['overlay']['frames_rendered']} frames)"
    )
