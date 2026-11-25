"""Idempotent SQLite batch job queue & FastAPI REST worker for scalable commercial deployment."""

from __future__ import annotations

import importlib
import json
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.egoannotate.config import load_config
from src.egoannotate.pipeline_v3 import EgoAnnotateV3Pipeline
from src.egoannotate.registry import REGISTRY
from src.egoannotate.video_io import compute_file_sha256

_PLUGIN_MODULES = (
    "src.egoannotate.geometry_world",
    "src.egoannotate.perception_hands",
    "src.egoannotate.perception_objects",
    "src.egoannotate.semantics",
)


class JobQueueStore:
    """Crash-safe SQLite job queue supporting idempotent deduplication, retries, and resume."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    video_path TEXT NOT NULL,
                    video_sha256 TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    report_path TEXT,
                    error TEXT,
                    created_utc TEXT NOT NULL,
                    updated_utc TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_sha ON jobs(video_sha256, profile, status)")

    def enqueue(
        self,
        video_path: Path,
        output_dir: Path,
        profile: str = "auto",
    ) -> dict[str, Any]:
        v_path = Path(video_path).resolve()
        if not v_path.exists():
            raise FileNotFoundError(f"Video not found: {v_path}")
        sha = compute_file_sha256(v_path)
        now = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM jobs WHERE video_sha256 = ? AND profile = ? AND status IN ('PENDING','RUNNING','COMPLETED') ORDER BY created_utc DESC LIMIT 1",
                (sha, profile),
            ).fetchone()
            if existing is not None:
                row_d = dict(existing)
                row_d["deduplicated"] = True
                return row_d

            job_id = f"job_{uuid.uuid4().hex[:12]}"
            conn.execute(
                """
                INSERT INTO jobs (job_id, video_path, video_sha256, output_dir, profile, status, attempts, created_utc, updated_utc)
                VALUES (?, ?, ?, ?, ?, 'PENDING', 0, ?, ?)
                """,
                (job_id, str(v_path), sha, str(Path(output_dir).resolve()), profile, now, now),
            )
            created = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            assert created is not None
            row_d = dict(created)
            row_d["deduplicated"] = False
            return row_d

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            return dict(row) if row is not None else None

    def list_jobs(self, status: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM jobs WHERE status = ? ORDER BY created_utc DESC",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM jobs ORDER BY created_utc DESC").fetchall()
            return [dict(r) for r in rows]

    def update_status(
        self,
        job_id: str,
        status: str,
        *,
        report_path: str | None = None,
        error: str | None = None,
        inc_attempts: bool = False,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            if inc_attempts:
                conn.execute(
                    "UPDATE jobs SET status=?, attempts=attempts+1, report_path=COALESCE(?, report_path), error=?, updated_utc=? WHERE job_id=?",
                    (status, report_path, error, now, job_id),
                )
            else:
                conn.execute(
                    "UPDATE jobs SET status=?, report_path=COALESCE(?, report_path), error=?, updated_utc=? WHERE job_id=?",
                    (status, report_path, error, now, job_id),
                )

    def run_pending_batch(self, max_workers: int = 1, max_retries: int = 2) -> list[dict[str, Any]]:
        """Execute all PENDING/FAILED (under retry limit) jobs with bounded concurrency."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status = 'PENDING' OR (status = 'FAILED' AND attempts < ?) ORDER BY created_utc ASC",
                (max_retries,),
            ).fetchall()
        pending = [dict(r) for r in rows]
        if not pending:
            return []

        def _exec_one(job: dict[str, Any]) -> dict[str, Any]:
            jid = str(job["job_id"])
            self.update_status(jid, "RUNNING", inc_attempts=True)
            try:
                cfg = load_config(profile=str(job["profile"]))
                pipe = EgoAnnotateV3Pipeline(cfg=cfg)
                pipe.run_video(Path(job["video_path"]), Path(job["output_dir"]), episode_id=jid)
                rep_p = str(Path(job["output_dir"]) / "run_report.json")
                self.update_status(jid, "COMPLETED", report_path=rep_p, error=None)
            except (OSError, ValueError, RuntimeError) as exc:
                self.update_status(jid, "FAILED", error=str(exc))
            return self.get_job(jid) or job

        results: list[dict[str, Any]] = []
        if max_workers <= 1:
            for job in pending:
                results.append(_exec_one(job))
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futs = [pool.submit(_exec_one, j) for j in pending]
                for fut in as_completed(futs):
                    results.append(fut.result())
        return results


class SubmitJobRequest(BaseModel):
    video_path: str = Field(min_length=1)
    output_dir: str = Field(min_length=1)
    profile: str = "auto"


def create_worker_app(db_path: Path) -> FastAPI:
    """Build FastAPI REST control plane backed by SQLite JobQueueStore."""
    store = JobQueueStore(db_path)
    app = FastAPI(title="EgoAnnotate v3 Worker API", version="3.0.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": "3.0.0", "db": str(db_path)}

    @app.get("/licenses")
    def licenses() -> dict[str, Any]:
        for mod_name in _PLUGIN_MODULES:
            importlib.import_module(mod_name)
        return {"backends": REGISTRY.audit_table()}

    @app.post("/jobs")
    def submit_job(req: SubmitJobRequest) -> dict[str, Any]:
        try:
            return store.enqueue(Path(req.video_path), Path(req.output_dir), profile=req.profile)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/jobs")
    def list_all_jobs(status: str | None = None) -> dict[str, Any]:
        return {"jobs": store.list_jobs(status=status)}

    @app.get("/jobs/{job_id}")
    def get_one_job(job_id: str) -> dict[str, Any]:
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        if job.get("report_path") and Path(job["report_path"]).exists():
            job["report_summary"] = json.loads(Path(job["report_path"]).read_text(encoding="utf-8"))
        return job

    return app
