"""Multi-GPU persistent batch orchestrator with 6-state ledger and DLQ retry."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import sqlite3
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

LEDGER_STATES = (
    "PENDING",
    "PERCEPTION",
    "SEMANTICS",
    "RETARGETING",
    "VERIFIED",
    "SHARDED",
    "FAILED_DLQ",
)


def compute_video_manifest_id(video_path: Path) -> str:
    """Compute deterministic sha256 manifest ID for a video file."""
    h = hashlib.sha256()
    with video_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def detect_worker_devices(max_gpus: int | None = None) -> list[str]:
    """Return available CUDA device strings (cuda:0..N) or ['cpu']."""
    if not torch.cuda.is_available():
        return ["cpu"]
    count = int(torch.cuda.device_count())
    if max_gpus is not None:
        count = min(count, max(1, int(max_gpus)))
    return [f"cuda:{i}" for i in range(max(1, count))]


class BatchStateLedger:
    """WAL-mode SQLite 6-state checkpoint ledger for fault-tolerant large-scale batches."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS video_ledger (
                    manifest_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    video_path TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    state TEXT NOT NULL,
                    gpu_device TEXT,
                    batch_scale REAL NOT NULL DEFAULT 1.0,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    quality_score REAL,
                    verdict TEXT,
                    wall_s REAL,
                    error_message TEXT,
                    updated_utc TEXT NOT NULL
                )
                """
            )

    def register_video(
        self,
        manifest_id: str,
        episode_id: str,
        video_path: Path,
        output_dir: Path,
    ) -> dict[str, Any]:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO video_ledger (
                    manifest_id, episode_id, video_path, output_dir, state, updated_utc
                ) VALUES (?, ?, ?, ?, 'PENDING', ?)
                """,
                (manifest_id, episode_id, str(video_path), str(output_dir), now),
            )
            row = conn.execute(
                "SELECT * FROM video_ledger WHERE manifest_id = ?", (manifest_id,)
            ).fetchone()
        return dict(row) if row is not None else {}

    def transition(
        self,
        manifest_id: str,
        state: str,
        *,
        gpu_device: str | None = None,
        batch_scale: float | None = None,
        increment_attempts: bool = False,
        quality_score: float | None = None,
        verdict: str | None = None,
        wall_s: float | None = None,
        error_message: str | None = None,
    ) -> None:
        if state not in LEDGER_STATES:
            raise ValueError(f"Unsupported ledger state: {state}")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT attempts, batch_scale, gpu_device FROM video_ledger WHERE manifest_id = ?",
                (manifest_id,),
            ).fetchone()
            attempts = int(row["attempts"]) if row is not None else 0
            cur_scale = float(row["batch_scale"]) if row is not None else 1.0
            cur_dev = row["gpu_device"] if row is not None else None
            if increment_attempts:
                attempts += 1
            conn.execute(
                """
                UPDATE video_ledger
                SET state = ?, gpu_device = ?, batch_scale = ?, attempts = ?,
                    quality_score = COALESCE(?, quality_score),
                    verdict = COALESCE(?, verdict),
                    wall_s = COALESCE(?, wall_s),
                    error_message = ?,
                    updated_utc = ?
                WHERE manifest_id = ?
                """,
                (
                    state,
                    gpu_device if gpu_device is not None else cur_dev,
                    float(batch_scale) if batch_scale is not None else cur_scale,
                    attempts,
                    quality_score,
                    verdict,
                    wall_s,
                    error_message,
                    _utc_now(),
                    manifest_id,
                ),
            )

    def get(self, manifest_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM video_ledger WHERE manifest_id = ?", (manifest_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def list_all(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM video_ledger ORDER BY episode_id").fetchall()
        return [dict(r) for r in rows]


class DeadLetterQueue:
    """Append-only JSONL dead-letter queue for quarantined clips."""

    def __init__(self, dlq_path: Path) -> None:
        self.dlq_path = Path(dlq_path)
        self.dlq_path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        manifest_id: str,
        episode_id: str,
        video_path: Path,
        error: str,
        trace_tail: str,
        batch_scale: float,
    ) -> None:
        entry = {
            "utc": _utc_now(),
            "manifest_id": manifest_id,
            "episode_id": episode_id,
            "video_path": str(video_path),
            "batch_scale": float(batch_scale),
            "error": str(error),
            "trace_tail": str(trace_tail)[-1200:],
        }
        with self.dlq_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")


@dataclass(frozen=True)
class BatchWorkItem:
    manifest_id: str
    episode_id: str
    video_path: str
    output_dir: str


def _summarize_report(vp: Path, rep: dict[str, Any], wall_s: float) -> dict[str, Any]:
    gpu = rep.get("gpu_enrichment", {})
    return {
        "status": "OK",
        "file": vp.name,
        "frames": f"{rep['video']['processed_frames']}/{rep['video']['total_frames']}",
        "duration_s": rep["video"]["duration_s"],
        "quality_score": rep["quality"]["robot_readiness_score"],
        "verdict": rep["quality"]["verdict"],
        "flags": rep["quality"]["flags"],
        "privacy_pass": rep["privacy_audit"]["privacy_pass"],
        "faces_blurred": rep["privacy_audit"]["frames_blurred"],
        "unblurred_faces": rep["privacy_audit"]["verified_unblurred_faces"],
        "dino_detections": gpu.get("grounding_dino", {}).get("total_detections", 0),
        "depth_median": gpu.get("depth_anything_v2_small", {}).get("median_depth_mean"),
        "vlm_captions": gpu.get("local_vlm", {}).get("sample_captions", [])[:4],
        "retargeting": {
            k: {"feas": v["overall_feasibility"], "fk_mm": v["mean_fk_pos_error_mm"]}
            for k, v in rep["retargeting"].items()
        },
        "wall_s": round(float(wall_s), 1),
    }


def _gpu_worker_loop(
    device_str: str,
    profile: str,
    ledger_path: str,
    dlq_path: str,
    items: list[BatchWorkItem],
    result_queue: Any,
) -> None:
    if device_str.startswith("cuda:"):
        gpu_index = device_str.split(":", 1)[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index

    from src.egoannotate.config import load_config
    from src.egoannotate.pipeline_v3 import EgoAnnotateV3Pipeline

    ledger = BatchStateLedger(Path(ledger_path))
    dlq = DeadLetterQueue(Path(dlq_path))

    overrides: dict[str, Any] = {}
    if device_str == "cpu":
        overrides = {"runtime": {"device": "cpu"}}
    cfg = load_config(profile=profile, overrides=overrides)
    pipe = EgoAnnotateV3Pipeline(cfg=cfg)

    for item in items:
        vp = Path(item.video_path)
        ep_out = Path(item.output_dir)
        rep_file = ep_out / "run_report.json"

        existing = ledger.get(item.manifest_id)
        if (
            existing is not None
            and existing.get("state") in ("VERIFIED", "SHARDED")
            and rep_file.exists()
        ):
            rep = json.loads(rep_file.read_text(encoding="utf-8"))
            summary = _summarize_report(vp, rep, float(existing.get("wall_s") or 0.0))
            result_queue.put((item.episode_id, summary))
            continue

        completed = False
        last_exc = ""
        last_trace = ""
        for attempt_idx, batch_scale in enumerate((1.0, 0.5), start=1):
            os.environ["EGOANNOTATE_BATCH_SCALE"] = str(batch_scale)
            ledger.transition(
                item.manifest_id,
                "PERCEPTION",
                gpu_device=device_str,
                batch_scale=batch_scale,
                increment_attempts=True,
            )
            t0 = time.time()
            try:
                pipe.run_video(vp, ep_out, episode_id=item.episode_id, use_cache=True)
                wall_s = time.time() - t0
                rep = json.loads(rep_file.read_text(encoding="utf-8"))
                q_score = float(rep["quality"]["robot_readiness_score"])
                verdict = str(rep["quality"]["verdict"])
                ledger.transition(
                    item.manifest_id,
                    "VERIFIED",
                    gpu_device=device_str,
                    batch_scale=batch_scale,
                    quality_score=q_score,
                    verdict=verdict,
                    wall_s=wall_s,
                    error_message=None,
                )
                summary = _summarize_report(vp, rep, wall_s)
                result_queue.put((item.episode_id, summary))
                completed = True
                break
            except (
                RuntimeError,
                ValueError,
                OSError,
                KeyError,
                IndexError,
                TypeError,
                MemoryError,
                AssertionError,
                ArithmeticError,
                LookupError,
            ) as exc:
                last_exc = str(exc)
                last_trace = traceback.format_exc()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if attempt_idx == 2:
                    ledger.transition(
                        item.manifest_id,
                        "FAILED_DLQ",
                        gpu_device=device_str,
                        batch_scale=batch_scale,
                        error_message=last_exc,
                    )
                    dlq.record(
                        manifest_id=item.manifest_id,
                        episode_id=item.episode_id,
                        video_path=vp,
                        error=last_exc,
                        trace_tail=last_trace,
                        batch_scale=batch_scale,
                    )
        if not completed:
            result_queue.put(
                (
                    item.episode_id,
                    {
                        "status": "FAILED_DLQ",
                        "file": vp.name,
                        "error": last_exc,
                        "trace": last_trace[-600:],
                    },
                )
            )


class MultiGPUBatchOrchestrator:
    """Distribute videos across all GPUs with persistent models, ledger, and DLQ."""

    def __init__(
        self,
        out_root: Path,
        profile: str = "gpu_prod",
        max_gpus: int | None = None,
    ) -> None:
        self.out_root = Path(out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.profile = profile
        self.devices = detect_worker_devices(max_gpus=max_gpus)
        self.ledger = BatchStateLedger(self.out_root / "batch_ledger.sqlite3")
        self.dlq = DeadLetterQueue(self.out_root / "dlq.jsonl")
        self.summary_path = self.out_root / "batch_audit_summary.json"

    def run_videos(self, video_paths: list[Path]) -> dict[str, Any]:
        ordered = sorted(video_paths, key=lambda p: p.stat().st_size, reverse=True)
        work_items: list[BatchWorkItem] = []
        for vp in ordered:
            ep_id = vp.stem
            mid = compute_video_manifest_id(vp)
            ep_out = self.out_root / ep_id
            self.ledger.register_video(mid, ep_id, vp, ep_out)
            work_items.append(
                BatchWorkItem(
                    manifest_id=mid,
                    episode_id=ep_id,
                    video_path=str(vp),
                    output_dir=str(ep_out),
                )
            )

        buckets: list[list[BatchWorkItem]] = [[] for _ in self.devices]
        for idx, item in enumerate(work_items):
            buckets[idx % len(self.devices)].append(item)

        results: dict[str, Any] = {}
        if len(self.devices) == 1:
            local_q: list[tuple[str, dict[str, Any]]] = []

            class _LocalQueue:
                def put(self, val: tuple[str, dict[str, Any]]) -> None:
                    local_q.append(val)

            _gpu_worker_loop(
                self.devices[0],
                self.profile,
                str(self.ledger.db_path),
                str(self.dlq.dlq_path),
                buckets[0],
                _LocalQueue(),
            )
            for ep_id, summary in local_q:
                results[ep_id] = summary
        else:
            ctx = mp.get_context("spawn")
            result_q = ctx.Queue()
            procs = []
            for dev, bucket in zip(self.devices, buckets):
                if not bucket:
                    continue
                p = ctx.Process(
                    target=_gpu_worker_loop,
                    args=(
                        dev,
                        self.profile,
                        str(self.ledger.db_path),
                        str(self.dlq.dlq_path),
                        bucket,
                        result_q,
                    ),
                )
                p.start()
                procs.append(p)

            expected = len(work_items)
            received = 0
            while received < expected:
                ep_id, summary = result_q.get()
                results[ep_id] = summary
                received += 1
                self.summary_path.write_text(
                    json.dumps(dict(sorted(results.items())), indent=2),
                    encoding="utf-8",
                )
            for p in procs:
                p.join()

        ordered_results = dict(sorted(results.items()))
        self.summary_path.write_text(json.dumps(ordered_results, indent=2), encoding="utf-8")
        return ordered_results
