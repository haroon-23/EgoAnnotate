"""Unified commercial CLI (`egoannotate`) for single-video, batch queue, REST worker, QA, and model ops."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from src.egoannotate.api import annotate_video, export_qa_artifacts, verify_policy_training
from src.egoannotate.observability import configure_json_logging
from src.egoannotate.queue_worker import JobQueueStore, create_worker_app
from src.egoannotate.registry import REGISTRY

_PLUGIN_MODULES = (
    "src.egoannotate.geometry_world",
    "src.egoannotate.perception_hands",
    "src.egoannotate.perception_objects",
    "src.egoannotate.semantics",
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="egoannotate",
        description="EgoAnnotate v3: Commercial egocentric video to robot-ready VLA data pipeline.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Process a complete video end-to-end (frames 0..N-1).")
    p_run.add_argument("video", type=Path)
    p_run.add_argument("--out", type=Path, required=True)
    p_run.add_argument("--profile", default="auto", choices=["auto", "cpu_smoke", "gpu_prod"])
    p_run.add_argument("--config", type=Path, default=None)
    p_run.add_argument("--tactile", type=Path, default=None, help="Optional hardware tactile CSV/HDF5 stream.")
    p_run.add_argument("--calibration", type=Path, default=None, help="Optional calibrated camera intrinsics YAML.")
    p_run.add_argument("--episode-id", default=None)
    p_run.add_argument("--no-qa", action="store_true")

    p_batch = sub.add_parser("batch", help="Enqueue a directory of videos into SQLite queue and process.")
    p_batch.add_argument("input_dir", type=Path)
    p_batch.add_argument("--out", type=Path, required=True)
    p_batch.add_argument("--db", type=Path, default=Path("egoannotate_jobs.db"))
    p_batch.add_argument("--profile", default="auto")
    p_batch.add_argument("--workers", type=int, default=1)
    p_batch.add_argument("--enqueue-only", action="store_true")

    p_worker = sub.add_parser("worker", help="Launch FastAPI REST worker service.")
    p_worker.add_argument("--db", type=Path, default=Path("egoannotate_jobs.db"))
    p_worker.add_argument("--host", default="127.0.0.1")
    p_worker.add_argument("--port", type=int, default=8080)

    p_qa = sub.add_parser("export-qa", help="Generate qa_report.html, viewer_3d.html, and full qa_overlay.mp4.")
    p_qa.add_argument("run_dir", type=Path)
    p_qa.add_argument("--no-overlay", action="store_true")

    p_train = sub.add_parser("train-smoke", help="Run PyTorch action-chunk policy smoke training on LeRobot v3 export.")
    p_train.add_argument("lerobot_dir", type=Path)
    p_train.add_argument("--epochs", type=int, default=6)

    sub.add_parser("audit-licenses", help="Print JSON commercial license audit of all registered plugins.")
    return p


def main(argv: list[str] | None = None) -> int:
    configure_json_logging()
    args = build_parser().parse_args(argv)

    if args.command == "run":
        report = annotate_video(
            args.video,
            args.out,
            profile=args.profile,
            config_path=args.config,
            external_tactile_path=args.tactile,
            calibration_yaml=args.calibration,
            episode_id=args.episode_id,
            generate_qa_artifacts=not args.no_qa,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "batch":
        store = JobQueueStore(args.db)
        mp4s = sorted(args.input_dir.glob("*.mp4"))
        enqueued = [store.enqueue(v, args.out / v.stem, profile=args.profile) for v in mp4s]
        if args.enqueue_only:
            print(json.dumps({"enqueued": enqueued}, indent=2))
            return 0
        done = store.run_pending_batch(max_workers=args.workers)
        print(json.dumps({"processed": done}, indent=2))
        return 0

    if args.command == "worker":
        import uvicorn

        app = create_worker_app(args.db)
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    if args.command == "export-qa":
        res = export_qa_artifacts(args.run_dir, render_overlay=not args.no_overlay)
        print(json.dumps(res, indent=2))
        return 0

    if args.command == "train-smoke":
        res = verify_policy_training(args.lerobot_dir, epochs=args.epochs)
        print(json.dumps(res, indent=2))
        return 0

    if args.command == "audit-licenses":
        for mod_name in _PLUGIN_MODULES:
            importlib.import_module(mod_name)
        print(json.dumps({"backends": REGISTRY.audit_table()}, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
