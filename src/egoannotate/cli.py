"""Unified commercial CLI (`egoannotate`) for single-video, multi-GPU batch, Ray, sharding & QA."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from src.egoannotate.api import annotate_video, export_qa_artifacts, verify_policy_training
from src.egoannotate.batch_orchestrator import MultiGPUBatchOrchestrator
from src.egoannotate.lerobot_sharding import pack_sharded_lerobot_v3
from src.egoannotate.observability import configure_json_logging
from src.egoannotate.policy_gate import validate_lerobot_v3_dataset
from src.egoannotate.queue_worker import JobQueueStore, create_worker_app
from src.egoannotate.ray_pipeline import RayStreamingOrchestrator
from src.egoannotate.registry import REGISTRY

_PLUGIN_MODULES = (
    "src.egoannotate.geometry_world",
    "src.egoannotate.gpu_backends_v4",
    "src.egoannotate.perception_hands",
    "src.egoannotate.perception_hands_rtmw",
    "src.egoannotate.perception_objects",
    "src.egoannotate.semantics",
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="egoannotate",
        description="EgoAnnotate v4: Commercial egocentric video to robot-ready VLA data pipeline.",
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

    p_bgpu = sub.add_parser(
        "batch-gpu",
        help="Run persistent multi-GPU worker pool with 6-state ledger, DLQ retry, and optional LeRobot v3 sharding.",
    )
    p_bgpu.add_argument("input_dir", type=Path)
    p_bgpu.add_argument("--out", type=Path, required=True)
    p_bgpu.add_argument("--profile", default="gpu_prod", choices=["auto", "cpu_smoke", "gpu_prod"])
    p_bgpu.add_argument("--max-gpus", type=int, default=None)
    p_bgpu.add_argument("--shard-lerobot", action="store_true")

    p_ray = sub.add_parser(
        "ray-batch",
        help="Run 5-actor-pool Ray Data streaming batch (falls back to MultiGPUBatchOrchestrator).",
    )
    p_ray.add_argument("input_dir", type=Path)
    p_ray.add_argument("--out", type=Path, required=True)
    p_ray.add_argument("--profile", default="gpu_prod", choices=["auto", "cpu_smoke", "gpu_prod"])
    p_ray.add_argument("--concurrency", type=int, default=2)
    p_ray.add_argument("--no-shard", action="store_true")

    p_shard = sub.add_parser(
        "shard-lerobot",
        help="Atomically pack verified episode outputs into a multi-episode sharded LeRobotDataset v3.0.",
    )
    p_shard.add_argument("runs_dir", type=Path)
    p_shard.add_argument("--out", type=Path, required=True)
    p_shard.add_argument("--max-frames-per-shard", type=int, default=50000)
    p_shard.add_argument("--min-quality", type=float, default=0.0)
    p_shard.add_argument("--skip-gate", action="store_true")
    p_shard.add_argument("--ledger", type=Path, default=None)

    p_val = sub.add_parser(
        "validate-dataset",
        help="Run Layer 7 policy-readiness gate on a single- or multi-episode LeRobot v3.0 dataset.",
    )
    p_val.add_argument("lerobot_dir", type=Path)
    p_val.add_argument("--epochs", type=int, default=4)
    p_val.add_argument("--no-smoke", action="store_true")

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

    if args.command == "batch-gpu":
        mp4s = sorted(args.input_dir.rglob("*.mp4"))
        orch = MultiGPUBatchOrchestrator(
            out_root=args.out,
            profile=args.profile,
            max_gpus=args.max_gpus,
        )
        summary = orch.run_videos(mp4s)
        payload: dict[str, object] = {"episodes": summary}
        if args.shard_lerobot:
            ep_dirs = [args.out / v.stem for v in mp4s if (args.out / v.stem).exists()]
            if ep_dirs:
                payload["sharding"] = pack_sharded_lerobot_v3(
                    ep_dirs,
                    args.out / "lerobot_v3_sharded",
                    run_validation_gate=True,
                    ledger_path=orch.ledger.db_path,
                )
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "ray-batch":
        mp4s = sorted(args.input_dir.rglob("*.mp4"))
        ray_orch = RayStreamingOrchestrator(
            out_root=args.out,
            profile=args.profile,
            concurrency_per_stage=args.concurrency,
        )
        res = ray_orch.run_batch(mp4s, shard_lerobot=not args.no_shard)
        print(json.dumps(res, indent=2))
        return 0

    if args.command == "shard-lerobot":
        ep_dirs = sorted(p.parent for p in args.runs_dir.rglob("run_report.json"))
        shard_res = pack_sharded_lerobot_v3(
            ep_dirs,
            args.out,
            max_frames_per_shard=args.max_frames_per_shard,
            min_quality_score=args.min_quality,
            run_validation_gate=not args.skip_gate,
            ledger_path=args.ledger,
        )
        print(json.dumps(shard_res, indent=2))
        return 0

    if args.command == "validate-dataset":
        gate = validate_lerobot_v3_dataset(
            args.lerobot_dir,
            run_smoke_train=not args.no_smoke,
            smoke_epochs=args.epochs,
        )
        print(json.dumps(gate, indent=2))
        return 0 if gate["passed"] else 1

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
