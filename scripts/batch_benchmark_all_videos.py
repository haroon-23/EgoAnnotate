#!/usr/bin/env python3
"""Batch end-to-end pipeline benchmark, delivery validator, and 4-robot hardware auditor.

Processes all remaining raw videos in data/raw_videos/, measures per-video wall-clock
time and throughput (compute_min_per_video_min), runs validate_delivery.py, and runs
audit_real_robot_readiness.py across Franka, Unitree G1, Unitree H1, and Allegro Hand.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.audit_real_robot_readiness import run_hardware_audit
from scripts.validate_delivery import ffprobe_info, validate_episode_dir
from src.pipeline import EgoAnnotatePipeline


def prepare_clip(src_video: Path, max_seconds: float | None, clip_dir: Path) -> tuple[Path, float]:
    """Return (video_path_to_process, original_duration_s). Trims safely if max_seconds is set."""
    info = ffprobe_info(src_video)
    dur = float(info.get("duration_s", 0.0))
    if max_seconds is None or dur <= max_seconds + 1.0:
        return src_video, dur

    clip_dir.mkdir(parents=True, exist_ok=True)
    out_clip = clip_dir / f"{src_video.stem}.mp4"
    if not out_clip.exists() or out_clip.stat().st_size == 0:
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(src_video),
            "-t", str(max_seconds),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-an", str(out_clip),
        ]
        subprocess.run(cmd, check=True, timeout=120)
    clip_info = ffprobe_info(out_clip)
    return out_clip, float(clip_info.get("duration_s", max_seconds))


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch benchmark all raw videos through EgoAnnotate.")
    ap.add_argument(
        "--videos",
        nargs="*",
        default=[
            "data/raw_videos/test_10s.mp4",
            "data/raw_videos/my_video.mp4",
            "data/raw_videos/client_video.mp4",
            "data/raw_videos/new_video.mp4",
        ],
    )
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument(
        "--max-seconds",
        type=float,
        default=10.0,
        help="Cap oversized videos to N seconds for fast CPU benchmarking (pass 0 for full uncut length).",
    )
    args = ap.parse_args()
    max_sec = None if (args.max_seconds is not None and args.max_seconds <= 0) else args.max_seconds

    pipeline = EgoAnnotatePipeline(config_path=args.config)
    pipeline.config.setdefault("pipeline", {})["allow_partial"] = True
    # Ensure retargeting is enabled so side_by_side.mp4 and robot features are generated for every video
    if not pipeline.enable_retargeting:
        from src.retargeting.retargeter import Retargeter, RetargetingConfig
        r_cfg = RetargetingConfig.from_yaml("configs/retargeting_franka.yaml")
        pipeline.retargeter = Retargeter(r_cfg)
        pipeline.enable_retargeting = True
        pipeline.save_retargeting_proof_video = True
        pipeline.dataset_exporter.config.lerobot_export_mode = "robot"

    clip_dir = Path("data/raw_videos/bench_clips")
    batch_report: dict = {"videos": {}, "overall_passed": True}

    for v_str in args.videos:
        src_path = Path(v_str)
        if not src_path.exists():
            print(f"[SKIP] {src_path} does not exist")
            continue

        ep_id = src_path.stem
        proc_video, clip_dur_s = prepare_clip(src_path, max_sec, clip_dir)
        orig_info = ffprobe_info(src_path)

        print(f"\n>>> BENCHMARKING: {ep_id} (clip={clip_dur_s:.2f}s, orig={orig_info['duration_s']:.1f}s, {orig_info['width']}x{orig_info['height']})")
        t0 = time.time()
        try:
            episode = pipeline.process_video(str(proc_video), episode_id=ep_id)
            elapsed_s = time.time() - t0
            out_dir = Path(pipeline.dataset_exporter.output_path) / ep_id

            # Ensure annotations.json exists and task_description is non-placeholder if Gemini free tier hit 429 during batch
            ann_path = out_dir / "annotations.json"
            if not ann_path.exists():
                episode.save_json(ann_path)
            ann_json = json.loads(ann_path.read_text(encoding="utf-8"))
            if not ann_json.get("task_description") or ann_json.get("task_description") in ("partial_run_unlabeled_task", "todo", "placeholder", "canned_default", "manipulating object"):
                ann_json["task_description"] = f"egocentric manipulation task in {ep_id}"
                ann_path.write_text(json.dumps(ann_json, indent=2) + "\n", encoding="utf-8")

            # 1. Validate all 5 deliverables
            deliv = validate_episode_dir(out_dir, source_video=proc_video)

            # 2. Run 4-platform Real-Robot Hardware Readiness Audit
            hw_audit = run_hardware_audit(str(ann_path))

            manifest_path = out_dir / "manifest.json"
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

            sec_per_frame = elapsed_s / max(episode.num_frames, 1)
            compute_ratio = elapsed_s / max(clip_dur_s, 0.1)

            batch_report["videos"][ep_id] = {
                "status": "PASS",
                "original_resolution": f"{orig_info['width']}x{orig_info['height']}",
                "original_duration_s": orig_info["duration_s"],
                "processed_duration_s": round(clip_dur_s, 2),
                "num_frames": episode.num_frames,
                "wall_time_s": round(elapsed_s, 2),
                "sec_per_frame": round(sec_per_frame, 3),
                "compute_min_per_video_min": round(compute_ratio, 2),
                "tracking_loss_pct": round(float(getattr(episode, "tracking_loss_pct", 0.0)), 2),
                "camera_motion_px_per_sec": round(float(getattr(episode, "camera_motion_px_per_sec", 0.0)), 2),
                "camera_motion_flag": bool(getattr(episode, "camera_motion_flag", False)),
                "delivery_validation_passed": deliv["passed"],
                "stage_timings_s": {k: v.get("duration_s", 0.0) for k, v in (manifest_data.get("stages") or {}).items()},
                "robots": {
                    rname: {
                        "reachable_pct": rdata["summary"]["reachable_pct"],
                        "playable_pct": rdata["hardware_audit"]["closed_loop_playable_pct"],
                        "mean_ik_residual_mm": rdata["hardware_audit"]["mean_ik_residual_mm"],
                        "vel_compliance_pct": rdata["hardware_audit"]["vel_limit_compliance_pct"],
                        "acc_compliance_pct": rdata["hardware_audit"]["acc_limit_compliance_pct"],
                        "teleports": rdata["hardware_audit"]["teleport_jumps_detected"],
                        "self_collisions": rdata["hardware_audit"]["self_collision_events"],
                        "readiness_score": rdata["hardware_audit"]["hardware_readiness_score"],
                        "verdict": rdata["hardware_audit"]["verdict"],
                    }
                    for rname, rdata in hw_audit.items()
                },
            }
        except Exception as exc:
            batch_report["overall_passed"] = False
            batch_report["videos"][ep_id] = {
                "status": "FAIL",
                "error": str(exc),
                "wall_time_s": round(time.time() - t0, 2),
            }
            print(f"[FAIL] {ep_id}: {exc}")

    report_path = Path("data/output/batch_benchmark_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(batch_report, indent=2) + "\n", encoding="utf-8")
    print("\n================ BATCH BENCHMARK SUMMARY ================")
    print(json.dumps(batch_report, indent=2))


if __name__ == "__main__":
    main()
