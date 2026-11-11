from __future__ import annotations
#!/usr/bin/env python3
"""Pipeline Stage Wall-Clock Latency, Peak RSS Memory, and Cloud Unit-Economics Benchmark."""

import argparse
import json
import resource
import sys
import tempfile
import time
from pathlib import Path
import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def get_peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return round(float(usage) / (1024.0 * 1024.0), 2)
    return round(float(usage) / 1024.0, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark EgoAnnotate stage throughput and unit economics")
    parser.add_argument("--video", type=str, default="data/raw_videos/test_10s.mp4")
    parser.add_argument("--episode-dir", type=str, default="data/output/test_10s")
    parser.add_argument("--cpu-dollar-per-hr", type=float, default=0.16, help="Cloud CPU worker $/hr (e.g. 4-vCPU spot)")
    parser.add_argument("--gpu-dollar-per-hr", type=float, default=1.20, help="Cloud GPU worker $/hr (e.g. L4/A10G)")
    parser.add_argument("--out", type=str, default="benchmark_report.json")
    args = parser.parse_args()

    vid_path = Path(args.video)
    ep_dir = Path(args.episode_dir)
    cap = cv2.VideoCapture(str(vid_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 300)
    dur_s = max(n_frames / fps, 0.1)

    stage_times = {}

    # Stage 1: Video Decode + Privacy Face Blur Scan
    t0 = time.perf_counter()
    from src.privacy import blur_faces_in_frame
    sampled = 0
    for i in range(min(60, n_frames)):
        ret, bgr = cap.read()
        if not ret:
            break
        if i % 3 == 0:
            blur_faces_in_frame(bgr)
        sampled += 1
    cap.release()
    scale_f = n_frames / max(sampled, 1)
    stage_times["1_video_decode_and_privacy_blur_s"] = round((time.perf_counter() - t0) * scale_f, 3)

    # Stage 2: Load Annotations & Compute 24-D Camera-Frame Hand State + Action Deltas
    t0 = time.perf_counter()
    from src.datatypes import AnnotatedEpisode
    from src.perception.backends import compute_24d_camera_frame_state_and_actions
    ep = AnnotatedEpisode.load_json(str(ep_dir / "annotations.json"))
    states_24d, actions_24d, hand_valid_2d = compute_24d_camera_frame_state_and_actions(ep.frames, 1024, 576)
    stage_times["2_camera_frame_24d_state_action_s"] = round(time.perf_counter() - t0, 3)

    # Stage 3: Independent PyBullet FK & Multi-Embodiment Audit (in temp dir to avoid mutating live artifacts)
    t0 = time.perf_counter()
    from src.retargeting.universal_multi_retargeter import compile_all_embodiments_for_episode
    with tempfile.TemporaryDirectory(prefix="ego_bench_") as tmp_d:
        tmp_p = Path(tmp_d)
        tmp_ann = tmp_p / "annotations.json"
        tmp_ann.write_bytes((ep_dir / "annotations.json").read_bytes())
        compile_all_embodiments_for_episode(tmp_ann, tmp_p, render_multi_video=False)
        stage_times["3_kinematic_retargeting_and_fk_audit_s"] = round(time.perf_counter() - t0, 3)

        # Stage 4: LeRobot v3 Parquet + Gzipped HDF5 Export (in temp dir)
        t0 = time.perf_counter()
        pa_tbl = pa.Table.from_arrays(
            [
                pa.FixedSizeListArray.from_arrays(pa.array(states_24d.reshape(-1), type=pa.float32()), 24),
                pa.FixedSizeListArray.from_arrays(pa.array(actions_24d.reshape(-1), type=pa.float32()), 24),
                pa.FixedSizeListArray.from_arrays(pa.array(hand_valid_2d.reshape(-1), type=pa.int64()), 2),
            ],
            names=["observation.state", "action", "observation.hand_valid"],
        )
        pq.write_table(pa_tbl, tmp_p / "bench.parquet")
        with h5py.File(tmp_p / "bench.h5", "w") as hf:
            hf.create_dataset("state", data=states_24d, compression="gzip", compression_opts=9)
            hf.create_dataset("action", data=actions_24d, compression="gzip", compression_opts=9)
        stage_times["4_lerobot_v3_and_hdf5_export_s"] = round(time.perf_counter() - t0, 3)

    total_cpu_s = round(sum(stage_times.values()), 3)
    sec_per_video_sec = round(total_cpu_s / dur_s, 3)
    cost_cpu_per_video_hr = round(sec_per_video_sec * args.cpu_dollar_per_hr, 4)
    est_gpu_sec_per_video_sec = round(sec_per_video_sec / 4.5, 3)
    cost_gpu_per_video_hr = round(est_gpu_sec_per_video_sec * args.gpu_dollar_per_hr, 4)

    report = {
        "clip": str(vid_path),
        "frames": n_frames,
        "video_duration_s": round(dur_s, 2),
        "stage_wall_clock_seconds": stage_times,
        "total_cpu_seconds": total_cpu_s,
        "cpu_seconds_per_video_second": sec_per_video_sec,
        "peak_rss_memory_mb": get_peak_rss_mb(),
        "unit_economics_per_verified_hour": {
            "cpu_rate_usd_per_hr": args.cpu_dollar_per_hr,
            "gpu_rate_usd_per_hr": args.gpu_dollar_per_hr,
            "projected_cpu_cost_usd_per_video_hour": cost_cpu_per_video_hr,
            "projected_gpu_cost_usd_per_video_hour": cost_gpu_per_video_hr,
        },
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
