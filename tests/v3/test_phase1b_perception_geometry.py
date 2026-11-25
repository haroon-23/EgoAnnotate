"""Phase 1B verification on the COMPLETE video (frames 0..N-1): hands, body, objects & 6-DoF world VO."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from src.egoannotate.config import load_config
from src.egoannotate.geometry_world import (
    R_WORLD_CAM0,
    MonocularWorldVO,
    register_hand_and_objects_to_world,
    register_objects_to_world,
)
from src.egoannotate.perception_hands import (
    MediaPipeHandTracker,
    MediaPipeUpperBodyTracker,
)
from src.egoannotate.perception_objects import OpenCVManipulatedObjectTracker
from src.egoannotate.registry import REGISTRY
from src.egoannotate.schemas import FrameAnnotation
from src.egoannotate.video_io import VideoStreamReader, compute_frame_image_quality

FULL_SAMPLE_VIDEO = Path("data/sample_raw/ad521ca5-0017-475b-9dd8-1fdb423a66b7.mp4")
FULL_CACHE_PATH = Path(".ego_cache/phase1b_full_video_cache.json")


def test_world_basis_is_right_handed_and_commercial_registry_populated() -> None:
    assert abs(float(np.linalg.det(R_WORLD_CAM0)) - 1.0) < 1e-9
    assert np.allclose(R_WORLD_CAM0.T @ R_WORLD_CAM0, np.eye(3), atol=1e-9)
    audit_rows = REGISTRY.audit_table()
    names = {(r["stage"], r["name"]): r["commercial_ok"] for r in audit_rows}
    assert names[("hand_pose", "mediapipe_hands")] is True
    assert names[("hand_pose", "hamer_mano_optional")] is False
    assert names[("vo_slam", "opencv_essential_vo")] is True
    assert names[("vo_slam", "dpvo_mit")] is True
    assert names[("depth", "depth_anything_v2_small_apache")] is True


def test_full_video_hands_body_objects_and_world_vo_multi_chunk() -> None:
    if not FULL_SAMPLE_VIDEO.exists():
        pytest.skip("Full Drive sample video not available locally")

    cfg = load_config(profile="cpu_smoke")
    reader = VideoStreamReader(FULL_SAMPLE_VIDEO)
    hand_tracker = MediaPipeHandTracker(cfg.perception, reader.intrinsics, cfg.geometry.assumed_palm_width_m)
    body_tracker = MediaPipeUpperBodyTracker()
    obj_tracker = OpenCVManipulatedObjectTracker(reader.intrinsics)
    vo = MonocularWorldVO(cfg.geometry, reader.intrinsics)

    frames_out: list[FrameAnnotation] = []
    prev_l: np.ndarray | None = None
    prev_r: np.ndarray | None = None
    prev_objs: dict[int, np.ndarray] = {}
    seen_frames: set[int] = set()
    detected_hands = 0
    detected_objs = 0
    detected_body = 0
    cam_positions: list[list[float]] = []

    dt_s = 1.0 / max(float(reader.probe["fps"]), 1.0)
    expected_total = int(reader.probe["nb_frames"])
    t0 = time.perf_counter()

    try:
        for chunk in reader.iter_chunks(chunk_size=300, overlap=10):
            for fi, ts, fr in zip(chunk.frame_indices, chunk.timestamps_s, chunk.frames):
                if fi in seen_frames:
                    continue
                seen_frames.add(fi)
                sharp, luma = compute_frame_image_quality(fr)
                cam_pose = vo.step(fr, fi, chunk.chunk_id, dt_s)
                cam_positions.append(cam_pose.pos_world_m)
                lh, rh = hand_tracker.process_frame(fr, ts)
                body = (
                    body_tracker.process_frame(fr)
                    if (fi % cfg.perception.body_sample_stride == 0)
                    else None
                )
                objs = obj_tracker.process_frame(fr, fi, lh, rh)

                prev_l = register_hand_and_objects_to_world(cam_pose, lh, prev_l, dt_s)
                prev_r = register_hand_and_objects_to_world(cam_pose, rh, prev_r, dt_s)
                register_objects_to_world(cam_pose, objs, prev_objs, dt_s)

                if lh.detected or rh.detected:
                    detected_hands += 1
                if objs:
                    detected_objs += 1
                if body is not None and body.detected:
                    detected_body += 1

                frame_kwargs: dict[str, object] = {
                    "frame_idx": fi,
                    "timestamp_s": round(ts, 5),
                    "sharpness_laplacian": round(sharp, 3),
                    "mean_luma": round(luma, 3),
                    "camera_pose": cam_pose,
                    "left_hand": lh,
                    "right_hand": rh,
                    "objects": objs,
                }
                if body is not None:
                    frame_kwargs["upper_body"] = body
                frames_out.append(FrameAnnotation(**frame_kwargs))  # type: ignore[arg-type]

            elapsed = time.perf_counter() - t0
            fps_proc = len(seen_frames) / max(elapsed, 1e-6)
            print(
                f"  [chunk {chunk.chunk_id:02d}] frames={len(seen_frames)}/{expected_total} "
                f"({fps_proc:.1f} fps, vo_ok={vo.tracking_ok_count}/{vo.total_frames}, "
                f"loop_closures={vo.loop_closures_applied})",
                flush=True,
            )
    finally:
        hand_tracker.close()
        body_tracker.close()

    meta = reader.build_metadata(processed_frames=len(seen_frames))
    scale_prov = vo.build_scale_provenance(reader.intrinsics, cfg.geometry.depth_backend)

    assert meta.is_full_video_processed is True
    assert len(seen_frames) >= expected_total - 2
    assert frames_out[0].frame_idx == 0
    assert frames_out[-1].frame_idx == len(frames_out) - 1
    assert vo.tracking_ok_count / max(vo.total_frames, 1) > 0.70
    assert scale_prov.scale_source == "anthropometric_palm_prior_only"
    assert scale_prov.is_calibrated_metric is False
    cam_arr = np.asarray(cam_positions, dtype=np.float64)
    assert np.all(np.isfinite(cam_arr))

    # Persist real SHA-256-keyed full-video Phase 1B output so Phase 1C verifies all 5400 real frames without duplicating CPU decode
    FULL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cache_payload = {
        "video_sha256": meta.video_sha256,
        "video_path": str(FULL_SAMPLE_VIDEO),
        "total_frames": len(frames_out),
        "scale_provenance": scale_prov.model_dump(),
        "sparse_world_points": vo.sparse_world_points[:500],
        "frames": [f.model_dump() for f in frames_out],
    }
    FULL_CACHE_PATH.write_text(json.dumps(cache_payload), encoding="utf-8")

    path_len_m = float(np.sum(np.linalg.norm(np.diff(cam_arr, axis=0), axis=1)))
    elapsed_total = time.perf_counter() - t0
    print(
        f"\n[PHASE 1B COMPLETE VIDEO] {FULL_SAMPLE_VIDEO.name}: frames={len(seen_frames)}/{expected_total} "
        f"({meta.duration_s:.1f}s), hand_frames={detected_hands}, obj_frames={detected_objs}, "
        f"body_frames={detected_body}, vo_ok={vo.tracking_ok_count}/{vo.total_frames}, "
        f"loop_closures={vo.loop_closures_applied}, cam_path={path_len_m:.3f}m, "
        f"sparse_pts={len(vo.sparse_world_points)}, table_z={scale_prov.table_plane_z_world_m}, "
        f"wall_time={elapsed_total:.1f}s"
    )
