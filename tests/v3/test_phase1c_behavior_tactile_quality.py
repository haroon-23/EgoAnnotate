"""Phase 1C verification on the COMPLETE 5400-frame video: behavior, Feix grasps, tactile, semantics & quality."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.egoannotate.behavior import (
    build_scene_graph_and_affordances,
    enrich_frames_with_gaze,
    extract_grasp_events,
    segment_atomic_actions,
)
from src.egoannotate.config import load_config
from src.egoannotate.geometry_world import (
    MonocularWorldVO,
    register_hand_and_objects_to_world,
    register_objects_to_world,
)
from src.egoannotate.perception_hands import MediaPipeHandTracker
from src.egoannotate.perception_objects import OpenCVManipulatedObjectTracker
from src.egoannotate.quality import score_episode_quality
from src.egoannotate.registry import REGISTRY
from src.egoannotate.schemas import FrameAnnotation, ScaleProvenance
from src.egoannotate.semantics import annotate_semantics_grounded
from src.egoannotate.tactile import (
    ingest_external_tactile_stream,
    populate_rgb_tactile_proxies,
)
from src.egoannotate.video_io import VideoStreamReader, compute_frame_image_quality

FULL_SAMPLE_VIDEO = Path("data/sample_raw/ad521ca5-0017-475b-9dd8-1fdb423a66b7.mp4")
FULL_CACHE_PATH = Path(".ego_cache/phase1b_full_video_cache.json")


def _load_or_compute_full_video_frames() -> tuple[list[FrameAnnotation], ScaleProvenance, list[list[float]]]:
    reader = VideoStreamReader(FULL_SAMPLE_VIDEO)
    expected_total = int(reader.probe["nb_frames"])

    if FULL_CACHE_PATH.exists():
        cached = json.loads(FULL_CACHE_PATH.read_text(encoding="utf-8"))
        if (
            cached.get("video_sha256") == reader.sha256
            and int(cached.get("total_frames", 0)) >= expected_total - 2
        ):
            frames = [FrameAnnotation.model_validate(f) for f in cached["frames"]]
            scale_prov = ScaleProvenance.model_validate(cached["scale_provenance"])
            sparse_pts = cached.get("sparse_world_points", [])
            return frames, scale_prov, sparse_pts

    cfg = load_config(profile="cpu_smoke")
    hand_tracker = MediaPipeHandTracker(cfg.perception, reader.intrinsics, cfg.geometry.assumed_palm_width_m)
    obj_tracker = OpenCVManipulatedObjectTracker(reader.intrinsics)
    vo = MonocularWorldVO(cfg.geometry, reader.intrinsics)

    frames_out: list[FrameAnnotation] = []
    prev_l: np.ndarray | None = None
    prev_r: np.ndarray | None = None
    prev_objs: dict[int, np.ndarray] = {}
    dt_s = 1.0 / max(float(reader.probe["fps"]), 1.0)
    try:
        for fi, ts, fr in reader.iter_frames():
            sharp, luma = compute_frame_image_quality(fr)
            cam_pose = vo.step(fr, fi, fi // 300, dt_s)
            lh, rh = hand_tracker.process_frame(fr, ts)
            objs = obj_tracker.process_frame(fr, fi, lh, rh)
            prev_l = register_hand_and_objects_to_world(cam_pose, lh, prev_l, dt_s)
            prev_r = register_hand_and_objects_to_world(cam_pose, rh, prev_r, dt_s)
            register_objects_to_world(cam_pose, objs, prev_objs, dt_s)
            frames_out.append(
                FrameAnnotation(
                    frame_idx=fi,
                    timestamp_s=round(ts, 5),
                    sharpness_laplacian=round(sharp, 3),
                    mean_luma=round(luma, 3),
                    camera_pose=cam_pose,
                    left_hand=lh,
                    right_hand=rh,
                    objects=objs,
                )
            )
    finally:
        hand_tracker.close()

    scale_prov = vo.build_scale_provenance(reader.intrinsics, cfg.geometry.depth_backend)
    return frames_out, scale_prov, vo.sparse_world_points[:500]


def test_full_video_behavior_tactile_semantics_and_quality(tmp_path: Path) -> None:
    if not FULL_SAMPLE_VIDEO.exists():
        pytest.skip("Full Drive sample video not available locally")

    cfg = load_config(profile="cpu_smoke")
    frames, scale_prov, sparse_pts = _load_or_compute_full_video_frames()
    assert len(frames) >= 5398, f"Expected full 5400-frame video, got {len(frames)}"

    enrich_frames_with_gaze(frames)
    populate_rgb_tactile_proxies(frames, cfg.tactile)

    # Honesty check across all 5400 frames
    assert all(f.tactile_left.source == "derived_rgb_proxy" for f in frames)
    assert all(f.tactile_right.source == "derived_rgb_proxy" for f in frames)
    assert all(f.tactile_right.uncertainty >= 0.35 for f in frames)

    grasps = extract_grasp_events(frames, cfg.behavior)
    segments = segment_atomic_actions(frames, grasps, cfg.behavior)
    assert len(segments) >= 5
    assert segments[0].start_frame == 0
    assert segments[-1].end_frame == len(frames) - 1

    lang = annotate_semantics_grounded(frames, segments, cfg.semantic)
    assert lang.task_goal
    assert lang.grounding_verified is True
    assert all(s.semantic_description for s in segments)
    assert all(0.0 <= s.geometry_agreement_score <= 1.0 for s in segments)

    sg = build_scene_graph_and_affordances(frames, grasps, sparse_pts)
    assert len(sg.nodes) >= 3
    assert "min_xyz_m" in sg.workspace_bounds_world_m

    qual = score_episode_quality(frames, segments, scale_prov, cfg.quality)
    assert 0.0 < qual.robot_readiness_score <= 1.0
    assert qual.verdict in ("PASS", "WARN", "QUARANTINE")
    assert len(qual.segment_reports) == len(segments)

    # Verify real external hardware tactile stream ingestion over first 30 timestamps
    csv_path = tmp_path / "glove_tactile.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_s", "hand", "z0", "z1", "z2", "z3", "z4", "z5"])
        for fr in frames[:30]:
            writer.writerow([fr.timestamp_s, "right", 1.2, 2.4, 0.8, 0.2, 0.1, 0.5])
    ingest_res = ingest_external_tactile_stream(frames, csv_path)
    assert ingest_res["source"] == "hardware_sensor"
    assert isinstance(ingest_res["synced_hand_frames"], int) and ingest_res["synced_hand_frames"] >= 30
    assert frames[0].tactile_right.source == "hardware_sensor"
    assert frames[0].tactile_right.raw_sensor_values == [1.2, 2.4, 0.8, 0.2, 0.1, 0.5]

    print(
        f"\n[PHASE 1C COMPLETE VIDEO] frames={len(frames)}, duration={frames[-1].timestamp_s:.1f}s, "
        f"grasps={len(grasps)}, segments={len(segments)}, subtasks={len(lang.subtasks)}, "
        f"readiness={qual.robot_readiness_score:.3f} ({qual.verdict}), flags={qual.flags}, "
        f"goal='{lang.task_goal}'"
    )


def test_quality_scorer_quarantines_dark_blurry_handless_footage() -> None:
    cfg = load_config(profile="cpu_smoke")
    degraded = [
        FrameAnnotation(frame_idx=i, timestamp_s=i / 30.0, sharpness_laplacian=2.0, mean_luma=4.0)
        for i in range(60)
    ]
    scale_prov = ScaleProvenance(scale_meters_per_unit=1.0, scale_uncertainty_pct=25.0, is_calibrated_metric=False)
    report = score_episode_quality(degraded, [], scale_prov, cfg.quality)
    assert report.verdict == "QUARANTINE"
    assert report.robot_readiness_score <= 0.40
    assert "low_hand_visibility" in report.flags
    assert "severe_motion_blur" in report.flags
    assert "underexposed_low_light" in report.flags


def test_vlm_registry_has_zero_paid_apis() -> None:
    vlm_specs = REGISTRY.list_stage("vlm")
    assert vlm_specs
    assert all(spec.commercial_ok and not spec.is_paid_api for spec in vlm_specs)
