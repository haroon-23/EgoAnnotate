"""Phase 1B verification: hands, upper-body, manipulated objects, 6-DoF VO & world-frame registration."""

from __future__ import annotations

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
from src.egoannotate.video_io import VideoStreamReader


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


def test_real_video_hands_body_objects_and_world_vo_multi_chunk() -> None:
    clip = Path("data/sample_raw/smoke_real_8s.mp4")
    if not clip.exists():
        pytest.skip("Real smoke clip not available locally")

    cfg = load_config(profile="cpu_smoke")
    reader = VideoStreamReader(clip)
    hand_tracker = MediaPipeHandTracker(cfg.perception, reader.intrinsics, cfg.geometry.assumed_palm_width_m)
    body_tracker = MediaPipeUpperBodyTracker()
    obj_tracker = OpenCVManipulatedObjectTracker(reader.intrinsics)
    vo = MonocularWorldVO(cfg.geometry, reader.intrinsics)

    prev_l: np.ndarray | None = None
    prev_r: np.ndarray | None = None
    prev_objs: dict[int, np.ndarray] = {}
    seen_frames: set[int] = set()
    detected_hands = 0
    detected_objs = 0
    cam_positions: list[list[float]] = []

    dt_s = 1.0 / max(float(reader.probe["fps"]), 1.0)
    try:
        for chunk in reader.iter_chunks(chunk_size=90, overlap=10):
            for fi, ts, fr in zip(chunk.frame_indices, chunk.timestamps_s, chunk.frames):
                if fi in seen_frames:
                    continue
                seen_frames.add(fi)
                cam_pose = vo.step(fr, fi, chunk.chunk_id, dt_s)
                cam_positions.append(cam_pose.pos_world_m)
                lh, rh = hand_tracker.process_frame(fr, ts)
                if fi % cfg.perception.body_sample_stride == 0:
                    _ = body_tracker.process_frame(fr)
                objs = obj_tracker.process_frame(fr, fi, lh, rh)

                prev_l = register_hand_and_objects_to_world(cam_pose, lh, prev_l, dt_s)
                prev_r = register_hand_and_objects_to_world(cam_pose, rh, prev_r, dt_s)
                register_objects_to_world(cam_pose, objs, prev_objs, dt_s)

                if lh.detected or rh.detected:
                    detected_hands += 1
                if objs:
                    detected_objs += 1
    finally:
        hand_tracker.close()
        body_tracker.close()

    scale_prov = vo.build_scale_provenance(reader.intrinsics, cfg.geometry.depth_backend)
    assert len(seen_frames) >= 200
    assert vo.tracking_ok_count / max(vo.total_frames, 1) > 0.70
    assert scale_prov.scale_source == "anthropometric_palm_prior_only"
    assert scale_prov.is_calibrated_metric is False
    cam_arr = np.asarray(cam_positions, dtype=np.float64)
    assert np.all(np.isfinite(cam_arr))
    path_len_m = float(np.sum(np.linalg.norm(np.diff(cam_arr, axis=0), axis=1)))
    print(
        f"\n[PHASE 1B REAL CLIP] frames={len(seen_frames)}, hand_frames={detected_hands}, "
        f"obj_frames={detected_objs}, vo_ok={vo.tracking_ok_count}/{vo.total_frames}, "
        f"cam_path={path_len_m:.3f}m, sparse_pts={len(vo.sparse_world_points)}, "
        f"table_z={scale_prov.table_plane_z_world_m}"
    )
