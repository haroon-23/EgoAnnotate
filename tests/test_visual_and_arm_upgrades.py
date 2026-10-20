"""Tests for v1.2 Dense Object Segmentation Tracker, World-Locked Arm Mapper, and 3D Tabletop Sim."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from src.datatypes import AnnotationFrame, ContactState, GraspType, HandLandmarks
from src.perception.dense_object_tracker import (
    detect_manipulated_object_roi_near_hands,
    extract_roi_segmentation_polygon,
    refine_episode_objects_and_masks,
)
from src.retargeting.world_locked_arm_mapper import build_world_locked_arm_trajectory


def _make_synthetic_scene_image(tmp_path: Path, idx: int, obj_cx: int = 120, obj_cy: int = 110) -> str:
    img = np.full((224, 224, 3), (210, 200, 190), dtype=np.uint8)  # light tabletop background
    # Draw distinct bright blue/red object at (obj_cx, obj_cy)
    cv2.circle(img, (obj_cx, obj_cy), 22, (20, 40, 220), -1)
    p = tmp_path / f"frame_{idx:03d}.jpg"
    cv2.imwrite(str(p), img)
    return str(p)


def test_dense_object_tracker_produces_bboxes_and_segmentation_polygons(tmp_path: Path):
    frames = []
    for i in range(8):
        img_p = _make_synthetic_scene_image(tmp_path, i, obj_cx=110 + i * 2, obj_cy=110)
        x = np.full(21, (110 + i * 2) / 224.0, dtype=np.float32)
        y = np.full(21, 110 / 224.0, dtype=np.float32)
        z = np.zeros(21, dtype=np.float32)
        y[9] -= 0.12
        r_hand = HandLandmarks(x=x, y=y, z=z, confidence=0.95, handedness="Right")
        r_contact = ContactState(
            fingers=np.array([True, True, False, False, False]),
            object_name="red_cylinder",
            in_contact=(i >= 2),
            confidence=0.9,
        )
        frames.append(
            AnnotationFrame(
                frame_idx=i,
                timestamp=i / 30.0,
                image_path=img_p,
                left_hand=None,
                right_hand=r_hand,
                right_contact=r_contact,
                right_grasp=GraspType("precision_pinch", 0.9, 0.02, 2),
            )
        )

    refine_episode_objects_and_masks(frames, default_object_name="red_cylinder")
    for f in frames:
        assert len(f.objects) == 1
        obj = f.objects[0]
        assert obj.bbox is not None and len(obj.bbox) == 4
        assert obj.mask_rle is not None and "polygon_norm" in obj.mask_rle
        assert len(obj.mask_rle["polygon_norm"]) >= 3


def test_world_locked_arm_mapper_compensates_camera_motion_and_tracks_object(tmp_path: Path):
    frames = []
    for i in range(10):
        img_p = _make_synthetic_scene_image(tmp_path, i)
        x = np.full(21, 0.5 + 0.005 * i, dtype=np.float32)
        y = np.full(21, 0.5 - 0.005 * i, dtype=np.float32)
        z = np.zeros(21, dtype=np.float32)
        y[9] -= 0.12
        r_hand = HandLandmarks(x=x, y=y, z=z, confidence=0.95, handedness="Right")
        r_contact = ContactState(
            fingers=np.array([True, True, False, False, False]),
            object_name="cup",
            in_contact=(3 <= i <= 7),
            confidence=0.9,
        )
        frames.append(
            AnnotationFrame(
                frame_idx=i,
                timestamp=i / 30.0,
                image_path=img_p,
                left_hand=None,
                right_hand=r_hand,
                right_contact=r_contact,
                right_grasp=GraspType("precision_pinch" if 3 <= i <= 7 else "open", 0.9, 0.02, 2),
            )
        )

    traj = build_world_locked_arm_trajectory(frames)
    assert traj.wrist_world_positions.shape == (10, 3)
    assert traj.object_world_positions.shape == (10, 3)
    assert traj.wrist_to_object_rel_m.shape == (10, 3)
    # Before grasp (frame 0) and after release (frame 9), object rests at table_z_m (0.28m)
    assert abs(float(traj.object_world_positions[0, 2]) - 0.28) < 1e-4
    assert abs(float(traj.object_world_positions[9, 2]) - 0.28) < 1e-4
