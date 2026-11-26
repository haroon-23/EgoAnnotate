"""Unit tests for Phase 2A robot-ready schemas, RTMW occlusion recovery, SAM21/MoGe helpers, and contracts."""

from __future__ import annotations

import itertools

import numpy as np

from src.egoannotate.gpu_backends_v4 import (
    SAM21MaskPropagatorGPU,
    _compute_normals_from_depth,
    _extract_json_object,
    lift_box_to_3d_centroid,
)
from src.egoannotate.perception_hands_rtmw import (
    interpolate_gloved_hand_dropouts,
    regularize_hand_bone_lengths,
)
from src.egoannotate.robot_contracts import (
    annotate_contact_and_tactile_events,
    enforce_gap_free_hierarchical_segments,
    populate_canonical_delta_actions,
)
from src.egoannotate.schemas import (
    AtomicActionSegment,
    CameraWorldPose,
    FrameAnnotation,
    HandState,
    HierarchicalLanguage,
    ManipulatedObjectState,
    TactileHandChannel,
)


def _make_frames(n: int = 90) -> list[FrameAnnotation]:
    out: list[FrameAnnotation] = []
    for i in range(n):
        pos = [0.1 + 0.002 * i, -0.05, 0.45]
        lm2 = [[100.0 + k, 120.0 + k] for k in range(21)]
        lm3 = [[pos[0] + 0.005 * k, pos[1], pos[2]] for k in range(21)]
        rh = HandState(
            handedness="right",
            detected=True,
            confidence=0.9,
            visibility=0.95,
            landmarks_2d_px=lm2,
            landmarks_3d_cam_m=lm3,
            landmarks_3d_world_m=lm3,
            wrist_pos_cam_m=pos,
            wrist_pos_world_m=pos,
            wrist_quat_xyzw_world=[0.0, 0.0, 0.0, 1.0],
            wrist_rot6d_world=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            gripper_aperture_m=0.06 if i < 20 else 0.02,
            finger_curls_5d=[0.2, 0.3, 0.4, 0.4, 0.4],
        )
        tac_r = TactileHandChannel(
            contact_prob=0.85 if 20 <= i < 60 else 0.1,
            slip_proxy_01=0.75 if i == 40 else 0.05,
        )
        objs = [
            ManipulatedObjectState(
                track_id=1,
                label="trowel",
                role="tool",
                score=0.9,
                bbox_xyxy=[80.0, 90.0, 160.0, 170.0],
                mask_iou_with_hand=0.22 if 20 <= i < 60 else 0.0,
            )
        ]
        out.append(
            FrameAnnotation(
                frame_idx=i,
                timestamp_s=round(i / 30.0, 6),
                camera_pose=CameraWorldPose(tracking_ok=True),
                right_hand=rh,
                tactile_right=tac_r,
                objects=objs,
            )
        )
    return out


def test_canonical_delta_actions_finite_difference_consistency() -> None:
    frames = _make_frames(30)
    res = populate_canonical_delta_actions(frames)
    actions = res["canonical_delta_action"]
    assert actions.shape == (30, 16)
    assert bool(np.all(res["proprio_valid"]))
    assert np.allclose(actions[0, 8], 0.002, atol=1e-5)
    assert actions[20, 15] == 1.0
    assert actions[5, 15] == 0.0


def test_contact_events_and_gap_free_3level_hierarchy() -> None:
    frames = _make_frames(90)
    counts = annotate_contact_and_tactile_events(frames)
    assert counts["grasp_onset"] == 1
    assert counts["release"] == 1
    assert counts["tool_workpiece_contact"] > 0

    segs = [
        AtomicActionSegment(
            segment_id=0,
            start_frame=15,
            end_frame=65,
            start_s=0.5,
            end_s=2.1667,
            verb="manipulate",
            acting_hand="right",
            target_objects=["trowel"],
        )
    ]
    lang = HierarchicalLanguage(
        task_goal="Spread cement with trowel.",
        vlm_backend="qwen3_vl_local",
        model_id="Qwen/Qwen3-VL-4B-Instruct",
    )
    fixed = enforce_gap_free_hierarchical_segments(frames, segs, lang)
    assert fixed[0].start_frame == 0
    assert fixed[-1].end_frame == 89
    for a, b in itertools.pairwise(fixed):
        assert b.start_frame == a.end_frame + 1
    manip = next(s for s in fixed if s.verb == "manipulate")
    assert manip.outcome_tag == "slip_recovery"
    assert len(lang.subtasks) >= 1
    assert lang.subtasks[0]["start_frame"] == 0
    assert lang.subtasks[-1]["end_frame"] == 89


def test_gloved_hand_dropout_interpolation_and_bone_regularization() -> None:
    frames = _make_frames(30)
    for k in range(10, 16):
        frames[k].right_hand.detected = False
        frames[k].right_hand.confidence = 0.0
    filled = interpolate_gloved_hand_dropouts(frames, max_gap_frames=18)
    assert filled == 6
    assert all(frames[k].right_hand.detected for k in range(10, 16))

    lm = np.zeros((21, 3), dtype=np.float32)
    lm[5] = [0.0, 0.0, 0.5]
    lm[17] = [0.25, 0.0, 0.5]
    reg = regularize_hand_bone_lengths(lm, assumed_palm_width_m=0.085)
    assert float(np.linalg.norm(reg[17] - reg[5])) <= 0.115


def test_depth_normals_centroid_lift_and_sam21_iou() -> None:
    k_px = np.array([[500.0, 0.0, 160.0], [0.0, 500.0, 120.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    depth = np.full((240, 320), 1.2, dtype=np.float32)
    normals = _compute_normals_from_depth(depth, k_px)
    assert normals.shape == (240, 320, 3)
    xyz, z_m = lift_box_to_3d_centroid([120.0, 80.0, 200.0, 160.0], depth, k_px, 320, 240)
    assert abs(z_m - 1.2) < 1e-3
    assert abs(xyz[0]) < 1e-3

    m1 = np.zeros((20, 20), dtype=bool)
    m2 = np.zeros((20, 20), dtype=bool)
    m1[2:12, 2:12] = True
    m2[2:12, 7:17] = True
    assert abs(SAM21MaskPropagatorGPU.compute_mask_iou(m1, m2) - (50.0 / 150.0)) < 1e-3
    assert _extract_json_object('prefix {"task_goal": "Cut cloth"} suffix')["task_goal"] == "Cut cloth"
