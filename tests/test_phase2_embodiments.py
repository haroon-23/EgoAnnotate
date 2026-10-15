"""Phase 2 tests: Persistent Left/Right hand identity and 3-platform Embodiment Layer validation."""
from __future__ import annotations

import numpy as np
import pytest

from src.datatypes import AnnotationFrame, HandLandmarks
from src.retargeting.embodiment_registry import Adapter, compute_anatomical_finger_curls
from src.retargeting.pose_mapper import PoseMapper, PoseMapperConfig


def _make_hand(cx: float, cy: float, cz: float, handedness: str = "Right", curl: float = 0.0) -> HandLandmarks:
    x = np.full(21, cx, dtype=np.float32)
    y = np.full(21, cy, dtype=np.float32)
    z = np.full(21, cz, dtype=np.float32)
    palm = 0.10
    x[5], y[5] = cx - 0.04, cy - palm
    x[9], y[9] = cx, cy - palm
    x[13], y[13] = cx + 0.03, cy - palm
    x[17], y[17] = cx + 0.05, cy - palm
    # Fingertips (4, 8, 12, 16, 20): extended = 2.0*palm, curled = 0.85*palm
    tip_ext = palm * (2.0 - 1.15 * np.clip(curl, 0.0, 1.0))
    for tip_idx in (4, 8, 12, 16, 20):
        y[tip_idx] = cy - tip_ext
    return HandLandmarks(x=x, y=y, z=z, confidence=0.95, handedness=handedness)


def test_persistent_hand_identity_never_swaps_mid_clip():
    """Phase 2: When right hand drops out while left hand is visible, PoseMapper must NOT jump to left hand."""
    cfg = PoseMapperConfig(preferred_hand="right", lock_driving_hand=True, enable_trajectory_smoothing=False)
    mapper = PoseMapper(cfg)

    frames = []
    for i in range(6):
        # Right hand present in frames 0, 1, 4, 5; absent in 2, 3. Left hand present in all frames.
        r_hand = _make_hand(0.6, 0.5, 0.0, "Right") if i not in (2, 3) else None
        l_hand = _make_hand(0.2, 0.5, 0.0, "Left")
        frames.append(
            AnnotationFrame(
                frame_idx=i,
                timestamp=i / 30.0,
                image_path=f"f_{i}.jpg",
                left_hand=l_hand,
                right_hand=r_hand,
            )
        )

    targets = mapper.map_frames(frames)
    used_sides = [t.hand_used for t in targets]
    assert used_sides == ["right", "right", None, None, "right", "right"]
    assert targets[2].hand_detected is False
    assert targets[3].hand_detected is False

    # And map_frames_for_hand("left") extracts strictly the left hand trajectory
    left_targets = mapper.map_frames_for_hand(frames, "left")
    assert all(t.hand_used == "left" and t.hand_detected for t in left_targets)


def test_three_embodiment_platforms_validation():
    """Phase 2: Validate Franka (arm), Unitree G1 (bimanual humanoid), and Allegro (dexterous hand)."""
    n_steps = 25
    t = np.linspace(0, 2 * np.pi, n_steps)
    # Smooth tabletop trajectory in reachable workspace
    right_traj = [
        (
            np.array([0.38 + 0.04 * np.cos(ti), -0.15 + 0.04 * np.sin(ti), 0.28], dtype=np.float64),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        )
        for ti in t
    ]
    left_traj = [
        (
            np.array([0.38 + 0.04 * np.cos(ti), 0.15 + 0.04 * np.sin(ti), 0.28], dtype=np.float64),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        )
        for ti in t
    ]
    grips = [float(0.5 + 0.5 * np.sin(ti)) for ti in t]
    hands = [_make_hand(0.5, 0.5, 0.0, "Right", curl=g) for g in grips]

    # 1. Single Arm + Parallel Gripper (Franka Panda)
    franka = Adapter("configs/embodiments/franka_arm.yaml")
    rep_franka = franka.solve_episode(right_traj, grips=grips)
    assert rep_franka.reachable_pct >= 90.0
    assert rep_franka.mean_ik_residual_mm < 50.0
    assert rep_franka.q_trajectory.shape == (n_steps, franka.model.nq)

    # 2. Dual-Arm Bimanual Humanoid (Unitree G1 Upper Body)
    g1 = Adapter("configs/embodiments/unitree_g1_upper.yaml")
    assert len(g1.tasks) == 2
    rep_g1 = g1.solve_episode(right_traj, left_targets=left_targets if False else left_traj, grips=grips)
    assert rep_g1.reachable_pct >= 90.0
    assert rep_g1.mean_ik_residual_mm < 80.0
    assert rep_g1.q_trajectory.shape == (n_steps, g1.model.nq)

    # 3. Dexterous Multi-Finger Hand (Allegro 16-DoF Hand)
    allegro = Adapter("configs/embodiments/allegro_hand.yaml")
    rep_allegro = allegro.solve_episode(right_traj, grips=grips, hand_landmarks_seq=hands)
    assert rep_allegro.reachable_pct == 100.0
    assert rep_allegro.q_trajectory.shape == (n_steps, 16)
    # Verify anatomical finger curls actually modulate the 16 finger joints
    assert float(np.ptp(rep_allegro.q_trajectory[:, 1])) > 0.4
