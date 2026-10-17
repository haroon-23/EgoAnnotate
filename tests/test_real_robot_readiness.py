"""Tests for v1.1 Real-Robot Control Filter: minimum-jerk gap bridging, bimanual repulsion, and hardware audit."""
from __future__ import annotations

import numpy as np
from src.retargeting.control_filter import (
    bridge_cartesian_gaps_minimum_jerk,
    compute_action_chunks,
    enforce_bimanual_separation,
    smooth_and_limit_joint_trajectory,
)
from src.retargeting.embodiment_registry import Adapter


def test_minimum_jerk_gap_bridge_eliminates_teleport_discontinuity():
    # Create a 20-frame trajectory with a 6-frame dropout in the middle where hand moves 20cm
    n = 20
    targets = []
    valid = []
    for i in range(n):
        if 6 <= i <= 11:
            # Corrupt dropout pose (e.g. snapped to workspace center)
            targets.append((np.array([0.80, 0.50, 0.80]), np.array([0.0, 0.0, 0.0, 1.0])))
            valid.append(False)
        else:
            x = 0.30 + 0.01 * i
            targets.append((np.array([x, -0.10, 0.35]), np.array([0.0, 0.0, 0.0, 1.0])))
            valid.append(True)

    bridged, playable = bridge_cartesian_gaps_minimum_jerk(
        targets, valid, home_pos=np.array([0.35, -0.10, 0.35])
    )
    assert np.all(playable)
    pos = np.stack([p for p, _ in bridged], axis=0)
    max_step_m = float(np.max(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
    # Smooth quintic bridge should never step more than ~2.5cm per frame across the gap
    assert max_step_m < 0.03


def test_bimanual_separation_prevents_wrist_crossing():
    r = [(np.array([0.30, 0.02, 0.25]), np.array([0.0, 0.0, 0.0, 1.0]))]
    l = [(np.array([0.30, -0.02, 0.25]), np.array([0.0, 0.0, 0.0, 1.0]))]
    r_safe, l_safe = enforce_bimanual_separation(r, l, min_separation_m=0.20)
    assert float(l_safe[0][0][1] - r_safe[0][0][1]) >= 0.199


def test_all_four_platforms_achieve_high_hardware_readiness_score():
    n = 30
    t = np.linspace(0, 2 * np.pi, n)
    r_traj = [(np.array([0.35 + 0.03 * np.cos(ti), -0.15 + 0.03 * np.sin(ti), 0.28]), np.array([0, 0, 0, 1.0])) for ti in t]
    l_traj = [(np.array([0.35 + 0.03 * np.cos(ti), 0.15 + 0.03 * np.sin(ti), 0.28]), np.array([0, 0, 0, 1.0])) for ti in t]
    valid = [True if i not in (10, 11, 12) else False for i in range(n)]
    grips = [0.5] * n

    for cfg in [
        "configs/embodiments/franka_arm.yaml",
        "configs/embodiments/unitree_g1_upper.yaml",
        "configs/embodiments/unitree_h1_upper.yaml",
        "configs/embodiments/allegro_hand.yaml",
    ]:
        ad = Adapter(cfg)
        rep = ad.solve_episode(r_traj, left_targets=l_traj, grips=grips, valid_mask=valid, left_valid_mask=valid)
        assert rep.hardware_audit.teleport_jumps_detected == 0
        assert rep.hardware_audit.self_collision_events == 0
        assert rep.hardware_audit.hardware_readiness_score >= 85.0
        assert rep.action_chunks_16.shape == (n, 16, ad.model.nq + 1)
