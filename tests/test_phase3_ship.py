"""Phase 3 tests: MCAP v0 export, delivery validator, documentation & embodiment H1/Allegro fixes."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from src.datatypes import AnnotatedEpisode, AnnotationFrame, HandLandmarks
from src.mcap_exporter import export_episode_to_mcap, verify_mcap_file
from src.retargeting.embodiment_registry import Adapter
from scripts.validate_delivery import validate_episode_dir


def test_mcap_exporter_roundtrip_and_magic(tmp_path: Path):
    frames = [
        AnnotationFrame(
            frame_idx=i,
            timestamp=i / 30.0,
            image_path=f"f_{i}.jpg",
            left_hand=None,
            right_hand=HandLandmarks(
                x=np.full(21, 0.5, dtype=np.float32),
                y=np.full(21, 0.5, dtype=np.float32),
                z=np.zeros(21, dtype=np.float32),
                confidence=0.9,
                handedness="Right",
            ),
        )
        for i in range(10)
    ]
    ep = AnnotatedEpisode(
        episode_id="mcap_test_ep",
        video_path="dummy.mp4",
        task_description="pick up the red mug from the table",
        frames=frames,
        segments=[],
        num_frames=10,
        duration_seconds=0.33,
        target_robot="franka_panda",
        camera_motion_flag=False,
        camera_motion_px_per_sec=1.2,
    )
    mcap_file = export_episode_to_mcap(ep, tmp_path / "episode.mcap")
    info = verify_mcap_file(mcap_file)
    assert info["valid_magic"] is True
    assert info["topics"]["/human/frame"] == 10
    assert info["topics"]["/robot/frame"] == 10
    assert info["topics"]["/episode/metadata"] == 1
    assert info["total_messages"] == 21


def test_unitree_h1_and_allegro_reachability_after_refinement():
    n = 15
    right_traj = [
        (np.array([0.38, -0.18, 0.25], dtype=np.float64), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64))
        for _ in range(n)
    ]
    left_traj = [
        (np.array([0.38, 0.18, 0.25], dtype=np.float64), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64))
        for _ in range(n)
    ]
    h1 = Adapter("configs/embodiments/unitree_h1_upper.yaml")
    rep_h1 = h1.solve_episode(right_traj, left_targets=left_traj, grips=[0.5] * n)
    assert rep_h1.reachable_pct > 50.0

    allegro = Adapter("configs/embodiments/allegro_hand.yaml")
    rep_al = allegro.solve_episode(right_traj, grips=[0.5] * n)
    assert rep_al.reachable_pct == 100.0
    assert rep_al.joint_limit_events == 0


def test_required_ship_docs_exist():
    for fname in ("README.md", "DATASET_CARD.md", "LICENSE", ".env.example", "requirements.txt", "HANDOVER.md"):
        p = Path(fname)
        assert p.exists() and p.stat().st_size > 100, f"Missing or empty {fname}"
