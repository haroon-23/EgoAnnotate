"""Unit tests for src/mcap_exporter.py (MCAP v0 zero-dependency writer and verifier)."""
from pathlib import Path
import pytest
import numpy as np

from src.datatypes import AnnotatedEpisode, AnnotationFrame, HandLandmarks, ActionSegment
from src.mcap_exporter import export_episode_to_mcap, verify_mcap_file, MCAP_MAGIC


def test_export_and_verify_mcap(tmp_path: Path):
    # Construct minimal synthetic episode
    frames = [
        AnnotationFrame(
            frame_idx=0,
            timestamp=0.0,
            image_path="frame_0.jpg",
            right_hand=HandLandmarks(
                x=np.zeros(21, dtype=np.float32),
                y=np.zeros(21, dtype=np.float32),
                z=np.zeros(21, dtype=np.float32),
                confidence=0.9,
                handedness="Right",
            ),
        ),
        AnnotationFrame(
            frame_idx=1,
            timestamp=1.0 / 30.0,
            image_path="frame_1.jpg",
        ),
    ]
    segments = [
        ActionSegment(
            name="pick",
            start_time=0.0,
            end_time=0.5,
            object_name="cup",
            hand_used="right",
        )
    ]
    ep = AnnotatedEpisode(
        episode_id="mcap_test_ep",
        video_path="dummy.mp4",
        task_description="pick up the cup",
        frames=frames,
        segments=segments,
        num_frames=len(frames),
        duration_seconds=len(frames) / 30.0,
        target_robot="franka_panda",
    )

    out_mcap = tmp_path / "test.mcap"
    export_episode_to_mcap(ep, out_mcap)

    assert out_mcap.exists()
    assert out_mcap.stat().st_size > len(MCAP_MAGIC) * 2

    info = verify_mcap_file(out_mcap)
    assert info["valid_magic"] is True
    assert info["total_messages"] == 1 + 2 * len(frames)  # 1 metadata + (human+robot)*2
    assert "/episode/metadata" in info["topics"]
    assert "/human/frame" in info["topics"]
    assert "/robot/frame" in info["topics"]
    assert info["topics"]["/human/frame"] == 2
    assert info["topics"]["/robot/frame"] == 2
