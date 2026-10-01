"""Video frame sampler for the local VLM backend (Phase D).

K evenly-spaced frames, <=512px, deterministic temporal order with per-frame
timestamps. Uses a tiny synthetic video written with cv2 — no model needed.
"""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vlm_backend import sample_video_frames


def _make_video(path, n_frames=30, w=800, h=600, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for i in range(n_frames):
        frame = np.full((h, w, 3), (i * 8) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_even_spacing_and_timestamps(tmp_path):
    video = tmp_path / "v.mp4"
    _make_video(video, n_frames=30, fps=30)
    frames = sample_video_frames(str(video), k=5)
    assert len(frames) == 5
    ts = [t for t, _ in frames]
    # Evenly spaced, increasing, first ~0s, last ~(n-1)/fps.
    assert ts[0] == pytest.approx(0.0, abs=0.01)
    assert ts[-1] == pytest.approx(29 / 30, abs=0.01)
    for a, b in zip(ts, ts[1:]):
        assert b > a
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    assert max(gaps) - min(gaps) < 0.05  # ~even


def test_max_side_capped_at_512(tmp_path):
    video = tmp_path / "v.mp4"
    _make_video(video, n_frames=10, w=800, h=600)
    frames = sample_video_frames(str(video), k=3)
    for _, img in frames:
        w, h = img.size
        assert max(w, h) <= 512
        # Aspect ratio preserved: 800x600 -> 512x384.
        assert (w, h) == (512, 384)


def test_small_video_not_upscaled(tmp_path):
    video = tmp_path / "v.mp4"
    _make_video(video, n_frames=8, w=160, h=120)
    frames = sample_video_frames(str(video), k=4)
    for _, img in frames:
        assert img.size == (160, 120)


def test_deterministic_temporal_order(tmp_path):
    video = tmp_path / "v.mp4"
    _make_video(video, n_frames=24, fps=24)
    a = sample_video_frames(str(video), k=6)
    b = sample_video_frames(str(video), k=6)
    assert [t for t, _ in a] == [t for t, _ in b]
    assert all(np.array_equal(np.asarray(ai), np.asarray(bi))
               for (_, ai), (_, bi) in zip(a, b))


def test_k_clamped_to_frame_count(tmp_path):
    video = tmp_path / "v.mp4"
    _make_video(video, n_frames=5, fps=30)
    frames = sample_video_frames(str(video), k=20)
    assert len(frames) == 5


def test_missing_video_raises(tmp_path):
    with pytest.raises(ValueError, match="Cannot open video"):
        sample_video_frames(str(tmp_path / "nope.mp4"), k=4)


def test_timestamps_match_sampled_indices(tmp_path):
    video = tmp_path / "v.mp4"
    _make_video(video, n_frames=30, fps=30)
    frames = sample_video_frames(str(video), k=3)
    ts = [t for t, _ in frames]
    # linspace(0, 29, 3) -> indices 0, 14.5, 29 -> rounded frames 0, 15(or 14), 29
    assert ts[0] == pytest.approx(0.0, abs=0.02)
    assert ts[-1] == pytest.approx(29 / 30, abs=0.02)
    # Pixel check: the middle frame is bright (frame idx ~14/15 -> intensity ~112-120).
    mid = np.asarray(frames[1][1]).mean()
    assert 80 < mid < 160
