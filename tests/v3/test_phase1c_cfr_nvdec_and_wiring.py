"""Tests for Phase 1C CFR 30 FPS resampling, NVENC probe, and wired batched IK."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.egoannotate.config import PrivacyConfig
from src.egoannotate.privacy_v3 import export_sanitized_mp4
from src.egoannotate.retargeting_v3 import load_kinematic_chains, solve_arm_trajectory_ik
from src.egoannotate.video_io import VideoStreamReader


def _make_synthetic_mp4(path: Path, fps: float, n_frames: int) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (160, 120))
    for i in range(n_frames):
        frame = np.full((120, 160, 3), (i * 3) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_video_stream_reader_normalizes_60fps_to_30fps_cfr(tmp_path: Path) -> None:
    src = tmp_path / "clip_60fps.mp4"
    _make_synthetic_mp4(src, fps=60.0, n_frames=120)

    reader = VideoStreamReader(
        src,
        target_fps=30.0,
        normalize_fps_cfr=True,
        use_nvdec=False,
    )
    items = list(reader.iter_frames())
    assert 58 <= len(items) <= 62
    ts = [t for _, t, _ in items]
    diffs = np.diff(ts)
    assert np.all(diffs > 0.0)
    assert float(np.max(np.abs(diffs - (1.0 / 30.0)))) < 1e-6

    meta = reader.build_metadata(processed_frames=len(items))
    assert meta.processing_fps == 30.0
    assert meta.is_full_video_processed is True


def test_export_sanitized_mp4_resamples_to_30fps(tmp_path: Path) -> None:
    src = tmp_path / "clip_60fps.mp4"
    dst = tmp_path / "sanitized_30fps.mp4"
    _make_synthetic_mp4(src, fps=60.0, n_frames=60)

    audit = export_sanitized_mp4(
        src,
        dst,
        cfg=PrivacyConfig(enable_face_blur=False, strip_audio=True, use_nvenc=False),
        target_fps=30.0,
    )
    assert dst.exists()
    assert audit["output_fps"] == 30.0
    assert 28 <= audit["frames_written"] <= 32
    assert audit["privacy_pass"] is True


def test_retargeting_solve_arm_ik_delegates_to_batched_solver() -> None:
    _, chains = load_kinematic_chains(Path("configs/embodiments_v3/urdfs/franka_panda_7dof.urdf"))
    chain = next(iter(chains.values()))
    targets = np.tile(np.array([[0.42, 0.0, 0.45]], dtype=np.float32), (120, 1))
    names, q_traj, achieved, err_mm = solve_arm_trajectory_ik(
        chain=chain,
        targets_base_m=targets,
        dt_s=1.0 / 30.0,
        max_vel_radps=2.4,
        enforce_limits=True,
        smooth_window=7,
    )
    assert len(names) == 7
    assert q_traj.shape == (120, 7)
    assert achieved.shape == (120, 3)
    assert float(np.mean(err_mm)) < 10.0
