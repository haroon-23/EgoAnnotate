"""Tests for src/training/lerobot_dataset.py (Phase B). Torch-guarded."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

torch = __import__("pytest").importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from merge_lerobot import merge_lerobot_datasets  # noqa: E402
from test_lerobot_exporter import _make_episode, _make_mp4, export_to_lerobot  # noqa: E402
from src.training.lerobot_dataset import LeRobotParquetDataset  # noqa: E402


def _build_merge(tmp_path: Path, specs) -> Path:
    """specs: list of (name, n_frames, with_video). Returns merged root."""
    ep_dirs = []
    for name, n_frames, with_video in specs:
        ep_dir = tmp_path / name
        ep_dir.mkdir(parents=True, exist_ok=True)
        video_path = None
        if with_video:
            mp4 = ep_dir / "capture.mp4"
            _make_mp4(mp4, n_frames=n_frames, fps=10.0)
            video_path = str(mp4)
        _make_episode(ep_dir, name, n_frames=n_frames, with_robot=True,
                      video_path=video_path, target_robot="franka")
        ep_dirs.append(export_to_lerobot(name, str(tmp_path), export_mode="robot"))
    return merge_lerobot_datasets(ep_dirs, tmp_path / "merged")


class TestLeRobotParquetDataset(unittest.TestCase):

    def test_sample_shapes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            merged = _build_merge(tmp_path, [("ep_a", 20, True), ("ep_b", 20, True)])
            ds = LeRobotParquetDataset(merged, chunk_size=8, image_size=96)
            try:
                self.assertEqual(len(ds), 40)
                self.assertEqual(ds.state_dim, 8)
                self.assertEqual(ds.action_dim, 8)
                s = ds[0]
                self.assertEqual(tuple(s["state"].shape), (8,))
                self.assertEqual(tuple(s["images"].shape), (3, 96, 96))
                self.assertEqual(tuple(s["action_chunk"].shape), (8, 8))
                self.assertEqual(s["episode_index"], 0)
                # images are float32 in 0..1
                self.assertTrue(float(s["images"].min()) >= 0.0)
                self.assertTrue(float(s["images"].max()) <= 1.0)
                # state is normalized (not raw radians): mean |.| small-ish
                self.assertTrue(float(s["state"].abs().mean()) < 5.0)
            finally:
                ds.close()

    def test_chunk_tail_padding_repeats_last_frame(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            merged = _build_merge(tmp_path, [("ep_a", 20, True), ("ep_b", 20, True)])
            ds = LeRobotParquetDataset(merged, chunk_size=8, image_size=96)
            try:
                # last frame of episode 0 -> chunk must repeat the last action
                s = ds[19]
                raw = ds.unnormalize_action(s["action_chunk"].numpy())
                df = pd.read_parquet(merged / "data" / "chunk-000" / "file-000.parquet")
                last_action = np.stack(
                    df[df["episode_index"] == 0]["action"].to_numpy())[-1]
                for k in range(8):
                    np.testing.assert_allclose(raw[k], last_action, rtol=1e-4)
            finally:
                ds.close()

    def test_normalize_unnormalize_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            merged = _build_merge(tmp_path, [("ep_a", 20, True)])
            ds = LeRobotParquetDataset(merged, chunk_size=8, image_size=96)
            try:
                s = ds[5]
                raw = ds.unnormalize_action(s["action_chunk"].numpy())
                df = pd.read_parquet(merged / "data" / "chunk-000" / "file-000.parquet")
                expect = np.stack(df["action"].to_numpy())[5:13].astype(np.float32)
                np.testing.assert_allclose(raw, expect, rtol=1e-4)
            finally:
                ds.close()

    def test_video_less_episode_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            merged = _build_merge(tmp_path, [("ep_a", 20, True), ("ep_b", 20, False)])
            with self.assertRaises(ValueError) as ctx:
                LeRobotParquetDataset(merged, chunk_size=8, image_size=96)
            self.assertIn("has no video", str(ctx.exception))

    def test_episode_boundary_mapping(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            merged = _build_merge(tmp_path, [("ep_a", 20, True), ("ep_b", 12, True)])
            ds = LeRobotParquetDataset(merged, chunk_size=8, image_size=96)
            try:
                self.assertEqual(ds[0]["episode_index"], 0)
                self.assertEqual(ds[19]["episode_index"], 0)
                self.assertEqual(ds[20]["episode_index"], 1)
                self.assertEqual(ds[31]["episode_index"], 1)
            finally:
                ds.close()


if __name__ == "__main__":
    unittest.main()
