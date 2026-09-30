"""Torch-free tests for scripts/merge_lerobot.py (Phase B).

Reuses the synthetic fixtures from tests/test_lerobot_exporter.py
(import, don't duplicate).
"""

import json
import logging
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from test_lerobot_exporter import _make_episode, _make_mp4, export_to_lerobot

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from merge_lerobot import find_lerobot_exports, merge_lerobot_datasets  # noqa: E402
from src.lerobot_exporter import compute_feature_stats  # noqa: E402

VIDEO_KEY = "observation.images.ego"


def _set_task_description(ep_dir: Path, episode_id: str, desc: str):
    """Rewrite the HDF5 task_description (the exporter's source of truth)."""
    with h5py.File(ep_dir / "episode_rlds.hdf5", "r+") as f:
        meta_grp = f[episode_id]["metadata"]
        del meta_grp["task_description"]
        meta_grp.create_dataset(
            "task_description", data=desc,
            dtype=h5py.string_dtype(encoding="utf-8"))


def _build_export(tmp_path: Path, name: str, n_frames: int = 6,
                  with_video: bool = False, fps: float = 10.0,
                  task: str | None = None, robot: str = "franka",
                  mode: str = "robot", with_robot_joints: bool = True):
    """Build one synthetic episode dir and export it to lerobot_v3/."""
    ep_dir = tmp_path / name
    ep_dir.mkdir(parents=True, exist_ok=True)
    video_path = None
    if with_video:
        mp4 = ep_dir / "capture.mp4"
        _make_mp4(mp4, n_frames=n_frames, fps=fps)
        video_path = str(mp4)
    _make_episode(ep_dir, name, n_frames=n_frames, with_robot=with_robot_joints,
                  video_path=video_path, target_robot=robot)
    if task is not None:
        _set_task_description(ep_dir, name, task)
    return export_to_lerobot(name, str(tmp_path), export_mode=mode)


def _run_validator(root: Path) -> int:
    script = (Path(__file__).resolve().parent.parent
              / "scripts" / "validate_lerobot_export.py")
    proc = subprocess.run([sys.executable, str(script), str(root)],
                          capture_output=True, text=True)
    return proc.returncode


class TestMergeLerobot(unittest.TestCase):

    def _three_episode_merge(self, tmp_path: Path, **kw):
        """2 video episodes @10fps + 1 video-less (fps defaults to 30);
        tasks: spoon / spoon (dup) / drawer."""
        ep_a = _build_export(tmp_path, "ep_a", n_frames=6, with_video=True,
                             task="pick up the spoon")
        ep_b = _build_export(tmp_path, "ep_b", n_frames=8, with_video=True,
                             task="pick up the spoon")
        ep_c = _build_export(tmp_path, "ep_c", n_frames=5, with_video=False,
                             task="open the drawer")
        merged = merge_lerobot_datasets([ep_a, ep_b, ep_c],
                                        tmp_path / "merged", **kw)
        return merged

    def test_merge_totals_and_layout(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            merged = self._three_episode_merge(tmp_path)

            with open(merged / "meta" / "info.json") as f:
                info = json.load(f)
            self.assertEqual(info["codebase_version"], "v3.0")
            self.assertEqual(info["total_episodes"], 3)
            self.assertEqual(info["total_frames"], 19)
            self.assertEqual(info["total_tasks"], 2)
            self.assertEqual(info["total_videos"], 2)
            self.assertEqual(info["total_chunks"], 1)
            self.assertEqual(info["splits"], {"train": "0:3"})
            self.assertEqual(info["export_mode"], "robot")
            self.assertEqual(info["robot_type"], "franka")
            # mixed fps: warn+allow, merged fps = first episode's
            self.assertAlmostEqual(info["fps"], 10.0)

            df = pd.read_parquet(merged / "data" / "chunk-000" / "file-000.parquet")
            self.assertEqual(len(df), 19)
            # global monotonic index 0..M-1
            np.testing.assert_array_equal(df["index"].to_numpy(), np.arange(19))
            # episode_index correct, frame_index resets per episode
            np.testing.assert_array_equal(
                df["episode_index"].to_numpy(), [0] * 6 + [1] * 8 + [2] * 5)
            np.testing.assert_array_equal(
                df["frame_index"].to_numpy(),
                list(range(6)) + list(range(8)) + list(range(5)))
            # task_index deduped: spoon->0, drawer->1
            np.testing.assert_array_equal(
                df["task_index"].to_numpy(), [0] * 6 + [0] * 8 + [1] * 5)
            # timestamps kept as per-episode video clock (10fps, 10fps, 30fps)
            for ep_i, fps in ((0, 10.0), (1, 10.0), (2, 30.0)):
                sl = df[df["episode_index"] == ep_i]
                np.testing.assert_allclose(
                    sl["timestamp"].to_numpy(),
                    sl["frame_index"].to_numpy() / fps, rtol=1e-3)

            # tasks.parquet deduped
            tasks = pd.read_parquet(merged / "meta" / "tasks.parquet")
            self.assertEqual(
                sorted(tasks["task"].tolist()), ["open the drawer", "pick up the spoon"])

    def test_merge_stats_recomputed_pooled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            merged = self._three_episode_merge(tmp_path)

            with open(merged / "meta" / "info.json") as f:
                info = json.load(f)
            with open(merged / "meta" / "stats.json") as f:
                stats = json.load(f)
            feature_keys = [k for k, v in info["features"].items()
                            if v["dtype"] != "video"]
            self.assertEqual(set(stats.keys()), set(feature_keys))

            # exact match with compute_feature_stats on the concatenated frames
            df = pd.read_parquet(merged / "data" / "chunk-000" / "file-000.parquet")
            arrays = {}
            for key in feature_keys:
                col = df[key]
                first = col.iloc[0]
                if isinstance(first, (list, np.ndarray)):
                    arrays[key] = np.stack(col.to_numpy())
                else:
                    arrays[key] = col.to_numpy()
            expected = compute_feature_stats(arrays)
            self.assertEqual(stats, expected)

    def test_merge_videos_copied_with_lookup_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            merged = self._three_episode_merge(tmp_path)

            v0 = merged / "videos" / VIDEO_KEY / "chunk-000" / "file-000.mp4"
            v1 = merged / "videos" / VIDEO_KEY / "chunk-000" / "file-001.mp4"
            self.assertTrue(v0.exists())
            self.assertTrue(v1.exists())

            ep_df = pd.read_parquet(
                merged / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
            self.assertEqual(len(ep_df), 3)
            # dataset_from_index continuity
            np.testing.assert_array_equal(
                ep_df["dataset_from_index"].to_numpy(), [0, 6, 14])
            np.testing.assert_array_equal(
                ep_df["dataset_to_index"].to_numpy(), [6, 14, 19])
            np.testing.assert_array_equal(
                ep_df["length"].to_numpy(), [6, 8, 5])
            # per-episode fps kept
            np.testing.assert_allclose(
                ep_df["fps"].to_numpy(), [10.0, 10.0, 30.0])
            # video lookup: rows 0,1 declare videos; row 2 is NaN
            self.assertEqual(int(ep_df[f"videos/{VIDEO_KEY}/file_index"].iloc[0]), 0)
            self.assertEqual(int(ep_df[f"videos/{VIDEO_KEY}/file_index"].iloc[1]), 1)
            self.assertTrue(pd.isna(ep_df[f"videos/{VIDEO_KEY}/file_index"].iloc[2]))
            self.assertAlmostEqual(
                float(ep_df[f"videos/{VIDEO_KEY}/to_timestamp"].iloc[0]), 5 / 10.0)

    def test_merge_mixed_fps_warns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with self.assertLogs("merge_lerobot", level="WARNING") as logs:
                self._three_episode_merge(tmp_path)
            self.assertTrue(any("Mixed fps" in m for m in logs.output),
                            f"no mixed-fps warning in {logs.output}")

    def test_merge_multi_chunk(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            merged = self._three_episode_merge(tmp_path, chunks_size=10)

            with open(merged / "meta" / "info.json") as f:
                info = json.load(f)
            self.assertEqual(info["total_chunks"], 3)
            for c in range(3):
                self.assertTrue(
                    (merged / "data" / f"chunk-{c:03d}" / "file-000.parquet").exists())
            ep_df = pd.read_parquet(
                merged / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
            # greedy packing: ep_a(6)->chunk0, ep_b(8)->chunk1, ep_c(5)->chunk2
            np.testing.assert_array_equal(
                ep_df["data/chunk_index"].to_numpy(), [0, 1, 2])
            # validator accepts multi-chunk layout + per-row video checks
            self.assertEqual(_run_validator(merged), 0)

    def test_validator_passes_merged_root(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            merged = self._three_episode_merge(tmp_path)
            self.assertEqual(_run_validator(merged), 0)

    def test_find_lerobot_exports(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ep_a = _build_export(tmp_path, "ep_a", n_frames=4, with_video=True)
            ep_b = _build_export(tmp_path, "ep_b", n_frames=4)
            found = find_lerobot_exports(tmp_path)
            self.assertEqual(found, sorted([ep_a.resolve(), ep_b.resolve()]))

    def test_merge_refuses_mixed_modes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ep_robot = _build_export(tmp_path, "ep_r", n_frames=4, mode="robot")
            ep_human = _build_export(tmp_path, "ep_h", n_frames=4, mode="human",
                                     with_robot_joints=False)
            with self.assertRaises(ValueError):
                merge_lerobot_datasets([ep_robot, ep_human], tmp_path / "merged")

    def test_merge_refuses_mixed_robot_type(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ep_a = _build_export(tmp_path, "ep_a", n_frames=4, robot="franka")
            ep_b = _build_export(tmp_path, "ep_b", n_frames=4, robot="unitree_g1")
            with self.assertRaises(ValueError) as ctx:
                merge_lerobot_datasets([ep_a, ep_b], tmp_path / "merged")
            self.assertIn("robot_type", str(ctx.exception))

    def test_merge_refuses_schema_mismatch_sidechannels(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # human mode: robot side-channels present only when retargeting ran
            ep_a = _build_export(tmp_path, "ep_a", n_frames=4, mode="human",
                                 with_robot_joints=True)
            ep_b = _build_export(tmp_path, "ep_b", n_frames=4, mode="human",
                                 with_robot_joints=False)
            with self.assertRaises(ValueError) as ctx:
                merge_lerobot_datasets([ep_a, ep_b], tmp_path / "merged")
            self.assertIn("observation.robot_joint_angles", str(ctx.exception))

    def test_merge_refuses_empty(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                merge_lerobot_datasets([], Path(tmp_dir) / "merged")

    def test_merge_dedupes_duplicate_dirs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ep_a = _build_export(tmp_path, "ep_a", n_frames=4, with_video=True)
            merged = merge_lerobot_datasets([ep_a, ep_a], tmp_path / "merged")
            with open(merged / "meta" / "info.json") as f:
                info = json.load(f)
            self.assertEqual(info["total_episodes"], 1)


if __name__ == "__main__":
    unittest.main()
