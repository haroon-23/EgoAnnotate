import json
import os
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd

from src.lerobot_exporter import (
    MissingRobotRetargetingError,
    compute_feature_stats,
    export_all_lerobot,
    export_to_lerobot,
)


def _make_episode(ep_dir: Path, episode_id: str, n_frames: int = 4,
                  with_robot: bool = False, video_path: str | None = None,
                  target_robot: str | None = None):
    """Build a synthetic episode dir: episode_rlds.hdf5 + metadata.json."""
    rng = np.random.default_rng(0)
    with h5py.File(ep_dir / "episode_rlds.hdf5", "w") as f:
        ep_grp = f.create_group(episode_id)
        steps_grp = ep_grp.create_group("steps")
        obs_grp = steps_grp.create_group("observation")
        meta_grp = ep_grp.create_group("metadata")

        obs_grp.create_dataset(
            "wrist_translation",
            data=(rng.random((n_frames, 3)) * 0.5).astype(np.float32))
        rot = np.zeros((n_frames, 6), dtype=np.float32)
        rot[:, 0] = 1.0
        rot[:, 4] = 1.0
        obs_grp.create_dataset("wrist_rotation", data=rot)
        obs_grp.create_dataset(
            "hand_pose", data=(rng.random((n_frames, 15)) * 0.9).astype(np.float32))
        proprio = np.zeros((n_frames, 8), dtype=np.float32)
        proprio[:, 3] = np.linspace(0.9, 0.2, n_frames)  # gripper openness
        obs_grp.create_dataset("proprioception", data=proprio)
        steps_grp.create_dataset(
            "action", data=(rng.random((n_frames, 24)) * 0.1).astype(np.float32))

        if with_robot:
            joints = np.linspace(0.0, 0.5, n_frames * 7,
                                 dtype=np.float32).reshape(n_frames, 7)
            obs_grp.create_dataset("robot_joint_angles", data=joints)
            obs_grp.create_dataset(
                "robot_gripper_opening_m",
                data=np.linspace(0.08, 0.01, n_frames,
                                 dtype=np.float32).reshape(n_frames, 1))

        utf8_type = h5py.string_dtype(encoding="utf-8")
        meta_grp.create_dataset("task_description", data="pick up the spoon",
                                dtype=utf8_type)

    meta = {"episode_id": episode_id, "task_description": "pick up the spoon"}
    if video_path:
        meta["video_path"] = video_path
    if target_robot:
        meta["target_robot"] = target_robot
    with open(ep_dir / "metadata.json", "w") as f:
        json.dump(meta, f)


def _make_mp4(path: Path, n_frames: int = 4, w: int = 64, h: int = 48, fps: float = 10.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        for i in range(n_frames):
            writer.write(np.full((h, w, 3), (i * 40) % 256, dtype=np.uint8))
    finally:
        writer.release()


class TestLeRobotExporter(unittest.TestCase):

    def test_export_to_lerobot_format(self):
        """Test the LeRobot v2.1 HDF5 to Parquet + MP4 dataset conversion."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            episode_id = "test_episode"
            ep_dir = tmp_path / episode_id
            ep_dir.mkdir(parents=True)

            # 1. Create a dummy frame_annotations.json to provide timestamps
            mock_annotations = [
                {
                    "frame_idx": 0,
                    "timestamp": 0.0,
                    "image_path": "data/frames/test_episode/frame_000000.png",
                    "left_hand_present": False,
                    "right_hand_present": True,
                },
                {
                    "frame_idx": 1,
                    "timestamp": 0.03333333333333333,
                    "image_path": "data/frames/test_episode/frame_000001.png",
                    "left_hand_present": False,
                    "right_hand_present": True,
                }
            ]
            with open(ep_dir / "frame_annotations.json", "w") as f:
                json.dump(mock_annotations, f)

            with open(ep_dir / "metadata.json", "w") as f:
                json.dump({"episode_id": episode_id, "task_description": "test task"}, f)

            # 2. Create mock episode_rlds.hdf5
            hdf5_path = ep_dir / "episode_rlds.hdf5"
            with h5py.File(hdf5_path, "w") as f:
                ep_grp = f.create_group(episode_id)
                steps_grp = ep_grp.create_group("steps")
                obs_grp = steps_grp.create_group("observation")
                meta_grp = ep_grp.create_group("metadata")

                # Mock datasets: 2 frames (N=2)
                # observation/wrist_translation [N, 3]
                obs_grp.create_dataset("wrist_translation", data=np.array([[0.1, 0.2, 0.3], [0.15, 0.25, 0.35]], dtype=np.float32))
                # observation/wrist_rotation [N, 6]
                obs_grp.create_dataset("wrist_rotation", data=np.array([[1, 0, 0, 0, 1, 0], [0.9, 0.1, 0, 0, 0.9, 0.1]], dtype=np.float32))
                # observation/hand_pose [N, 15]
                obs_grp.create_dataset("hand_pose", data=np.array([[0.5] * 15, [0.6] * 15], dtype=np.float32))
                # observation/proprioception [N, 8] -> gripper is at index 3
                # [wrist_x, wrist_y, wrist_z, gripper, left_contact, right_contact, left_grasp, right_grasp]
                obs_grp.create_dataset("proprioception", data=np.array([[0, 0, 0, 0.8, 0, 0, 0, 0], [0, 0, 0, 0.7, 0, 0, 0, 0]], dtype=np.float32))
                # action [N, 24]
                steps_grp.create_dataset("action", data=np.array([[0.05] * 24, [0.06] * 24], dtype=np.float32))

                # Metadata
                utf8_type = h5py.string_dtype(encoding="utf-8")
                meta_grp.create_dataset("task_description", data="test task", dtype=utf8_type)

            # 3. Mock image compilation to avoid reading nonexistent frames folder
            dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)

            with unittest.mock.patch("pathlib.Path.exists", return_value=True):
                with unittest.mock.patch("cv2.imread", return_value=dummy_frame):
                    with unittest.mock.patch("imageio.get_writer") as mock_writer_cls:
                        # Setup mock writer instance
                        mock_writer = unittest.mock.MagicMock()
                        mock_writer_cls.return_value = mock_writer

                        # Mock Path.glob to return some fake paths
                        with unittest.mock.patch("pathlib.Path.glob", return_value=[Path("frame_000000.png"), Path("frame_000001.png")]):
                            lerobot_dir = export_to_lerobot(episode_id, tmp_dir)

            # 4. Verify outputs
            self.assertTrue(lerobot_dir.exists())

            # Verify meta/info.json
            info_json_path = lerobot_dir / "meta" / "info.json"
            self.assertTrue(info_json_path.exists())
            with open(info_json_path, "r") as inf:
                info_data = json.load(inf)
                self.assertEqual(info_data["codebase_version"], "v3.0")
                self.assertEqual(info_data["total_frames"], 2)

                # Check names order in observation.state
                obs_names = info_data["features"]["observation.state"]["names"]
                self.assertEqual(obs_names[9], "gripper")
                self.assertEqual(obs_names[10], "f0")
                self.assertEqual(len(obs_names), 24)

            # Verify Parquet contents
            parquet_path = lerobot_dir / "data" / "chunk-000" / "file-000.parquet"
            self.assertTrue(parquet_path.exists())
            df = pd.read_parquet(parquet_path)

            self.assertEqual(len(df), 2)
            self.assertIn("observation.state", df.columns)
            self.assertIn("action", df.columns)

            # Verify observation.state values (wrist_trans (3) + wrist_rot (6) + gripper (1) + hand_pose_14 (14))
            expected_state_0 = [0.1, 0.2, 0.3] + [1.0, 0.0, 0.0, 0.0, 1.0, 0.0] + [0.8] + [0.5] * 14
            np.testing.assert_allclose(df["observation.state"].iloc[0], expected_state_0, rtol=1e-5)


class TestRobotExportMode(unittest.TestCase):
    """Robot-native export: 8D absolute joint+gripper state/action."""

    def test_robot_mode_features(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ep_dir = tmp_path / "ep1"
            ep_dir.mkdir(parents=True)
            _make_episode(ep_dir, "ep1", n_frames=4, with_robot=True,
                          target_robot="franka")

            lerobot_dir = export_to_lerobot("ep1", tmp_dir, export_mode="robot")

            with open(lerobot_dir / "meta" / "info.json") as f:
                info = json.load(f)
            self.assertEqual(info["robot_type"], "franka")
            self.assertEqual(info["export_mode"], "robot")
            self.assertEqual(info["action_convention"], "absolute_joint_positions")
            self.assertEqual(info["features"]["observation.state"]["shape"], [8])
            self.assertEqual(info["features"]["action"]["shape"], [8])
            self.assertEqual(info["features"]["observation.state"]["names"][-1], "gripper_m")

            df = pd.read_parquet(lerobot_dir / "data" / "chunk-000" / "file-000.parquet")
            state = np.stack(df["observation.state"].to_numpy())
            action = np.stack(df["action"].to_numpy())
            self.assertEqual(state.shape, (4, 8))
            # Absolute convention: action[t] == state[t+1]; last frame holds.
            np.testing.assert_allclose(action[:-1], state[1:], rtol=1e-5)
            np.testing.assert_allclose(action[-1], state[-1], rtol=1e-5)
            # Joints are the retargeted trajectory, gripper in meters.
            self.assertTrue(np.all(state[:, 7] >= 0.0))

            # stats.json covers exactly the info.json feature keys, with quantiles.
            with open(lerobot_dir / "meta" / "stats.json") as f:
                stats = json.load(f)
            feature_keys = {k for k, v in info["features"].items()
                            if v["dtype"] != "video"}
            self.assertEqual(set(stats.keys()), feature_keys)
            for key in ("observation.state", "action"):
                for agg in ("min", "max", "mean", "std", "count", "q01", "q99"):
                    self.assertIn(agg, stats[key])
            # No v2.x episodes_stats.parquet artifact.
            self.assertFalse((lerobot_dir / "meta" / "episodes_stats.parquet").exists())

    def test_robot_mode_missing_retargeting_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ep_dir = tmp_path / "ep1"
            ep_dir.mkdir(parents=True)
            _make_episode(ep_dir, "ep1", n_frames=4, with_robot=False)
            with self.assertRaises(MissingRobotRetargetingError):
                export_to_lerobot("ep1", tmp_dir, export_mode="robot")

    def test_export_all_skips_episodes_without_robot_data(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            for eid, with_robot in (("ep_ok", True), ("ep_no_robot", False)):
                ep_dir = tmp_path / eid
                ep_dir.mkdir(parents=True)
                _make_episode(ep_dir, eid, n_frames=3, with_robot=with_robot,
                              target_robot="franka")
            paths = export_all_lerobot(tmp_dir, export_mode="robot")
            self.assertEqual(len(paths), 1)
            self.assertTrue(str(paths[0]).endswith("ep_ok/lerobot_v3"))

    def test_invalid_mode_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ep_dir = tmp_path / "ep1"
            ep_dir.mkdir(parents=True)
            _make_episode(ep_dir, "ep1", n_frames=2)
            with self.assertRaises(ValueError):
                export_to_lerobot("ep1", tmp_dir, export_mode="alien")


class TestVideoExport(unittest.TestCase):
    """P2: real video columns written from the source capture video."""

    def test_video_written_from_metadata_video_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ep_dir = tmp_path / "ep1"
            ep_dir.mkdir(parents=True)
            src_mp4 = ep_dir / "capture.mp4"
            _make_mp4(src_mp4, n_frames=4, fps=10.0)
            _make_episode(ep_dir, "ep1", n_frames=4, video_path=str(src_mp4))

            lerobot_dir = export_to_lerobot("ep1", tmp_dir)

            mp4_path = (lerobot_dir / "videos" / "observation.images.ego"
                        / "chunk-000" / "file-000.mp4")
            self.assertTrue(mp4_path.exists())

            with open(lerobot_dir / "meta" / "info.json") as f:
                info = json.load(f)
            self.assertEqual(info["total_videos"], 1)
            self.assertAlmostEqual(info["fps"], 10.0, places=1)
            vfeat = info["features"]["observation.images.ego"]
            self.assertEqual(vfeat["dtype"], "video")
            for k in ("video.fps", "video.codec", "video.pix_fmt",
                      "video.height", "video.width"):
                self.assertIn(k, vfeat["info"])

            ep_df = pd.read_parquet(
                lerobot_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
            self.assertIn("videos/observation.images.ego/chunk_index", ep_df.columns)
            self.assertIn("videos/observation.images.ego/to_timestamp", ep_df.columns)

            # v3 R4: timestamp == frame_index / fps
            df = pd.read_parquet(lerobot_dir / "data" / "chunk-000" / "file-000.parquet")
            np.testing.assert_allclose(
                df["timestamp"].to_numpy(),
                df["frame_index"].to_numpy() / 10.0, rtol=1e-3)
            self.assertTrue(str(df["timestamp"].dtype).startswith("float32"))

    def test_no_video_when_source_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ep_dir = tmp_path / "ep1"
            ep_dir.mkdir(parents=True)
            _make_episode(ep_dir, "ep1", n_frames=3)  # no video_path in metadata
            lerobot_dir = export_to_lerobot("ep1", tmp_dir)
            with open(lerobot_dir / "meta" / "info.json") as f:
                info = json.load(f)
            self.assertEqual(info["total_videos"], 0)
            self.assertNotIn("observation.images.ego", info["features"])


class TestStatsAndValidation(unittest.TestCase):
    """P4 pooled stats correctness + P6 L0 validation ladder."""

    def test_compute_feature_stats_pooled(self):
        arr = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]])
        stats = compute_feature_stats({"observation.state": arr})
        s = stats["observation.state"]
        np.testing.assert_allclose(s["min"], [0.0, 10.0])
        np.testing.assert_allclose(s["max"], [4.0, 30.0])
        np.testing.assert_allclose(s["mean"], [2.0, 20.0])
        self.assertEqual(s["count"], 3)
        # Pooled quantiles over rows (not per-episode averaged).
        self.assertLess(s["q01"][0], s["q99"][0])

    def _run_validator(self, lerobot_dir: Path) -> int:
        script = Path(__file__).resolve().parent.parent / "scripts" / "validate_lerobot_export.py"
        proc = subprocess.run(
            [sys.executable, str(script), str(lerobot_dir)],
            capture_output=True, text=True)
        return proc.returncode

    def test_validator_passes_human_export_with_video(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ep_dir = tmp_path / "ep1"
            ep_dir.mkdir(parents=True)
            src_mp4 = ep_dir / "capture.mp4"
            _make_mp4(src_mp4, n_frames=4, fps=10.0)
            _make_episode(ep_dir, "ep1", n_frames=4, video_path=str(src_mp4))
            lerobot_dir = export_to_lerobot("ep1", tmp_dir)
            self.assertEqual(self._run_validator(lerobot_dir), 0)

    def test_validator_passes_robot_export(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ep_dir = tmp_path / "ep1"
            ep_dir.mkdir(parents=True)
            _make_episode(ep_dir, "ep1", n_frames=4, with_robot=True,
                          target_robot="franka")
            lerobot_dir = export_to_lerobot("ep1", tmp_dir, export_mode="robot")
            self.assertEqual(self._run_validator(lerobot_dir), 0)

    def test_validator_fails_on_bogus_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.assertEqual(self._run_validator(Path(tmp_dir)), 1)


if __name__ == "__main__":
    unittest.main()
