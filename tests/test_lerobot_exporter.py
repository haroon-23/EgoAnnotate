import os
import json
import unittest
import unittest.mock
import numpy as np
import tempfile
import pandas as pd
import h5py
from pathlib import Path

from src.lerobot_exporter import export_to_lerobot


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


if __name__ == "__main__":
    unittest.main()
