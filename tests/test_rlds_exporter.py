import os
import unittest
import unittest.mock
import numpy as np
import tempfile
import json
import h5py
from pathlib import Path

from src.rlds_exporter import rot6d_from_normal, export_to_rlds, GRASP_MAPPING


class TestRLDSExporter(unittest.TestCase):

    def test_rot6d_from_normal_identity(self):
        """Test rot6d fallback with zero/near-zero normal."""
        zero_normal = np.array([0.0, 0.0, 0.0])
        rot6d = rot6d_from_normal(zero_normal)
        # Should return identity rotation (first two columns of identity matrix)
        expected = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(rot6d, expected)

    def test_rot6d_from_normal_orthogonal(self):
        """Test rot6d orthonormality under normal conditions."""
        normal = np.array([0.0, 0.0, 1.0])
        rot6d = rot6d_from_normal(normal)
        x = rot6d[:3]
        y = rot6d[3:]
        
        # Verify length 1
        self.assertAlmostEqual(np.linalg.norm(x), 1.0, places=5)
        self.assertAlmostEqual(np.linalg.norm(y), 1.0, places=5)
        
        # Verify orthogonal to each other and normal
        self.assertAlmostEqual(np.dot(x, y), 0.0, places=5)
        self.assertAlmostEqual(np.dot(x, normal), 0.0, places=5)
        self.assertAlmostEqual(np.dot(y, normal), 0.0, places=5)

    def test_export_to_rlds(self):
        """Test the full RLDS HDF5 structure creation with mock data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            episode_id = "test_episode"
            ep_dir = tmp_path / episode_id
            ep_dir.mkdir(parents=True)

            # Mock image file
            img_rel_path = "frame_000000.png"
            # We will patch the image loading in our test or just write a small dummy png
            img_path = ep_dir / img_rel_path
            # Let's make a 224x224x3 dummy image
            dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
            import cv2
            cv2.imwrite(str(img_path), dummy_img)

            # Mock frame annotations JSON
            # 21 landmarks flat is 63 floats
            dummy_kp = [0.0] * 63
            # We want to make sure normal isn't degenerate (wrist 0, index 5, pinky 17)
            # index 0 (0,0,0)
            # index 5 (0.1, 0.0, 0.0) -> index 15,16,17
            # index 17 (0.0, 0.1, 0.0) -> index 51,52,53
            dummy_kp[15] = 0.1
            dummy_kp[51] = 0.1
            
            mock_annotations = [
                {
                    "frame_idx": 0,
                    "timestamp": 0.0,
                    "image_path": str(img_path.relative_to(tmp_path)), # relative to fake output dir
                    "left_hand_present": False,
                    "right_hand_present": True,
                    "left_hand_keypoints": [0.0] * 63,
                    "right_hand_keypoints": dummy_kp,
                    "left_contact": False,
                    "right_contact": True,
                    "left_grasp_type": None,
                    "right_grasp_type": "open",
                    "action_wrist_delta": [0.1, 0.2, 0.3],
                    "action_finger_angles": [0.5] * 15,
                    "action_gripper_openness": 0.8,
                    "language_instruction": "test instruction",
                }
            ]

            mock_metadata = {
                "episode_id": episode_id,
                "task_description": "test task",
                "num_frames": 1,
                "duration_seconds": 1.0,
            }

            with open(ep_dir / "frame_annotations.json", "w") as f:
                json.dump(mock_annotations, f)
            with open(ep_dir / "metadata.json", "w") as f:
                json.dump(mock_metadata, f)

            # Export
            # Mock the image search path relative to project_root to load our dummy frame
            with unittest.mock.patch("pathlib.Path.exists", return_value=True):
                with unittest.mock.patch("cv2.imread", return_value=dummy_img):
                    rlds_file = export_to_rlds(episode_id, tmp_dir)

            self.assertTrue(rlds_file.exists())

            # Verify contents
            with h5py.File(rlds_file, "r") as f:
                self.assertIn(episode_id, f)
                ep_grp = f[episode_id]
                
                self.assertIn("steps", ep_grp)
                steps_grp = ep_grp["steps"]
                
                self.assertIn("observation", steps_grp)
                obs_grp = steps_grp["observation"]
                
                self.assertIn("image", obs_grp)
                self.assertEqual(obs_grp["image"].shape, (1, 224, 224, 3))
                
                self.assertIn("wrist_translation", obs_grp)
                self.assertEqual(obs_grp["wrist_translation"].shape, (1, 3))
                # Metric space check: wrist_normalized * 0.5 (wrist is 0,0,0 so translation is 0)
                np.testing.assert_allclose(obs_grp["wrist_translation"][0], [0,0,0])
                
                self.assertIn("wrist_rotation", obs_grp)
                self.assertEqual(obs_grp["wrist_rotation"].shape, (1, 6))
                
                self.assertIn("hand_pose", obs_grp)
                self.assertEqual(obs_grp["hand_pose"].shape, (1, 15))
                self.assertEqual(obs_grp["hand_pose"].attrs["hand_pose_note"], "proxy using finger angles, labs should fit MANO PCA")
                
                self.assertIn("proprioception", obs_grp)
                self.assertEqual(obs_grp["proprioception"].shape, (1, 8))
                
                # Check proprioception contents:
                # [wrist_x, wrist_y, wrist_z, gripper, left_contact, right_contact, left_grasp, right_grasp]
                expected_proprio = [0.0, 0.0, 0.0, 0.8, 0.0, 1.0, 5.0, float(GRASP_MAPPING["open"])]
                np.testing.assert_allclose(obs_grp["proprioception"][0], expected_proprio)
                
                self.assertIn("language_instruction", obs_grp)
                self.assertEqual(obs_grp["language_instruction"][0].decode('utf-8'), "test instruction")
                
                self.assertIn("action", steps_grp)
                self.assertEqual(steps_grp["action"].shape, (1, 24))
                # First frame action wrist_rot_delta should be 0
                # action = [wrist_delta(3), wrist_rot_delta(6), finger_angles(15)]
                # wrist_delta = [0.1, 0.2, 0.3], wrist_rot_delta = [0]*6, finger_angles = [0.5]*15
                expected_action = [0.1, 0.2, 0.3] + [0.0]*6 + [0.5]*15
                np.testing.assert_allclose(steps_grp["action"][0], expected_action)
                
                self.assertIn("is_first", steps_grp)
                self.assertTrue(steps_grp["is_first"][0])
                
                self.assertIn("is_last", steps_grp)
                self.assertTrue(steps_grp["is_last"][0])
                
                self.assertIn("is_terminal", steps_grp)
                self.assertTrue(steps_grp["is_terminal"][0])
                
                self.assertIn("metadata", ep_grp)
                meta_grp = ep_grp["metadata"]
                self.assertEqual(meta_grp["task_description"][()].decode('utf-8'), "test task")
                self.assertEqual(meta_grp["num_frames"][()], 1)
                self.assertAlmostEqual(meta_grp["duration_seconds"][()], 1.0)


if __name__ == "__main__":
    unittest.main()
