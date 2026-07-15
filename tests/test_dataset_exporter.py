import json
import os
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
import sys

# Add parent directory to path to allow import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datatypes import (
    AnnotatedEpisode,
    AnnotationFrame,
    HandLandmarks,
    ObjectAnnotation,
    ContactState,
    GraspType,
    ActionSegment,
    RobotAgnosticAction
)
from src.dataset_exporter import DatasetExporter, ExporterConfig


class TestDatasetExporter(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for exporter test outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

        # Create dummy frame images
        self.frame_paths = []
        for i in range(2):
            img_path = self.output_dir / f"frame_{i}.png"
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), img)
            self.frame_paths.append(str(img_path))

        # Build mock episode data structures
        x = np.linspace(0.4, 0.6, 21)
        y = np.linspace(0.4, 0.6, 21)
        z = np.zeros(21)
        
        hand_l = HandLandmarks(x=x, y=y, z=z, confidence=0.9, handedness="Left")
        hand_r = HandLandmarks(x=x, y=y, z=z, confidence=0.9, handedness="Right")
        
        frame_0 = AnnotationFrame(
            frame_idx=0,
            timestamp=0.0,
            image_path=self.frame_paths[0],
            left_hand=hand_l,
            right_hand=hand_r,
            left_contact=ContactState(fingers=np.ones(5, dtype=bool), object_name="mug", in_contact=True),
            right_contact=ContactState(fingers=np.zeros(5, dtype=bool), object_name=None, in_contact=False),
            left_grasp=GraspType(type="power_wrap", confidence=0.7, thumb_index_distance=0.15, num_curled_fingers=4),
            right_grasp=GraspType(type="open", confidence=0.8, thumb_index_distance=0.20, num_curled_fingers=0),
            action=RobotAgnosticAction(
                wrist_delta=np.array([0.1, -0.2, 0.3], dtype=np.float32),
                finger_angles=np.ones(15, dtype=np.float32) * 0.9,
                gripper_openness=0.8,
                hand_orientation=np.array([0, 0, 1, 0], dtype=np.float32)
            ),
            frame_description="picking up a mug",
            action_segment="pick_up"
        )
        
        frame_1 = AnnotationFrame(
            frame_idx=1,
            timestamp=0.1,
            image_path=self.frame_paths[1],
            left_hand=hand_l,
            right_hand=None,
            left_contact=ContactState(fingers=np.zeros(5, dtype=bool), object_name=None, in_contact=False),
            left_grasp=GraspType(type="open", confidence=0.8, thumb_index_distance=0.20, num_curled_fingers=0),
            action=RobotAgnosticAction(
                wrist_delta=np.zeros(3, dtype=np.float32),
                finger_angles=np.zeros(15, dtype=np.float32),
                gripper_openness=1.0,
                hand_orientation=np.array([0, 0, 1, 0], dtype=np.float32)
            ),
            frame_description="hovering hand",
            action_segment="idle"
        )

        self.episode = AnnotatedEpisode(
            episode_id="ep_test_999",
            video_path="mock_video.mp4",
            task_description="testing export task",
            frames=[frame_0, frame_1],
            segments=[
                ActionSegment(name="pick_up", start_time=0.0, end_time=0.1, object_name="mug", hand_used="left", description="picking up a mug")
            ],
            num_frames=2,
            duration_seconds=0.2
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_config_initialization(self):
        """Test default ExporterConfig values."""
        config = ExporterConfig(output_dir=str(self.output_dir))
        self.assertEqual(config.format, "json")
        self.assertFalse(config.include_image_bytes)
        self.assertTrue(config.save_viz_video)

    def test_export_episode_json(self):
        """Test complete JSON export flow and assert all required files and fields exist."""
        config = ExporterConfig(output_dir=str(self.output_dir), format="json", save_viz_video=True)
        exporter = DatasetExporter(config)
        
        episode_dir = exporter.export_episode(self.episode)
        
        # Verify output directory path
        self.assertEqual(episode_dir, self.output_dir / "ep_test_999")
        self.assertTrue(episode_dir.exists())
        
        # Verify expected files exist
        self.assertTrue((episode_dir / "metadata.json").exists())
        self.assertTrue((episode_dir / "action_segments.json").exists())
        self.assertTrue((episode_dir / "frame_annotations.json").exists())
        self.assertTrue((episode_dir / "summary.json").exists())
        self.assertTrue((episode_dir / "visualization.mp4").exists())
        
        # Verify metadata
        with open(episode_dir / "metadata.json", "r") as f:
            meta = json.load(f)
        self.assertEqual(meta["episode_id"], "ep_test_999")
        self.assertEqual(meta["num_frames"], 2)
        
        # Verify frame annotations schema
        with open(episode_dir / "frame_annotations.json", "r") as f:
            frames_json = json.load(f)
            
        self.assertEqual(len(frames_json), 2)
        f0 = frames_json[0]
        
        # Validate all required fields
        required_fields = [
            "frame_idx", "timestamp", "image_path",
            "left_hand_present", "right_hand_present",
            "left_hand_keypoints", "right_hand_keypoints",
            "left_contact", "left_contact_object",
            "right_contact", "right_contact_object",
            "left_grasp_type", "right_grasp_type",
            "action_wrist_delta", "action_finger_angles", "action_gripper_openness",
            "language_instruction", "action_segment"
        ]
        for field in required_fields:
            self.assertIn(field, f0)
            
        # Specific assertions on first frame fields
        self.assertEqual(f0["frame_idx"], 0)
        self.assertTrue(f0["left_hand_present"])
        self.assertTrue(f0["right_hand_present"])
        self.assertEqual(len(f0["left_hand_keypoints"]), 63)
        self.assertTrue(f0["left_contact"])
        self.assertEqual(f0["left_contact_object"], "mug")
        self.assertEqual(f0["left_grasp_type"], "power_wrap")
        
        # Action delta check
        self.assertAlmostEqual(f0["action_wrist_delta"][0], 0.1)
        self.assertEqual(len(f0["action_finger_angles"]), 15)
        self.assertAlmostEqual(f0["action_gripper_openness"], 0.8)
        self.assertEqual(f0["language_instruction"], "picking up a mug")
        self.assertEqual(f0["action_segment"], "pick_up")
        
        # Verify summary statistics
        with open(episode_dir / "summary.json", "r") as f:
            summary = json.load(f)
            
        self.assertEqual(summary["total_frames"], 2)
        self.assertEqual(summary["left_contact_frames"], 1)
        self.assertEqual(summary["right_contact_frames"], 0)
        self.assertEqual(summary["left_grasp_distribution"]["power_wrap"], 1)
        self.assertEqual(summary["left_grasp_distribution"]["open"], 1)

    def test_export_episode_parquet_graceful_fallback(self):
        """Test Parquet export run (assert it runs and falls back to JSON or produces Parquet)."""
        config = ExporterConfig(output_dir=str(self.output_dir), format="parquet", save_viz_video=False)
        exporter = DatasetExporter(config)
        
        episode_dir = exporter.export_episode(self.episode)
        self.assertTrue(episode_dir.exists())
        
        # At least one of frame_annotations.json or frame_annotations.parquet must exist
        has_json = (episode_dir / "frame_annotations.json").exists()
        has_parquet = (episode_dir / "frame_annotations.parquet").exists()
        self.assertTrue(has_json or has_parquet)


if __name__ == "__main__":
    unittest.main()
