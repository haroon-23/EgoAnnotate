"""Integration test for end-to-end EgoAnnotatePipeline with retargeting enabled."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import yaml

# Ensure src/ importable from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datatypes import (
    HandLandmarks,
    ObjectAnnotation,
    ActionSegment,
    CandidateSegment,
)
from src.pipeline import EgoAnnotatePipeline


class TestPipelineRetargetingIntegration(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create a dummy 1-second 30fps video (30 frames)
        self.video_path = self.temp_path / "test_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.video_path), fourcc, 30.0, (224, 224))
        for i in range(30):
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            cv2.circle(img, (112, 112), 20, (0, 0, 255), -1)
            writer.write(img)
        writer.release()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_config(self, retargeting_enabled: bool) -> Path:
        out_dir = self.temp_path / "output"
        frames_dir = self.temp_path / "frames"
        config = {
            "pipeline": {
                "target_fps": 30,
                "image_size": [224, 224],
                "frames_dir": str(frames_dir),
                "output_dir": str(out_dir),
            },
            "hand_tracking": {
                "running_mode": "VIDEO",
                "num_hands": 2,
                "min_detection_confidence": 0.3,
                "min_tracking_confidence": 0.5,
            },
            "object_detection": {
                "keyframes_per_video": 1,
                "prompt": "Identify objects.",
            },
            "contact_detection": {
                "proximity_threshold_px": 25,
            },
            "grasp_classification": {
                "pinch_threshold": 0.05,
            },
            "action_computation": {
                "compute_wrist_delta": True,
            },
            "signal_segmentation": {
                "min_segment_duration_sec": 0.2,
            },
            "output": {
                "format": "json",
                "save_viz_video": False,
            },
            "retargeting": {
                "enable": retargeting_enabled,
                "config_path": "configs/retargeting_franka.yaml",
                "save_proof_video": True,
            },
        }
        cfg_file = self.temp_path / f"test_config_{retargeting_enabled}.yaml"
        with open(cfg_file, "w") as f:
            yaml.dump(config, f)
        return cfg_file

    def _mock_pipeline_stages(self, pipeline: EgoAnnotatePipeline):
        # Mock hand tracker
        mock_track_results = []
        for i in range(30):
            frac = i / 29.0
            x = np.zeros(21, dtype=np.float32)
            y = np.zeros(21, dtype=np.float32)
            z = np.zeros(21, dtype=np.float32)
            x[0], y[0], z[0] = 0.5, 0.5, 0.0
            x[5], y[5], z[5] = 0.6, 0.4, -0.05
            x[17], y[17], z[17] = 0.4, 0.4, -0.05
            x[4], y[4], z[4] = 0.55, 0.45, -0.03
            x[8], y[8], z[8] = 0.62, 0.35, -0.06
            hand_r = HandLandmarks(x=x, y=y, z=z, confidence=0.9, handedness="Right")
            mock_track_results.append({"left": None, "right": hand_r})

        pipeline.hand_tracker.track_frames = MagicMock(return_value=mock_track_results)

        # Mock object detector
        mock_objs = [
            ObjectAnnotation(name="mug", location_description="center", touched=True, bbox=np.array([0.4, 0.4, 0.6, 0.6]))
        ]
        pipeline.object_detector.detect_per_frame_objects_with_bboxes = MagicMock(
            return_value=[mock_objs] * 30
        )

        # Mock language generator
        pipeline.language_generator.generate_episode_description = MagicMock(return_value="pick up the mug")
        pipeline.language_generator.generate_segment_descriptions = MagicMock(return_value=["pick up the mug"])

    def test_pipeline_runs_with_retargeting_enabled(self):
        """Pipeline must run end-to-end with retargeting enabled and produce joint trajectories + proof video."""
        cfg_file = self._create_config(retargeting_enabled=True)
        pipeline = EgoAnnotatePipeline(str(cfg_file))
        self._mock_pipeline_stages(pipeline)

        self.assertTrue(pipeline.enable_retargeting)
        self.assertIsNotNone(pipeline.retargeter)

        episode = pipeline.process_video(str(self.video_path), episode_id="integration_test")

        self.assertEqual(episode.target_robot, "panda")
        self.assertEqual(len(episode.frames), 30)

        # Check frame retargeting properties
        first_frame = episode.frames[0]
        self.assertIsNotNone(first_frame.robot_joint_angles)
        self.assertEqual(len(first_frame.robot_joint_angles), 7)
        self.assertIsNotNone(first_frame.robot_gripper_opening_m)
        self.assertIsNotNone(first_frame.robot_reachable)

        # Check JSON export
        ep_dir = self.temp_path / "output" / "integration_test"
        self.assertTrue(ep_dir.exists())

        annot_path = ep_dir / "frame_annotations.json"
        self.assertTrue(annot_path.exists())

        with open(annot_path, "r") as f:
            annot_data = json.load(f)

        first_annot = annot_data[0]
        self.assertIn("robot_joint_angles", first_annot)
        self.assertIn("robot_gripper_opening_m", first_annot)
        self.assertIn("right_grasp_type", first_annot)
        self.assertIn("language_instruction", first_annot)
        self.assertEqual(len(first_annot["robot_joint_angles"]), 7)

        # Check proof video
        sbs_path = ep_dir / "side_by_side.mp4"
        self.assertTrue(sbs_path.exists(), f"Proof video not found at {sbs_path}")

    def test_pipeline_runs_with_retargeting_disabled(self):
        """Pipeline must run end-to-end without error when retargeting is disabled."""
        cfg_file = self._create_config(retargeting_enabled=False)
        pipeline = EgoAnnotatePipeline(str(cfg_file))
        self._mock_pipeline_stages(pipeline)

        self.assertFalse(pipeline.enable_retargeting)
        self.assertIsNone(pipeline.retargeter)

        episode = pipeline.process_video(str(self.video_path), episode_id="retargeting_disabled")
        self.assertEqual(episode.target_robot, "human_egocentric")
        self.assertIsNone(episode.frames[0].robot_joint_angles)


if __name__ == "__main__":
    unittest.main(verbosity=2)
