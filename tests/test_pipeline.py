import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import yaml
import sys

# Add parent directory to path to allow import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datatypes import (
    AnnotatedEpisode,
    HandLandmarks,
    ObjectAnnotation,
    ActionSegment,
    CandidateSegment,
    ContactState,
    GraspType
)
from src.pipeline import EgoAnnotatePipeline


class TestPipelineIntegration(unittest.TestCase):

    def setUp(self):
        # Create a temporary workspace for test inputs, configurations, and outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # 1. Create a dummy 2-second 30fps video (60 frames)
        self.video_path = self.temp_path / "test_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.video_path), fourcc, 30.0, (224, 224))
        
        # Draw a simple scene: a red circle (object) and a moving hand shape (blue circle)
        # Red circle at center (112, 112)
        # Moving blue circle from top-left (20, 20) to center (112, 112)
        for i in range(60):
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            # Red circle (BGR: 0, 0, 255)
            cv2.circle(img, (112, 112), 20, (0, 0, 255), -1)
            
            # Moving hand shape (BGR: 255, 0, 0)
            hx = int(20 + (112 - 20) * (i / 59))
            hy = int(20 + (112 - 20) * (i / 59))
            cv2.circle(img, (hx, hy), 15, (255, 0, 0), -1)
            
            writer.write(img)
        writer.release()

        # 2. Create a minimal configuration yaml pointing to temporary test folders
        self.config_data = {
            "pipeline": {
                "target_fps": 30.0,
                "image_size": [224, 224],
                "frames_dir": str(self.temp_path / "frames"),
                "output_dir": str(self.temp_path / "output")
            },
            "hand_tracking": {
                "model_path": str(self.temp_path / "hand.task"),
                "num_hands": 2,
                "min_detection_confidence": 0.5,
                "min_tracking_confidence": 0.5,
                "smoothing_window": 5
            },
            "object_detection": {
                "keyframes_per_video": 3,
                "prompt": "mock-prompt"
            },
            "contact_detection": {
                "proximity_threshold_px": 25,
                "fingertip_indices": [4, 8, 12, 16, 20]
            },
            "grasp_classification": {
                "pinch_threshold": 0.05,
                "wrap_threshold": 0.15
            },
            "action_computation": {
                "compute_wrist_delta": True,
                "compute_finger_angles": True,
                "compute_gripper_state": True,
                "normalize_actions": True
            },
            "action_segmentation": {
                "prompt": "mock-segmentation"
            },
            "language_annotation": {
                "episode_prompt": "mock-episode-prompt",
                "segment_prompt": "mock-segment-prompt"
            },
            "gemini": {
                "model": "gemini-1.5-pro-latest",
                "api_key": None
            },
            "output": {
                "format": "json",
                "include_image_bytes": False,
                "save_viz_video": True
            }
        }

        self.config_path = self.temp_path / "test_config.yaml"
        with open(self.config_path, "w") as f:
            yaml.dump(self.config_data, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("src.hand_tracker.mp")
    @patch("src.hand_tracker.vision.HandLandmarker.create_from_options")
    @patch("google.generativeai.configure")
    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.upload_file")
    def test_full_pipeline_integration(self, mock_upload, mock_gen_model, mock_gen_configure, mock_landmarker_create, mock_mp):
        """Run full integration test of the pipeline and assert output file schemas."""
        # Set dummy env key to satisfy initialization validation
        os.environ["GEMINI_API_KEY"] = "mock-api-key"
        
        # Instantiate pipeline
        pipeline = EgoAnnotatePipeline(config_path=str(self.config_path))
        
        # Mock VLM / MediaPipe trackers to isolate network and target mock data
        # Define 60 hand tracking frames where hand moves from top-left (0.1, 0.1) to center (0.5, 0.5)
        mock_track_results = []
        for i in range(60):
            frac = i / 59
            x = np.ones(21) * (0.1 + 0.4 * frac)
            y = np.ones(21) * (0.1 + 0.4 * frac)
            z = np.zeros(21)
            
            # Set wrist (0)
            x[0], y[0] = 0.1 + 0.4 * frac, 0.1 + 0.4 * frac
            # Set index tip (8)
            x[8], y[8] = 0.1 + 0.4 * frac, 0.1 + 0.4 * frac
            
            hand_r = HandLandmarks(x=x, y=y, z=z, confidence=0.9, handedness="Right")
            mock_track_results.append({"left": None, "right": hand_r})
            
        pipeline.hand_tracker.track_frames = MagicMock(return_value=mock_track_results)

        # Mock Object Detector: object "red_circle" at BBox [0.4, 0.4, 0.6, 0.6]
        # (hand index tip 8 will end up inside the bbox, triggering contact!)
        mock_objects = [
            ObjectAnnotation(name="red_circle", location_description="center", touched=False, bbox=np.array([0.4, 0.4, 0.6, 0.6]))
        ]
        pipeline.object_detector.detect_objects = MagicMock(return_value=mock_objects)

        # Mock SignalSegmenter + SegmentLabeler (replaces old action_segmenter)
        mock_candidates = [
            CandidateSegment(
                start_frame=0, end_frame=30, start_time=0.0, end_time=1.0,
                transition_type="contact_on", contact_state="no_contact", grasp_type="none",
                object_name=None
            ),
            CandidateSegment(
                start_frame=30, end_frame=60, start_time=1.0, end_time=2.0,
                transition_type="grasp_change", contact_state="contact", grasp_type="precision_pinch",
                object_name="red_circle"
            ),
        ]
        mock_labeled_segments = [
            ActionSegment(name="approach", start_time=0.0, end_time=1.0, object_name="red_circle", hand_used="right", description="approaching"),
            ActionSegment(name="contact", start_time=1.0, end_time=2.0, object_name="red_circle", hand_used="right", description="touching"),
        ]
        
        pipeline.signal_segmenter.get_candidates = MagicMock(return_value=mock_candidates)
        if pipeline.segment_labeler is not None:
            pipeline.segment_labeler.label_segments = MagicMock(return_value=mock_labeled_segments)

        # Mock Language Generator
        pipeline.language_generator.generate_episode_description = MagicMock(return_value="approaching and touching a red circle")
        pipeline.language_generator.generate_segment_descriptions = MagicMock(return_value=["approaching", "touching"])

        # 3. Run the pipeline on test_video.mp4 with episode_id="test"
        episode = pipeline.process_video(str(self.video_path), episode_id="test")
        
        # Detailed test reports buffer
        report = []
        errors = []

        # 4. Verify outputs exist
        out_dir = Path(self.config_data["pipeline"]["output_dir"]) / "test"
        
        expected_files = [
            ("frame_annotations.json", out_dir / "frame_annotations.json"),
            ("metadata.json", out_dir / "metadata.json"),
            ("action_segments.json", out_dir / "action_segments.json"),
            ("summary.json", out_dir / "summary.json"),
            ("visualization.mp4", out_dir / "visualization.mp4")
        ]
        
        for name, path in expected_files:
            try:
                self.assertTrue(path.exists(), f"Output file does not exist: {path}")
                report.append(f"Output exist check: {name} -> EXISTS")
            except AssertionError as e:
                errors.append(str(e))

        # 5. Load frame_annotations.json and verify
        if (out_dir / "frame_annotations.json").exists():
            with open(out_dir / "frame_annotations.json", "r") as f:
                frames_json = json.load(f)
                
            try:
                self.assertGreaterEqual(len(frames_json), 50, f"Expected at least 50 frames, got {len(frames_json)}")
                report.append(f"Frames count check: {len(frames_json)} frames -> OK")
            except AssertionError as e:
                errors.append(str(e))
                
            for idx, frame in enumerate(frames_json):
                try:
                    self.assertIn("image_path", frame)
                    self.assertIn("timestamp", frame)
                    self.assertIn("action_wrist_delta", frame)
                    self.assertEqual(len(frame["action_wrist_delta"]), 3)
                    self.assertIn("action_gripper_openness", frame)
                    
                    openness = frame["action_gripper_openness"]
                    self.assertTrue(0.0 <= openness <= 1.0, f"Frame {idx}: gripper openness {openness} out of bounds")
                except AssertionError as e:
                    errors.append(f"Frame {idx} validation error: {e}")
                    break
            else:
                report.append("Frame schemas and action boundaries checks -> OK")

        # 6. Load metadata.json and verify
        if (out_dir / "metadata.json").exists():
            with open(out_dir / "metadata.json", "r") as f:
                meta_json = json.load(f)
                
            try:
                self.assertEqual(meta_json.get("episode_id"), "test")
                report.append("Metadata check: episode_id == 'test' -> OK")
            except AssertionError as e:
                errors.append(str(e))
                
            try:
                self.assertTrue(len(meta_json.get("task_description", "")) > 0)
                report.append("Metadata check: task_description non-empty -> OK")
            except AssertionError as e:
                errors.append(str(e))
                
            try:
                self.assertGreater(meta_json.get("num_frames", 0), 0)
                report.append(f"Metadata check: num_frames = {meta_json.get('num_frames')} -> OK")
            except AssertionError as e:
                errors.append(str(e))

        # 7. Print detailed test report
        print("\n" + "=" * 60)
        print("PIPELINE INTEGRATION TEST REPORT")
        print("=" * 60)
        for line in report:
            print(f" [PASS] {line}")
        if errors:
            print("\n [FAIL] Assertion errors occurred during run:")
            for err in errors:
                print(f"   - {err}")
            print("=" * 60 + "\n")
            raise AssertionError(f"Integration test failed with {len(errors)} assertions.")
        else:
            print("\n ALL INTEGRATION TEST ASSERTIONS PASSED SUCCESSFULLY!")
            print("=" * 60 + "\n")


if __name__ == "__main__":
    unittest.main()
