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
    ActionSegment
)
from src.visualizer import EgoVisualizer, VizConfig


class TestEgoVisualizer(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for writing dummy frame images and output videos
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)

        # Create 2 dummy images for frames
        self.frame_paths = []
        for i in range(2):
            img_path = self.test_path / f"frame_{i}.png"
            # Simple 224x224 black canvas image
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), img)
            self.frame_paths.append(str(img_path))

    def tearDown(self):
        self.test_dir.cleanup()

    def test_config_initialization(self):
        """Test default VizConfig connections and BGR hand colors."""
        config = VizConfig()
        self.assertEqual(config.left_hand_color, (0, 255, 0))
        self.assertEqual(config.right_hand_color, (255, 0, 255))
        self.assertTrue(len(config.connections) > 0)
        self.assertIn((0, 1), config.connections)

    def test_build_hand_description_logic(self):
        """Test HUD label text builder mapping logic."""
        config = VizConfig()
        visualizer = EgoVisualizer(config)
        
        # Test Not Tracked
        l1, l2, l3 = visualizer._build_hand_description("left", None, None, None, [])
        self.assertEqual(l1, "LEFT HAND")
        self.assertEqual(l2, "idle hovering")
        self.assertEqual(l3, "not tracked")
        
        # Test Tracked without contact
        x = np.ones(21) * 0.5
        y = np.ones(21) * 0.5
        z = np.zeros(21)
        hand = HandLandmarks(x=x, y=y, z=z, confidence=0.9, handedness="Right")
        grasp = GraspType(type="precision_pinch", confidence=0.7, thumb_index_distance=0.02, num_curled_fingers=0)
        contact = ContactState(fingers=np.zeros(5, dtype=bool), object_name=None, in_contact=False)
        
        l1, l2, l3 = visualizer._build_hand_description("right", hand, grasp, contact, [])
        self.assertEqual(l1, "RIGHT HAND")
        self.assertEqual(l2, "idle hovering")
        self.assertEqual(l3, "precision pinch without contact")

        # Test Tracked with active contact
        contact_active = ContactState(fingers=np.ones(5, dtype=bool), object_name="mug", in_contact=True)
        l1, l2, l3 = visualizer._build_hand_description("right", hand, grasp, contact_active, [])
        self.assertEqual(l1, "RIGHT HAND")
        self.assertEqual(l2, "precision pinch")
        self.assertEqual(l3, "interacting with mug")

    def test_render_episode_video_creation(self):
        """Test rendering a mock episode, writing joints, HUD panels, and verify output MP4 file exists."""
        config = VizConfig()
        visualizer = EgoVisualizer(config)
        
        # Build 2 frames with simple hands
        x = np.linspace(0.4, 0.6, 21)
        y = np.linspace(0.4, 0.6, 21)
        z = np.zeros(21)
        
        hand_l = HandLandmarks(x=x, y=y, z=z, confidence=0.9, handedness="Left")
        hand_r = HandLandmarks(x=x+0.1, y=y+0.1, z=z, confidence=0.8, handedness="Right")
        
        frame_0 = AnnotationFrame(
            frame_idx=0,
            timestamp=0.0,
            image_path=self.frame_paths[0],
            left_hand=hand_l,
            right_hand=hand_r,
            right_contact=ContactState(fingers=np.ones(5, dtype=bool), object_name="cup", in_contact=True),
            right_grasp=GraspType(type="power_wrap", confidence=0.7, thumb_index_distance=0.15, num_curled_fingers=4)
        )
        
        frame_1 = AnnotationFrame(
            frame_idx=1,
            timestamp=0.1,
            image_path=self.frame_paths[1],
            left_hand=hand_l,
            right_hand=hand_r
        )
        
        # Build episode
        episode = AnnotatedEpisode(
            episode_id="ep_001",
            video_path="dummy_src.mp4",
            task_description="moving cup",
            frames=[frame_0, frame_1],
            segments=[
                ActionSegment(name="pick_up", start_time=0.0, end_time=0.1, object_name="cup", hand_used="right", description="picking up cup")
            ],
            num_frames=2,
            duration_seconds=0.2
        )
        
        # Render output
        out_video_path = self.test_path / "annotated.mp4"
        visualizer.render_episode(episode, out_video_path)
        
        # Assert file exists and is larger than 0 bytes
        self.assertTrue(out_video_path.exists())
        self.assertGreater(out_video_path.stat().st_size, 0)
        
        # Optional: verify video file structure
        cap = cv2.VideoCapture(str(out_video_path))
        self.assertTrue(cap.isOpened())
        
        # Check frame count
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.assertEqual(frame_count, 2)
        
        cap.release()


if __name__ == "__main__":
    unittest.main()
