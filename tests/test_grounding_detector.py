"""Tests for Grounding DINO detector."""

import os
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datatypes import ObjectAnnotation
from src.grounding_detector import GroundingDINOConfig, GroundingDINODetector, create_grounding_detector


class TestGroundingDINODetector(unittest.TestCase):
    """Test cases for GroundingDINODetector."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Find a test image from data/raw_videos
        cls.test_video = Path(__file__).parent.parent / "data" / "raw_videos" / "my_video.mp4"
        if not cls.test_video.exists():
            cls.test_video = Path(__file__).parent.parent / "data" / "raw_videos" / "new_video.mp4"
        
        # Extract a single frame for testing
        cls.test_frame = None
        if cls.test_video.exists():
            cap = cv2.VideoCapture(str(cls.test_video))
            ret, frame = cap.read()
            cap.release()
            if ret:
                cls.test_frame = frame
    
    def test_detector_creation(self):
        """Test that detector can be created (or gracefully fails)."""
        config = GroundingDINOConfig(
            model_name="google/owlvit-base-patch32",
            confidence_threshold=0.3,
        )
        
        # Should not raise, may return None if transformers not available
        detector = create_grounding_detector(config)
        
        # If transformers is available, detector should be created
        # If not, it should return None gracefully
        if detector is not None:
            self.assertIsInstance(detector, GroundingDINODetector)
            self.assertTrue(detector.is_available())
    
    def test_detect_on_frame(self):
        """Test detection on a real frame."""
        if self.test_frame is None:
            self.skipTest("No test video available")
        
        config = GroundingDINOConfig(
            model_name="google/owlvit-base-patch32",
            confidence_threshold=0.3,
        )
        detector = create_grounding_detector(config)
        
        if detector is None:
            self.skipTest("Grounding DINO not available (transformers not installed or model load failed)")
        
        # Test with common egocentric objects
        object_names = ["hand", "cup", "table", "fabric", "machine"]
        results = detector.detect(self.test_frame, object_names)
        
        # Should return a list (may be empty if nothing detected)
        self.assertIsInstance(results, list)
        
        # If any detections, verify structure
        for obj in results:
            self.assertIsInstance(obj, ObjectAnnotation)
            self.assertIsInstance(obj.name, str)
            self.assertIsInstance(obj.bbox, np.ndarray)
            self.assertEqual(obj.bbox.shape, (4,))
            # Bbox should be normalized [0, 1]
            self.assertTrue(np.all(obj.bbox >= 0.0))
            self.assertTrue(np.all(obj.bbox <= 1.0))
            self.assertLessEqual(obj.bbox[0], obj.bbox[2])  # x_min <= x_max
            self.assertLessEqual(obj.bbox[1], obj.bbox[3])  # y_min <= y_max
    
    def test_empty_object_list(self):
        """Test detection with empty object list returns empty."""
        if self.test_frame is None:
            self.skipTest("No test video available")
        
        config = GroundingDINOConfig(model_name="google/owlvit-base-patch32")
        detector = create_grounding_detector(config)
        
        if detector is None:
            self.skipTest("Grounding DINO not available")
        
        results = detector.detect(self.test_frame, [])
        self.assertEqual(results, [])
    
    def test_bbox_normalization(self):
        """Test that bboxes are properly normalized to [0, 1]."""
        if self.test_frame is None:
            self.skipTest("No test video available")
        
        config = GroundingDINOConfig(model_name="google/owlvit-base-patch32")
        detector = create_grounding_detector(config)
        
        if detector is None:
            self.skipTest("Grounding DINO not available")
        
        # Use a simple object name likely to be detected
        results = detector.detect(self.test_frame, ["hand"])
        
        for obj in results:
            # All coordinates should be in [0, 1]
            self.assertTrue(np.all(obj.bbox >= 0.0))
            self.assertTrue(np.all(obj.bbox <= 1.0))
            # Valid box
            self.assertLess(obj.bbox[0], obj.bbox[2])
            self.assertLess(obj.bbox[1], obj.bbox[3])


if __name__ == "__main__":
    unittest.main()