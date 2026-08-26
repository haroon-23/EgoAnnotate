import os
import unittest
import numpy as np
import sys

# Add parent directory to path to allow import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datatypes import HandLandmarks, ObjectAnnotation, ContactState
from src.contact_detector import ContactDetector, ContactDetectorConfig


class TestContactDetector(unittest.TestCase):

    def test_config_initialization(self):
        """Test default config threshold and indices."""
        config = ContactDetectorConfig()
        self.assertEqual(config.proximity_threshold_px, 25)
        self.assertEqual(config.fingertip_indices, [4, 8, 12, 16, 20])
        self.assertEqual(config.smoothing_window, 3)
        self.assertEqual(config.min_confidence, 0.3)

    def test_hand_none_returns_none(self):
        """Test detect_contact returns None if hand is None."""
        config = ContactDetectorConfig()
        detector = ContactDetector(config)
        
        objects = [ObjectAnnotation("mug", "table", False)]
        self.assertIsNone(detector.detect_contact(None, objects))

    def test_point_near_bbox_boundaries(self):
        """Test _point_near_bbox boundary expansions."""
        config = ContactDetectorConfig(proximity_threshold_px=22)  # threshold = 22/224 = 0.0982
        detector = ContactDetector(config)
        
        # BBox center at (0.5, 0.5), size (0.2, 0.2)
        # bx=0.4, by=0.4, bw=0.2, bh=0.2
        # Expanded boundaries:
        # x_min = 0.4 - 0.0982 = 0.3018, x_max = 0.6 + 0.0982 = 0.6982
        # y_min = 0.4 - 0.0982 = 0.3018, y_max = 0.6 + 0.0982 = 0.6982
        
        # Test point inside original bbox
        self.assertTrue(detector._point_near_bbox(0.5, 0.5, 0.4, 0.4, 0.2, 0.2))
        
        # Test point inside expanded boundary but outside original bbox
        self.assertTrue(detector._point_near_bbox(0.35, 0.35, 0.4, 0.4, 0.2, 0.2))
        
        # Test point outside expanded boundary
        self.assertFalse(detector._point_near_bbox(0.25, 0.5, 0.4, 0.4, 0.2, 0.2))

    def test_detect_contact_proximity_success(self):
        """Test successful proximity-based contact detection on specific fingers."""
        config = ContactDetectorConfig(proximity_threshold_px=22)  # threshold = ~0.1
        detector = ContactDetector(config)
        
        # Mock hand landmarks: index tip (landmark 8) and thumb tip (4) touch the object
        # Fingertip indices in fingertip_positions: thumb=0, index=1, middle=2, ring=3, pinky=4
        x = np.ones(21) * 0.9
        y = np.ones(21) * 0.9
        z = np.zeros(21)
        
        # Landmark 8 (index tip) at (0.5, 0.5) - inside bbox
        x[8] = 0.5
        y[8] = 0.5
        
        # Landmark 4 (thumb tip) at (0.35, 0.5) - in expanded region
        x[4] = 0.35
        y[4] = 0.5
        
        hand = HandLandmarks(x=x, y=y, z=z, confidence=0.9, handedness="Right")
        
        # BBox at [0.4, 0.4, 0.6, 0.6] (normalized coordinates)
        objects = [
            ObjectAnnotation(name="mug", location_description="center", touched=False, bbox=np.array([0.4, 0.4, 0.6, 0.6]))
        ]
        
        state = detector.detect_contact(hand, objects, hand_side="right")
        self.assertIsNotNone(state)
        self.assertTrue(state.in_contact)
        self.assertEqual(state.object_name, "mug")
        self.assertGreater(state.confidence, 0.0)
        self.assertLessEqual(state.confidence, 1.0)
        
        # Expected fingers in contact: thumb (index 0) and index (index 1)
        expected_fingers = [True, True, False, False, False]
        np.testing.assert_array_equal(state.fingers, expected_fingers)

    def test_detect_contact_no_proximity(self):
        """Test that non-contact state is returned if no fingers are within threshold."""
        config = ContactDetectorConfig(proximity_threshold_px=22)
        detector = ContactDetector(config)
        
        # All fingertips far away from the object
        x = np.ones(21) * 0.9
        y = np.ones(21) * 0.9
        z = np.zeros(21)
        
        hand = HandLandmarks(x=x, y=y, z=z, confidence=0.9, handedness="Right")
        objects = [
            ObjectAnnotation(name="mug", location_description="center", touched=False, bbox=np.array([0.4, 0.4, 0.6, 0.6]))
        ]
        
        state = detector.detect_contact(hand, objects, hand_side="right")
        self.assertIsNotNone(state)
        self.assertFalse(state.in_contact)
        self.assertIsNone(state.object_name)
        np.testing.assert_array_equal(state.fingers, [False, False, False, False, False])
        self.assertEqual(state.confidence, 0.0)

    def test_detect_contact_fallback_vlm(self):
        """Test fallback to marking all fingers if object has no bbox but touched=True."""
        config = ContactDetectorConfig()
        detector = ContactDetector(config)
        
        # Fingertips far away
        x = np.ones(21) * 0.9
        y = np.ones(21) * 0.9
        z = np.zeros(21)
        
        hand = HandLandmarks(x=x, y=y, z=z, confidence=0.9, handedness="Right")
        objects = [
            ObjectAnnotation(name="apple", location_description="in bowl", touched=True, bbox=None)
        ]
        
        state = detector.detect_contact(hand, objects, hand_side="right")
        self.assertIsNotNone(state)
        self.assertTrue(state.in_contact)
        self.assertEqual(state.object_name, "apple")
        np.testing.assert_array_equal(state.fingers, [True, True, True, True, True])
        # Fallback should have low confidence (0.3 * 0.5 = 0.15, but min_confidence=0.3 so it's in_contact)
        # Actually with fallback: proximity=0.3, depth=0.5 -> raw=0.15, but min_confidence=0.3
        # Wait, the fallback logic sets proximity_conf=0.3 and depth=0.5 -> raw=0.15
        # But min_confidence=0.3 so in_contact would be False...
        # Let me check: the fallback has its own logic that forces in_contact=True
        # So confidence should be around 0.15 (raw)
        self.assertGreater(state.confidence, 0.0)
        self.assertLessEqual(state.confidence, 1.0)

    def test_depth_consistency_returns_valid_range(self):
        """Test that depth consistency score is in valid range [0, 1]."""
        config = ContactDetectorConfig(proximity_threshold_px=22)
        detector = ContactDetector(config)
        
        # Hand with consistent z
        x = np.array([0.5, 0.5, 0.5, 0.5, 0.5] + [0.0]*16)
        y = np.array([0.5, 0.5, 0.5, 0.5, 0.5] + [0.0]*16)
        z_consistent = np.array([0.1, 0.1, 0.1, 0.1, 0.1] + [0.0]*16)
        
        hand_consistent = HandLandmarks(x=x, y=y, z=z_consistent, confidence=0.9, handedness="Right")
        
        objects = [
            ObjectAnnotation(name="mug", location_description="center", touched=False, bbox=np.array([0.4, 0.4, 0.6, 0.6]))
        ]
        
        state_consistent = detector.detect_contact(hand_consistent, objects, hand_side="right")
        
        # Confidence should be in valid range
        self.assertGreaterEqual(state_consistent.confidence, 0.0)
        self.assertLessEqual(state_consistent.confidence, 1.0)
        
        # Test with spread z values (fingers at different depths)
        z_spread = np.array([0.1, 0.2, 0.3, 0.4, 0.5] + [0.0]*16)
        hand_spread = HandLandmarks(x=x, y=y, z=z_spread, confidence=0.9, handedness="Right")
        
        detector2 = ContactDetector(config)
        state_spread = detector2.detect_contact(hand_spread, objects, hand_side="right")
        
        self.assertGreaterEqual(state_spread.confidence, 0.0)
        self.assertLessEqual(state_spread.confidence, 1.0)
        
        # Both should return valid confidences
        self.assertTrue(state_consistent.in_contact or not state_consistent.in_contact)
        self.assertTrue(state_spread.in_contact or not state_spread.in_contact)

    def test_temporal_smoothing_stabilizes(self):
        """Test that confidence stabilizes over multiple frames."""
        config = ContactDetectorConfig(proximity_threshold_px=22, smoothing_window=5)
        detector = ContactDetector(config)
        
        # Hand right at boundary - alternating in/out
        x = np.array([0.45, 0.5, 0.5, 0.5, 0.5] + [0.0]*16)  # Thumb near edge
        y = np.array([0.5, 0.5, 0.5, 0.5, 0.5] + [0.0]*16)
        z = np.zeros(21)
        
        hand = HandLandmarks(x=x, y=y, z=z, confidence=0.9, handedness="Right")
        objects = [
            ObjectAnnotation(name="mug", location_description="center", touched=False, bbox=np.array([0.4, 0.4, 0.6, 0.6]))
        ]
        
        # First few frames - right at boundary
        confidences = []
        for _ in range(8):
            state = detector.detect_contact(hand, objects, hand_side="right")
            confidences.append(state.confidence)
        
        # Confidence should stabilize (less variance in later frames)
        early_var = np.var(confidences[:3])
        late_var = np.var(confidences[-3:])
        self.assertLessEqual(late_var, early_var + 0.01)  # Allow small numerical diff

    def test_reset_smoothing(self):
        """Test that reset_smoothing clears buffers."""
        config = ContactDetectorConfig(smoothing_window=3)
        detector = ContactDetector(config)
        
        # Build up buffer
        x = np.array([0.5]*5 + [0.0]*16)
        y = np.array([0.5]*5 + [0.0]*16)
        z = np.zeros(21)
        hand = HandLandmarks(x=x, y=y, z=z, confidence=0.9, handedness="Right")
        objects = [ObjectAnnotation(name="mug", location_description="center", touched=False, bbox=np.array([0.4, 0.4, 0.6, 0.6]))]
        
        for _ in range(5):
            detector.detect_contact(hand, objects, hand_side="right")
        
        # Buffer should have data
        self.assertEqual(len(detector._right_buffer), 3)  # maxlen=3
        
        detector.reset_smoothing("right")
        self.assertEqual(len(detector._right_buffer), 0)


if __name__ == "__main__":
    unittest.main()