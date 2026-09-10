import os
import unittest
import numpy as np
import sys

# Add parent directory to path to allow import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datatypes import HandLandmarks, GraspType
from src.grasp_classifier import GraspClassifier, GraspClassifierConfig


class TestGraspClassifier(unittest.TestCase):

    def test_config_initialization(self):
        """Test default config thresholds and indices."""
        config = GraspClassifierConfig()
        self.assertEqual(config.pinch_threshold, 0.05)
        self.assertEqual(config.wrap_threshold, 0.15)
        self.assertEqual(config.pip_indices, [6, 10, 14, 18])
        self.assertEqual(config.tip_indices, [8, 12, 16, 20])

    def test_hand_none_returns_none(self):
        """Test classify returns None if hand is None."""
        config = GraspClassifierConfig()
        detector = GraspClassifier(config)
        self.assertIsNone(detector.classify(None))

    def test_classify_precision_pinch(self):
        """Test classification of precision_pinch (small thumb-index distance)."""
        config = GraspClassifierConfig()
        classifier = GraspClassifier(config)
        
        # Landmarks: let's put wrist at 0
        x = np.ones(21) * 0.5
        y = np.ones(21) * 0.5
        z = np.zeros(21)
        
        # Position thumb tip (4) and index tip (8) close together
        x[4] = 0.50
        y[4] = 0.50
        x[8] = 0.52 # distance = 0.02 < 0.05
        y[8] = 0.50
        
        # Set all fingers as straight (not curled)
        # Wrist is at (0.5, 0.5)
        # index: pip(6) at 0.45, tip(8) at 0.52 (d_tip > d_pip -> not curled)
        x[6] = 0.49
        # middle: pip(10) at 0.49, tip(12) at 0.51
        x[10] = 0.49
        x[12] = 0.51
        # ring: pip(14) at 0.49, tip(16) at 0.51
        x[14] = 0.49
        x[16] = 0.51
        # pinky: pip(18) at 0.49, tip(20) at 0.51
        x[18] = 0.49
        x[20] = 0.51
        
        hand = HandLandmarks(x=x, y=y, z=z, confidence=0.8, handedness="Left")
        
        result = classifier.classify(hand)
        self.assertIsNotNone(result)
        self.assertEqual(result.type, "precision_pinch")
        self.assertEqual(result.num_curled_fingers, 0)
        self.assertLess(result.thumb_index_distance, 0.05)

    def test_classify_power_wrap(self):
        """Test classification of power_wrap (>= 3 curled fingers, thumb-index distance > 0.15)."""
        config = GraspClassifierConfig()
        classifier = GraspClassifier(config)
        
        # Wrist at (0, 0, 0)
        x = np.zeros(21)
        y = np.zeros(21)
        z = np.zeros(21)
        
        # Thumb tip (4) and index tip (8) far apart (dist = 0.20 > 0.15)
        x[4] = 0.20
        x[8] = 0.0
        
        # Curl 3 fingers (index, middle, ring). tip closer to wrist than PIP
        # Index: PIP(6) dist = 0.3, tip(8) dist = 0.0 -> curled
        x[6] = 0.3
        x[8] = 0.0
        # Middle: PIP(10) dist = 0.3, tip(12) dist = 0.1 -> curled
        x[10] = 0.3
        x[12] = 0.1
        # Ring: PIP(14) dist = 0.3, tip(16) dist = 0.1 -> curled
        x[14] = 0.3
        x[16] = 0.1
        # Pinky: PIP(18) dist = 0.3, tip(20) dist = 0.4 -> NOT curled
        x[18] = 0.3
        x[20] = 0.4
        
        hand = HandLandmarks(x=x, y=y, z=z, confidence=0.9, handedness="Right")
        
        result = classifier.classify(hand)
        self.assertIsNotNone(result)
        self.assertEqual(result.type, "power_wrap")
        self.assertEqual(result.num_curled_fingers, 3)
        self.assertGreater(result.thumb_index_distance, 0.15)

    def test_classify_hook(self):
        """Test classification of hook (>= 2 curled fingers, thumb-index distance > 0.105)."""
        config = GraspClassifierConfig()
        classifier = GraspClassifier(config)
        
        # Wrist at (0, 0, 0)
        x = np.zeros(21)
        y = np.zeros(21)
        z = np.zeros(21)
        
        # Thumb tip (4) and index tip (8) distance = 0.12 (between 0.105 and 0.15)
        x[4] = 0.12
        x[8] = 0.0
        
        # Curl 2 fingers (middle, ring)
        # Index (5-8): PIP(6) dist = 0.3, tip(8) dist = 0.0 -> curled (Wait, if index tip is at 0.0, it is curled, so index is curled)
        # Let's place index tip at 0.4 -> NOT curled
        x[6] = 0.3
        x[8] = 0.4
        # Middle (9-12): PIP(10) dist = 0.3, tip(12) dist = 0.1 -> curled
        x[10] = 0.3
        x[12] = 0.1
        # Ring (13-16): PIP(14) dist = 0.3, tip(16) dist = 0.1 -> curled
        x[14] = 0.3
        x[16] = 0.1
        # Pinky (17-20): PIP(18) dist = 0.3, tip(20) dist = 0.4 -> NOT curled
        x[18] = 0.3
        x[20] = 0.4
        
        hand = HandLandmarks(x=x, y=y, z=z, confidence=0.9, handedness="Right")
        
        result = classifier.classify(hand)
        self.assertIsNotNone(result)
        self.assertEqual(result.type, "hook")
        self.assertEqual(result.num_curled_fingers, 2)
        self.assertGreater(result.thumb_index_distance, 0.105)

    def test_classify_open(self):
        """Test classification of open hand (0 curled fingers, dist > pinch)."""
        config = GraspClassifierConfig()
        classifier = GraspClassifier(config)
        
        # Wrist at (0, 0, 0)
        x = np.zeros(21)
        y = np.zeros(21)
        z = np.zeros(21)
        
        # Thumb tip (4) and index tip (8) distance = 0.20
        x[4] = 0.20
        x[8] = 0.4
        
        # All fingers extended (tip-to-wrist > PIP-to-wrist)
        for pip_idx, tip_idx in zip(config.pip_indices, config.tip_indices):
            x[pip_idx] = 0.3
            x[tip_idx] = 0.4
            
        hand = HandLandmarks(x=x, y=y, z=z, confidence=0.9, handedness="Right")
        
        result = classifier.classify(hand)
        self.assertIsNotNone(result)
        self.assertEqual(result.type, "open")
        self.assertEqual(result.num_curled_fingers, 0)


class TestTemporalGraspVoter(unittest.TestCase):

    def test_majority_voting_and_unknown_inheritance(self):
        """Test TemporalGraspVoter 5-frame majority voting and nearest confident label inheritance."""
        from src.grasp_classifier import TemporalGraspVoter
        voter = TemporalGraspVoter(window=5)

        # 1. Majority vote overrides single outlier frame
        labels = ["power_wrap", "power_wrap", "open", "power_wrap", "power_wrap"]
        present = [True, True, True, True, True]
        out = voter.smooth(labels, present)
        smoothed_labels = [res[0] for res in out]
        self.assertEqual(smoothed_labels, ["power_wrap"] * 5)

        # 2. Unknown inside a hold inherits nearest confident label
        labels_unk = ["power_wrap", "unknown", "unknown", "power_wrap"]
        present_unk = [True, True, True, True]
        out_unk = voter.smooth(labels_unk, present_unk)
        for label, inherited in out_unk:
            self.assertEqual(label, "power_wrap")

        # 3. Unpresent hand returns None
        out_unpresent = voter.smooth(["power_wrap"], [False])
        self.assertEqual(out_unpresent[0], (None, False))


if __name__ == "__main__":
    unittest.main()
