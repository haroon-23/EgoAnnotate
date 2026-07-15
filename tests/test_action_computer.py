import os
import unittest
import numpy as np
import sys

# Add parent directory to path to allow import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datatypes import AnnotationFrame, HandLandmarks, RobotAgnosticAction
from src.action_computer import ActionComputer, ActionComputerConfig


class TestActionComputer(unittest.TestCase):

    def test_config_initialization(self):
        """Test default ActionComputerConfig values."""
        config = ActionComputerConfig()
        self.assertTrue(config.compute_wrist_delta)
        self.assertTrue(config.compute_finger_angles)
        self.assertTrue(config.compute_gripper_state)
        self.assertTrue(config.normalize_actions)
        self.assertEqual(config.percentile_low, 1.0)
        self.assertEqual(config.percentile_high, 99.0)

    def test_empty_frames(self):
        """Test that empty input returns empty list."""
        computer = ActionComputer(ActionComputerConfig())
        self.assertEqual(computer.compute_actions([]), [])

    def test_wrist_delta_and_dominant_hand(self):
        """Test dominant hand selection and wrist delta computation across sequential frames."""
        config = ActionComputerConfig(normalize_actions=False)
        computer = ActionComputer(config)
        
        # Frame 0: Right hand at (0.5, 0.5, 0.1)
        x0 = np.ones(21) * 0.5
        y0 = np.ones(21) * 0.5
        z0 = np.ones(21) * 0.1
        # Set wrist (0) specifically
        x0[0], y0[0], z0[0] = 0.5, 0.5, 0.1
        
        hand0 = HandLandmarks(x=x0, y=y0, z=z0, confidence=0.9, handedness="Right")
        frame0 = AnnotationFrame(frame_idx=0, timestamp=0.0, image_path="f0.png", right_hand=hand0)
        
        # Frame 1: Right hand moved wrist to (0.52, 0.49, 0.11)
        x1 = np.ones(21) * 0.5
        y1 = np.ones(21) * 0.5
        z1 = np.ones(21) * 0.1
        x1[0], y1[0], z1[0] = 0.52, 0.49, 0.11
        
        hand1 = HandLandmarks(x=x1, y=y1, z=z1, confidence=0.9, handedness="Right")
        frame1 = AnnotationFrame(frame_idx=1, timestamp=0.1, image_path="f1.png", right_hand=hand1)
        
        frames = [frame0, frame1]
        updated_frames = computer.compute_actions(frames)
        
        # Frame 0 action: wrist_delta is zeros since prev_frame is None
        action0 = updated_frames[0].action
        self.assertIsNotNone(action0)
        np.testing.assert_array_almost_equal(action0.wrist_delta, [0.0, 0.0, 0.0])
        
        # Frame 1 action: wrist_delta should be (p1 - p0) * 10
        # p0 = [0.5, 0.5, 0.1], p1 = [0.52, 0.49, 0.11]
        # delta = [0.02, -0.01, 0.01] * 10 = [0.2, -0.1, 0.1]
        action1 = updated_frames[1].action
        self.assertIsNotNone(action1)
        np.testing.assert_array_almost_equal(action1.wrist_delta, [0.2, -0.1, 0.1])

    def test_finger_angles_length_and_range(self):
        """Test finger angles vector length is 15 and values are within range [-1, 1]."""
        config = ActionComputerConfig(normalize_actions=False)
        computer = ActionComputer(config)
        
        # Flat/open hand landmarks
        x = np.zeros(21)
        y = np.zeros(21)
        z = np.zeros(21)
        
        # Base/MCP/PIP/DIP/Tip for Index
        # Wrist at (0,0)
        # MCP (5) at (0, 0.1)
        # PIP (6) at (0, 0.2)
        # DIP (7) at (0, 0.3)
        # Tip (8) at (0, 0.4) -> perfectly straight index finger
        x[0], y[0] = 0.0, 0.0
        x[5], y[5] = 0.0, 0.1
        x[6], y[6] = 0.0, 0.2
        x[7], y[7] = 0.0, 0.3
        x[8], y[8] = 0.0, 0.4
        
        hand = HandLandmarks(x=x, y=y, z=z, confidence=0.9, handedness="Left")
        frame = AnnotationFrame(frame_idx=0, timestamp=0.0, image_path="f0.png", left_hand=hand)
        
        updated = computer.compute_actions([frame])
        action = updated[0].action
        self.assertIsNotNone(action)
        self.assertEqual(len(action.finger_angles), 15)
        
        # Check index finger angles (index: index 3 to 5 in finger_angles)
        # Straight should be close to 1.0
        # Vector v1 = (5->6) = [0, 0.1], v2 = (6->7) = [0, 0.1] -> theta = 0 -> angle = 1.0
        # Joint 1 angle (CMC/Wrist-MCP for index) is in index 3
        # Joint 2 angle (MCP-PIP for index) is in index 4
        # Joint 3 angle (PIP-DIP for index) is in index 5
        self.assertAlmostEqual(action.finger_angles[3], 1.0, places=4)
        self.assertAlmostEqual(action.finger_angles[4], 1.0, places=4)
        self.assertAlmostEqual(action.finger_angles[5], 1.0, places=4)
        
        # Verify all values in action.finger_angles are clipped in [-1, 1]
        for val in action.finger_angles:
            self.assertTrue(-1.0 <= val <= 1.0)

    def test_gripper_openness_and_hand_orientation(self):
        """Test gripper openness calculation and unit normal palm orientation 4-vector."""
        config = ActionComputerConfig(normalize_actions=False)
        computer = ActionComputer(config)
        
        # Build open hand pose
        x = np.zeros(21)
        y = np.zeros(21)
        z = np.zeros(21)
        
        # Wrist at (0, 0, 0)
        # Index MCP (5) at (0.1, 0.0, 0.0)
        # Pinky MCP (17) at (0.0, 0.1, 0.0)
        # Palm plane normal vector should be along Z axis [0, 0, 1] (or -1)
        x[5], y[5], z[5] = 0.1, 0.0, 0.0
        x[17], y[17], z[17] = 0.0, 0.1, 0.0
        
        # Thumb tip (4) and Pinky tip (20) far apart, index tip (8) also far
        # Let's set thumb tip at (0.2, 0.0, 0.0)
        # pinky tip at (0.0, 0.2, 0.0)
        # index tip at (0.2, 0.2, 0.0)
        x[4], y[4], z[4] = 0.2, 0.0, 0.0
        x[20], y[20], z[20] = 0.0, 0.2, 0.0
        x[8], y[8], z[8] = 0.2, 0.2, 0.0
        
        hand = HandLandmarks(x=x, y=y, z=z, confidence=0.9, handedness="Right")
        frame = AnnotationFrame(frame_idx=0, timestamp=0.0, image_path="f0.png", right_hand=hand)
        
        updated = computer.compute_actions([frame])
        action = updated[0].action
        self.assertIsNotNone(action)
        
        # Gripper openness check
        self.assertTrue(0.0 <= action.gripper_openness <= 1.0)
        
        # Palm orientation check: shape should be (4,) representing [nx, ny, nz, d]
        self.assertEqual(action.hand_orientation.shape, (4,))
        # Wrist is at (0,0,0) so offset d should be 0.0 (d = -normal . wrist)
        self.assertAlmostEqual(action.hand_orientation[3], 0.0, places=4)
        
        # The normal vector part should be unit length
        normal_part = action.hand_orientation[:3]
        np.testing.assert_allclose(np.linalg.norm(normal_part), 1.0)

    def test_action_normalization(self):
        """Test that wrist deltas are successfully normalized to [-1, 1] using percentile clipping."""
        config = ActionComputerConfig(normalize_actions=True)
        computer = ActionComputer(config)
        
        # Create a series of frames with a moving right hand wrist
        frames = []
        for i in range(10):
            x = np.ones(21) * 0.5
            y = np.ones(21) * 0.5
            z = np.ones(21) * 0.1
            # Wrist moves from 0.0 to 0.09
            x[0] = 0.5 + 0.01 * i
            
            hand = HandLandmarks(x=x, y=y, z=z, confidence=0.9, handedness="Right")
            frame = AnnotationFrame(frame_idx=i, timestamp=0.1 * i, image_path=f"f{i}.png", right_hand=hand)
            frames.append(frame)
            
        updated_frames = computer.compute_actions(frames)
        
        # All normalized deltas should be inside [-1, 1]
        for f in updated_frames:
            action = f.action
            if action is not None:
                for val in action.wrist_delta:
                    self.assertTrue(-1.0001 <= val <= 1.0001)


if __name__ == "__main__":
    unittest.main()
