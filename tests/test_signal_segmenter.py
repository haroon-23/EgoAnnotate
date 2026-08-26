"""Tests for SignalSegmenter."""

import os
import sys
import unittest
import numpy as np

# Add parent directory to path to allow import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datatypes import ContactState, GraspType
from src.signal_segmenter import SignalSegmenter, SignalSegmenterConfig


def make_contact(in_contact: bool, object_name: str = "mug") -> ContactState:
    """Create a ContactState for testing."""
    return ContactState(
        fingers=np.array([in_contact] * 5, dtype=bool),
        object_name=object_name if in_contact else None,
        in_contact=in_contact,
        confidence=0.9 if in_contact else 0.0,
    )


def make_grasp(grasp_type: str) -> GraspType:
    """Create a GraspType for testing."""
    return GraspType(
        type=grasp_type,
        confidence=0.8,
        thumb_index_distance=0.05,
        num_curled_fingers=2,
    )


class TestSignalSegmenter(unittest.TestCase):
    
    def setUp(self):
        """Create a standard 30 FPS timeline."""
        self.fps = 30
        self.n_frames = 120  # 4 seconds
        self.timestamps = [i / self.fps for i in range(self.n_frames)]
        
        self.config = SignalSegmenterConfig(
            min_segment_duration_sec=0.2,
            merge_gap_sec=0.3,
            idle_threshold_sec=1.0,
        )
        self.segmenter = SignalSegmenter(self.config)
    
    def test_synthetic_timeline(self):
        """Test the synthetic hand timeline from the task description.
        
        frames 0-30:   no contact (1.0 sec)
        frames 30-60:  contact, precision_pinch (1.0 sec)
        frames 60-90:  contact, power_wrap (1.0 sec)
        frames 90-120: no contact (1.0 sec)
        """
        # Build timelines
        left_contact = [None] * self.n_frames
        right_contact = [None] * self.n_frames
        left_grasp = [None] * self.n_frames
        right_grasp = [None] * self.n_frames
        
        # frames 0-30: no contact
        for i in range(30):
            right_contact[i] = make_contact(False)
            right_grasp[i] = make_grasp("none")
        
        # frames 30-60: contact, precision_pinch
        for i in range(30, 60):
            right_contact[i] = make_contact(True, "mug")
            right_grasp[i] = make_grasp("precision_pinch")
        
        # frames 60-90: contact, power_wrap
        for i in range(60, 90):
            right_contact[i] = make_contact(True, "mug")
            right_grasp[i] = make_grasp("power_wrap")
        
        # frames 90-120: no contact
        for i in range(90, 120):
            right_contact[i] = make_contact(False)
            right_grasp[i] = make_grasp("none")
        
        candidates = self.segmenter.get_candidates(
            left_contact, right_contact,
            left_grasp, right_grasp,
            self.timestamps
        )
        
        # Should have 4 segments
        self.assertEqual(len(candidates), 4)
        
        # Segment 0: idle (0-1.0 sec)
        self.assertAlmostEqual(candidates[0].start_time, 0.0, places=1)
        self.assertAlmostEqual(candidates[0].end_time, 1.0, places=1)
        self.assertEqual(candidates[0].contact_state, "no_contact")
        self.assertEqual(candidates[0].grasp_type, "none")
        
        # Segment 1: contact + precision_pinch (1.0-2.0 sec)
        self.assertAlmostEqual(candidates[1].start_time, 1.0, places=1)
        self.assertAlmostEqual(candidates[1].end_time, 2.0, places=1)
        self.assertEqual(candidates[1].contact_state, "contact")
        self.assertEqual(candidates[1].grasp_type, "precision_pinch")
        self.assertEqual(candidates[1].object_name, "mug")
        
        # Segment 2: contact + power_wrap (2.0-3.0 sec)
        self.assertAlmostEqual(candidates[2].start_time, 2.0, places=1)
        self.assertAlmostEqual(candidates[2].end_time, 3.0, places=1)
        self.assertEqual(candidates[2].contact_state, "contact")
        self.assertEqual(candidates[2].grasp_type, "power_wrap")
        self.assertEqual(candidates[2].object_name, "mug")
        
        # Segment 3: idle (3.0-4.0 sec)
        self.assertAlmostEqual(candidates[3].start_time, 3.0, places=1)
        self.assertAlmostEqual(candidates[3].end_time, 4.0, places=1)
        self.assertEqual(candidates[3].contact_state, "no_contact")
        self.assertEqual(candidates[3].grasp_type, "none")
    
    def test_jitter_filtering(self):
        """Test that short contact flickers are merged."""
        # Contact on for 3 frames (0.1 sec) then off - should be merged
        left_contact = [None] * self.n_frames
        right_contact = [None] * self.n_frames
        left_grasp = [None] * self.n_frames
        right_grasp = [None] * self.n_frames
        
        for i in range(self.n_frames):
            if 30 <= i < 33:  # 3 frames = 0.1 sec
                right_contact[i] = make_contact(True, "mug")
                right_grasp[i] = make_grasp("precision_pinch")
            else:
                right_contact[i] = make_contact(False)
                right_grasp[i] = make_grasp("none")
        
        candidates = self.segmenter.get_candidates(
            left_contact, right_contact,
            left_grasp, right_grasp,
            self.timestamps
        )
        
        # The 3-frame contact should be merged into adjacent idle
        # Result should be 1 segment (or the contact absorbed)
        total_duration = sum(c.end_time - c.start_time for c in candidates)
        self.assertAlmostEqual(total_duration, 4.0, places=1)
    
    def test_grasp_change_creates_boundary(self):
        """Test that grasp type change during contact creates a segment boundary."""
        left_contact = [None] * self.n_frames
        right_contact = [None] * self.n_frames
        left_grasp = [None] * self.n_frames
        right_grasp = [None] * self.n_frames
        
        # Continuous contact but grasp changes
        for i in range(self.n_frames):
            right_contact[i] = make_contact(True, "mug")
            if i < 60:
                right_grasp[i] = make_grasp("precision_pinch")
            else:
                right_grasp[i] = make_grasp("power_wrap")
        
        candidates = self.segmenter.get_candidates(
            left_contact, right_contact,
            left_grasp, right_grasp,
            self.timestamps
        )
        
        # Should have 2 segments: precision_pinch and power_wrap
        contact_segments = [c for c in candidates if c.contact_state == "contact"]
        self.assertEqual(len(contact_segments), 2)
        
        self.assertEqual(contact_segments[0].grasp_type, "precision_pinch")
        self.assertEqual(contact_segments[1].grasp_type, "power_wrap")
    
    def test_idle_boundary(self):
        """Test that sustained idle creates boundaries."""
        left_contact = [None] * self.n_frames
        right_contact = [None] * self.n_frames
        left_grasp = [None] * self.n_frames
        right_grasp = [None] * self.n_frames
        
        # Contact for 1 sec, idle for 1.5 sec, contact for 1.5 sec
        for i in range(self.n_frames):
            right_grasp[i] = make_grasp("none")
            if i < 30:  # 1 sec contact
                right_contact[i] = make_contact(True, "mug")
                right_grasp[i] = make_grasp("power_wrap")
            elif i < 75:  # 1.5 sec idle (> 1.0 threshold)
                right_contact[i] = make_contact(False)
            else:  # 1.5 sec contact
                right_contact[i] = make_contact(True, "cup")
                right_grasp[i] = make_grasp("precision_pinch")
        
        candidates = self.segmenter.get_candidates(
            left_contact, right_contact,
            left_grasp, right_grasp,
            self.timestamps
        )
        
        # Should have 3 segments: contact, idle, contact
        self.assertEqual(len(candidates), 3)
        self.assertEqual(candidates[0].contact_state, "contact")
        self.assertEqual(candidates[1].contact_state, "no_contact")
        self.assertEqual(candidates[2].contact_state, "contact")
        
        # Idle segment should be >= 1.0 sec
        idle_seg = candidates[1]
        self.assertGreaterEqual(idle_seg.end_time - idle_seg.start_time, 1.0)
    
    def test_empty_timeline(self):
        """Test empty input returns empty list."""
        candidates = self.segmenter.get_candidates(
            [], [], [], [], []
        )
        self.assertEqual(candidates, [])
    
    def test_single_frame(self):
        """Test single frame returns one segment."""
        candidates = self.segmenter.get_candidates(
            [None], [make_contact(False)],
            [None], [make_grasp("none")],
            [0.0]
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].start_frame, 0)
        self.assertEqual(candidates[0].end_frame, 1)
    
    def test_min_segment_duration(self):
        """Test min_segment_duration_sec filters correctly."""
        # 5 frames contact (0.167 sec) < 0.2 min
        left_contact = [None] * 50
        right_contact = [None] * 50
        left_grasp = [None] * 50
        right_grasp = [None] * 50
        timestamps = [i / 30 for i in range(50)]
        
        for i in range(50):
            right_grasp[i] = make_grasp("none")
            if 15 <= i < 20:  # 5 frames
                right_contact[i] = make_contact(True, "mug")
                right_grasp[i] = make_grasp("precision_pinch")
            else:
                right_contact[i] = make_contact(False)
        
        candidates = self.segmenter.get_candidates(
            left_contact, right_contact,
            left_grasp, right_grasp,
            timestamps
        )
        
        # Short contact should be merged, resulting in 1 segment
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].contact_state, "no_contact")


if __name__ == "__main__":
    unittest.main()