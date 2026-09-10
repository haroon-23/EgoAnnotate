"""Unit tests for TrajectorySmoother (Change A)."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R

from src.retargeting.pose_mapper import TargetPose
from src.retargeting.trajectory_smoother import TrajectorySmoother, TrajectorySmootherConfig


class TestTrajectorySmoothing(unittest.TestCase):
    """Test suite for TrajectorySmoother."""

    def setUp(self):
        self.config = TrajectorySmootherConfig(
            enabled=True,
            window_length=9,
            polyorder=2,
            respect_gaps=True,
        )
        self.smoother = TrajectorySmoother(self.config)

    def test_variance_reduction_on_noisy_signal(self):
        """Verify smoothing reduces frame-to-frame delta variance on a noisy sine wave."""
        np.random.seed(42)
        n_frames = 60
        t = np.linspace(0, 2 * np.pi, n_frames)
        clean_x = np.sin(t)
        clean_y = np.cos(t)
        clean_z = t * 0.1

        # Add Gaussian jitter noise
        noise_x = clean_x + np.random.normal(0, 0.05, n_frames)
        noise_y = clean_y + np.random.normal(0, 0.05, n_frames)
        noise_z = clean_z + np.random.normal(0, 0.05, n_frames)

        poses = []
        for i in range(n_frames):
            poses.append(TargetPose(
                frame_idx=i,
                timestamp=i * 0.033,
                position=np.array([noise_x[i], noise_y[i], noise_z[i]]),
                quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
                hand_detected=True,
                hand_used="right",
                is_interpolated=False,
                scaling_metadata={},
            ))

        # Compute raw frame-to-frame jitter (first differences)
        raw_pos = np.array([p.position for p in poses])
        raw_deltas = np.linalg.norm(np.diff(raw_pos, axis=0), axis=1)
        raw_variance = float(np.var(raw_deltas))

        # Apply trajectory smoothing
        smoothed_poses = self.smoother.smooth_poses(poses)
        smoothed_pos = np.array([p.position for p in smoothed_poses])
        smoothed_deltas = np.linalg.norm(np.diff(smoothed_pos, axis=0), axis=1)
        smoothed_variance = float(np.var(smoothed_deltas))

        # Verify variance of frame-to-frame deltas is reduced significantly
        self.assertLess(smoothed_variance, raw_variance)

        # Verify trajectory mean is preserved (no massive systematic offset)
        mean_offset = np.linalg.norm(np.mean(smoothed_pos, axis=0) - np.mean(raw_pos, axis=0))
        self.assertLess(mean_offset, 0.02)

    def test_quaternion_orientation_smoothing_validity(self):
        """Verify orientation smoothing produces valid unit quaternions."""
        n_frames = 30
        t = np.linspace(0, 1.0, n_frames)

        poses = []
        for i in range(n_frames):
            # Slowly rotating frame with small angular noise
            angle = t[i] * np.pi / 4 + np.random.normal(0, 0.02)
            rot = R.from_euler('z', angle)
            q = rot.as_quat()  # [x, y, z, w]

            poses.append(TargetPose(
                frame_idx=i,
                timestamp=i * 0.033,
                position=np.array([0.4, 0.0, 0.3]),
                quaternion=q,
                hand_detected=True,
                hand_used="right",
                is_interpolated=False,
                scaling_metadata={},
            ))

        smoothed_poses = self.smoother.smooth_poses(poses)
        for p in smoothed_poses:
            norm = np.linalg.norm(p.quaternion)
            self.assertAlmostEqual(norm, 1.0, places=5)

    def test_gap_boundary_respect(self):
        """Verify smoothing does not blend across hand tracking loss gaps."""
        n_frames = 40
        poses = []
        for i in range(n_frames):
            # Frame 15 to 25 are gap-interpolated / lost
            is_gap = 15 <= i <= 25
            pos_x = 10.0 if i > 25 else 1.0
            poses.append(TargetPose(
                frame_idx=i,
                timestamp=i * 0.033,
                position=np.array([pos_x, 0.0, 0.0]),
                quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
                hand_detected=not is_gap,
                hand_used="right" if not is_gap else None,
                is_interpolated=is_gap,
                scaling_metadata={},
            ))

        smoothed_poses = self.smoother.smooth_poses(poses)
        # Check that gap pose positions remain unmodified
        for i in range(15, 26):
            self.assertEqual(smoothed_poses[i].position[0], 10.0 if i > 25 else 1.0)


if __name__ == "__main__":
    unittest.main()
