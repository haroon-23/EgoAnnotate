"""Unit tests for MetricCalibrator (Change B)."""
import unittest
import numpy as np

from src.datatypes import HandLandmarks
from src.retargeting.metric_calibration import MetricCalibrator, MetricCalibrationConfig


class TestMetricCalibration(unittest.TestCase):
    """Test suite for MetricCalibrator."""

    def setUp(self):
        self.config = MetricCalibrationConfig(enabled=True)
        self.calibrator = MetricCalibrator(self.config)

    def test_anthropometric_hand_reference_scaling(self):
        """Verify scale factor calculation from wrist-to-middle-MCP distance."""
        # Create a synthetic hand where wrist (0) to middle MCP (9) distance is 0.1 normalized units
        x = np.zeros(21)
        y = np.zeros(21)
        z = np.zeros(21)
        y[9] = 0.10  # distance = 0.10 units

        hand = HandLandmarks(
            x=x, y=y, z=z, confidence=0.9, handedness="Right", is_interpolated=False
        )

        wrists = [np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.5, 0.5])]
        result = self.calibrator.calibrate([hand, hand], wrists)

        # Scale factor should be 0.090 m / 0.10 units = 0.90 m/unit
        self.assertAlmostEqual(result.scale_factor_m_per_unit, 0.90, places=4)
        self.assertEqual(result.calibration_method, "anthropometric_hand_reference")
        self.assertIn("+/-15%", result.error_margin_str)

    def test_camera_intrinsics_override(self):
        """Verify camera intrinsics path configuration."""
        cfg = MetricCalibrationConfig(
            enabled=True,
            camera_intrinsics={"fx": 1000.0, "fy": 1000.0, "cx": 640.0, "cy": 360.0},
        )
        calibrator = MetricCalibrator(cfg)

        x = np.zeros(21)
        y = np.zeros(21)
        z = np.zeros(21)
        y[9] = 0.09

        hand = HandLandmarks(
            x=x, y=y, z=z, confidence=0.9, handedness="Right", is_interpolated=False
        )
        result = calibrator.calibrate([hand], [np.array([0.0, 0.0, 0.0])])
        self.assertEqual(result.calibration_method, "camera_intrinsics")

    def test_metadata_audit_trail(self):
        """Verify scaling metadata contains mandatory caveats and error range."""
        x = np.zeros(21)
        y = np.zeros(21)
        z = np.zeros(21)
        y[9] = 0.09
        hand = HandLandmarks(x=x, y=y, z=z, confidence=0.9, handedness="Right", is_interpolated=False)

        result = self.calibrator.calibrate([hand], [np.array([0.0, 0.0, 0.0])])
        meta = result.metadata

        self.assertIn("anthropometric_source", meta)
        self.assertIn("documented_error_margin", meta)
        self.assertIn("caveat", meta)
        self.assertIn("+/-15%", meta["caveat"])


if __name__ == "__main__":
    unittest.main()
