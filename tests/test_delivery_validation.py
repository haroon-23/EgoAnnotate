"""Tests for delivery validation logic in DatasetExporter.

Verifies that validate_delivery() correctly:
- Passes for valid overlays matching source video resolution
- Fails for missing overlay files
- Fails for resolution mismatches (the exact bug this fix addresses)
- Fails for zero-byte files
- Fails for missing episode_rlds.hdf5  (NEW: all-4-deliverable enforcement)
- Fails for missing side_by_side.mp4  (NEW: when retargeting data is present)
- Fails for missing lerobot_v3/       (NEW: all-4-deliverable enforcement)
"""
import os
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset_exporter import DatasetExporter, ExporterConfig


def _create_test_video(path: Path, width: int, height: int, num_frames: int = 10, fps: float = 30.0) -> None:
    """Create a minimal test video at given resolution."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Add some content so it's not blank
        cv2.putText(frame, f"Frame {i}", (10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(frame)
    writer.release()


class TestDeliveryValidation(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        self.config = ExporterConfig(output_dir=str(self.output_dir), save_overlay_video=False)
        self.exporter = DatasetExporter(self.config)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_validate_delivery_passes_matching_resolution(self):
        """Overlay with same resolution as source should pass validation."""
        episode_dir = self.output_dir / "ep_valid"
        episode_dir.mkdir(parents=True, exist_ok=True)

        source_path = self.output_dir / "source.mp4"
        _create_test_video(source_path, 640, 480, num_frames=10)
        _create_test_video(episode_dir / "overlay_annotated.mp4", 640, 480, num_frames=10)
        self._make_valid_rlds(episode_dir)
        self._make_valid_lerobot(episode_dir)

        # Should not raise
        self.exporter.validate_delivery(episode_dir, str(source_path))

    def test_validate_delivery_fails_resolution_mismatch(self):
        """Overlay at 224x224 for a 640x480 source should fail — this is the exact bug."""
        episode_dir = self.output_dir / "ep_mismatch"
        episode_dir.mkdir(parents=True, exist_ok=True)

        source_path = self.output_dir / "source_hd.mp4"
        _create_test_video(source_path, 640, 480, num_frames=10)
        _create_test_video(episode_dir / "overlay_annotated.mp4", 224, 224, num_frames=10)
        self._make_valid_rlds(episode_dir)
        self._make_valid_lerobot(episode_dir)

        with self.assertRaises(RuntimeError) as ctx:
            self.exporter.validate_delivery(episode_dir, str(source_path))
        self.assertIn("RESOLUTION MISMATCH", str(ctx.exception))
        self.assertIn("224x224", str(ctx.exception))
        self.assertIn("640x480", str(ctx.exception))

    def test_validate_delivery_fails_missing_overlay(self):
        """Missing overlay file should fail validation."""
        episode_dir = self.output_dir / "ep_missing"
        episode_dir.mkdir(parents=True, exist_ok=True)
        # No overlay, no RLDS, no LeRobot — all missing, error must mention overlay
        with self.assertRaises(RuntimeError) as ctx:
            self.exporter.validate_delivery(episode_dir, "dummy.mp4")
        self.assertIn("overlay_annotated.mp4", str(ctx.exception))
        self.assertIn("MISSING", str(ctx.exception))

    def test_validate_delivery_fails_zero_bytes(self):
        """Zero-byte overlay should fail validation."""
        episode_dir = self.output_dir / "ep_zero"
        episode_dir.mkdir(parents=True, exist_ok=True)
        (episode_dir / "overlay_annotated.mp4").touch()

        with self.assertRaises(RuntimeError) as ctx:
            self.exporter.validate_delivery(episode_dir, "dummy.mp4")
        self.assertIn("ZERO BYTES", str(ctx.exception))

    def test_validate_delivery_graceful_when_source_missing(self):
        """If source video can't be probed, validation logs warning but doesn't crash."""
        episode_dir = self.output_dir / "ep_no_source"
        episode_dir.mkdir(parents=True, exist_ok=True)

        overlay_path = episode_dir / "overlay_annotated.mp4"
        _create_test_video(overlay_path, 640, 480, num_frames=5)

        # Also need HDF5 and lerobot_v3 for full validation to pass
        import h5py, json, pandas as pd
        rlds_path = episode_dir / "episode_rlds.hdf5"
        with h5py.File(rlds_path, "w") as hf:
            ep = hf.create_group("ep_no_source")
            steps = ep.create_group("steps")
            obs = steps.create_group("observation")
            steps.create_dataset("action", data=np.zeros((5, 24), dtype=np.float32))
            utf8 = h5py.string_dtype(encoding="utf-8")
            ds = obs.create_dataset("language_instruction", (5,), dtype=utf8)
            for i in range(5):
                ds[i] = "test"

        lr_dir = episode_dir / "lerobot_v3" / "meta"
        lr_dir.mkdir(parents=True)
        with open(lr_dir / "info.json", "w") as f:
            json.dump({"codebase_version": "v3.0"}, f)
        data_dir = episode_dir / "lerobot_v3" / "data" / "chunk-000"
        data_dir.mkdir(parents=True)
        pd.DataFrame({"frame_index": [0]}).to_parquet(data_dir / "file-000.parquet", index=False)

        # Should not raise — source probe fails gracefully
        self.exporter.validate_delivery(episode_dir, "/nonexistent/video.mp4")

    def test_ffprobe_returns_correct_resolution(self):
        """Verify _ffprobe_video_info returns accurate dimensions."""
        test_video = self.output_dir / "probe_test.mp4"
        _create_test_video(test_video, 1920, 1080, num_frames=5)

        info = self.exporter._ffprobe_video_info(str(test_video))
        self.assertIsNotNone(info)
        self.assertEqual(info["width"], 1920)
        self.assertEqual(info["height"], 1080)

    def test_ffprobe_returns_none_for_nonexistent(self):
        """_ffprobe_video_info returns None for missing files."""
        info = self.exporter._ffprobe_video_info("/does/not/exist.mp4")
        self.assertIsNone(info)

    # -------------------------------------------------------------------
    # NEW: Multi-artifact fail-loud tests
    # -------------------------------------------------------------------

    def _make_valid_overlay(self, episode_dir: Path, source_path: Path) -> None:
        """Helper: create matching overlay and source videos."""
        _create_test_video(source_path, 640, 480, num_frames=10)
        _create_test_video(episode_dir / "overlay_annotated.mp4", 640, 480, num_frames=10)

    def _make_valid_rlds(self, episode_dir: Path) -> None:
        """Helper: create a minimal valid episode_rlds.hdf5."""
        import h5py
        ep_id = episode_dir.name
        rlds_path = episode_dir / "episode_rlds.hdf5"
        with h5py.File(rlds_path, "w") as hf:
            ep = hf.create_group(ep_id)
            steps = ep.create_group("steps")
            obs = steps.create_group("observation")
            steps.create_dataset("action", data=np.zeros((5, 24), dtype=np.float32))
            utf8 = h5py.string_dtype(encoding="utf-8")
            ds = obs.create_dataset("language_instruction", (5,), dtype=utf8)
            for i in range(5):
                ds[i] = "test instruction"

    def _make_valid_lerobot(self, episode_dir: Path) -> None:
        """Helper: create a minimal valid lerobot_v3/ structure."""
        import json, pandas as pd
        lr_meta = episode_dir / "lerobot_v3" / "meta"
        lr_meta.mkdir(parents=True)
        with open(lr_meta / "info.json", "w") as f:
            json.dump({"codebase_version": "v3.0", "total_frames": 5}, f)
        data_dir = episode_dir / "lerobot_v3" / "data" / "chunk-000"
        data_dir.mkdir(parents=True)
        pd.DataFrame({"frame_index": list(range(5))}).to_parquet(
            data_dir / "file-000.parquet", index=False
        )

    def test_validate_delivery_fails_missing_rlds(self):
        """Missing episode_rlds.hdf5 must raise RuntimeError."""
        episode_dir = self.output_dir / "ep_no_rlds"
        episode_dir.mkdir(parents=True)
        source_path = self.output_dir / "source_rlds.mp4"
        self._make_valid_overlay(episode_dir, source_path)
        self._make_valid_lerobot(episode_dir)
        # No RLDS file created deliberately

        with self.assertRaises(RuntimeError) as ctx:
            self.exporter.validate_delivery(episode_dir, str(source_path))
        self.assertIn("episode_rlds.hdf5", str(ctx.exception))
        self.assertIn("MISSING", str(ctx.exception))

    def test_validate_delivery_fails_missing_side_by_side(self):
        """Missing side_by_side.mp4 must raise RuntimeError when retargeting ran."""
        episode_dir = self.output_dir / "ep_no_sbs"
        episode_dir.mkdir(parents=True)
        source_path = self.output_dir / "source_sbs.mp4"
        self._make_valid_overlay(episode_dir, source_path)
        self._make_valid_rlds(episode_dir)
        self._make_valid_lerobot(episode_dir)
        # side_by_side.mp4 not created — pass its path explicitly to signal retargeting ran
        missing_sbs = episode_dir / "side_by_side.mp4"

        with self.assertRaises(RuntimeError) as ctx:
            self.exporter.validate_delivery(episode_dir, str(source_path), side_by_side_path=missing_sbs)
        self.assertIn("side_by_side.mp4", str(ctx.exception))
        self.assertIn("MISSING", str(ctx.exception))

    def test_validate_delivery_fails_missing_lerobot(self):
        """Missing lerobot_v3/ directory must raise RuntimeError."""
        episode_dir = self.output_dir / "ep_no_lr"
        episode_dir.mkdir(parents=True)
        source_path = self.output_dir / "source_lr.mp4"
        self._make_valid_overlay(episode_dir, source_path)
        self._make_valid_rlds(episode_dir)
        # No lerobot_v3/ created deliberately

        with self.assertRaises(RuntimeError) as ctx:
            self.exporter.validate_delivery(episode_dir, str(source_path))
        self.assertIn("lerobot_v3", str(ctx.exception))
        self.assertIn("MISSING", str(ctx.exception))

    def test_validate_delivery_all_four_pass(self):
        """All 4 deliverables present and valid — should not raise."""
        episode_dir = self.output_dir / "ep_all_ok"
        episode_dir.mkdir(parents=True)
        source_path = self.output_dir / "source_all_ok.mp4"
        self._make_valid_overlay(episode_dir, source_path)
        self._make_valid_rlds(episode_dir)
        self._make_valid_lerobot(episode_dir)
        # No side_by_side required (no retargeting data)
        self.exporter.validate_delivery(episode_dir, str(source_path))


if __name__ == "__main__":
    unittest.main()
