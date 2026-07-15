import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add parent directory to path to allow import from scripts and src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.run import main


class TestRunCLI(unittest.TestCase):

    @patch("sys.argv", ["scripts/run.py"])
    def test_run_no_args_exits(self):
        """Test that running with no arguments prints an error and exits with status code 1."""
        with self.assertRaises(SystemExit) as cm:
            with patch("sys.stderr.write") as mock_stderr:
                main()
        self.assertEqual(cm.exception.code, 1)

    @patch("glob.glob")
    @patch("scripts.run.EgoAnnotatePipeline")
    def test_run_with_video_args_success(self, mock_pipeline_cls, mock_glob):
        """Test that running with a valid video path invokes the pipeline process_videos method."""
        # Set up mock pipeline instance
        mock_pipeline_inst = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline_inst
        mock_pipeline_inst.process_videos.return_value = ["mock_episode"]
        mock_pipeline_inst.dataset_exporter.config.output_dir = "mock_out"

        # Mock glob.glob to return nothing (direct paths tested)
        mock_glob.return_value = []

        # Mock os.path.exists to say our video exists
        with patch("os.path.exists", return_value=True):
            # Mock sys.argv to specify video
            with patch("sys.argv", ["scripts/run.py", "--video", "data/test.mp4", "--config", "configs/default.yaml"]):
                # Run main()
                with patch("builtins.print") as mock_print:
                    main()

            # Verify that EgoAnnotatePipeline was instantiated with correct config
            mock_pipeline_cls.assert_called_once_with(config_path="configs/default.yaml")

            # Verify process_videos was called with our video path
            mock_pipeline_inst.process_videos.assert_called_once_with(["data/test.mp4"])


if __name__ == "__main__":
    unittest.main()
