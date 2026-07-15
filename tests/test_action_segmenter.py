import os
import unittest
from unittest.mock import MagicMock, patch
import sys

# Add parent directory to path to allow import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datatypes import ActionSegment
from src.action_segmenter import GeminiActionSegmenter, ActionSegmenterConfig


class TestGeminiActionSegmenter(unittest.TestCase):

    def test_config_initialization(self):
        """Test default ActionSegmenterConfig values."""
        config = ActionSegmenterConfig()
        self.assertEqual(config.gemini_model, "gemini-1.5-pro-latest")
        self.assertIn("temporal segments", config.prompt)

    def test_api_key_missing_raises_error(self):
        """Test that ValueError is raised if GEMINI_API_KEY environment variable is missing."""
        old_key = os.environ.pop("GEMINI_API_KEY", None)
        try:
            config = ActionSegmenterConfig()
            with self.assertRaises(ValueError) as context:
                GeminiActionSegmenter(config)
            self.assertIn("GEMINI_API_KEY environment variable is not set", str(context.exception))
        finally:
            if old_key is not None:
                os.environ["GEMINI_API_KEY"] = old_key

    @patch("google.generativeai.configure")
    @patch("google.generativeai.GenerativeModel")
    def test_time_parsing_formats(self, mock_model, mock_configure):
        """Test parsing of different time string formats."""
        os.environ["GEMINI_API_KEY"] = "mock-api-key-value"
        segmenter = GeminiActionSegmenter(ActionSegmenterConfig())
        
        # Test "MM:SS.ms"
        self.assertEqual(segmenter._parse_time("01:30.50"), 90.50)
        # Test "MM:SS"
        self.assertEqual(segmenter._parse_time("02:15"), 135.0)
        # Test "HH:MM:SS"
        self.assertEqual(segmenter._parse_time("01:00:10"), 3610.0)
        # Test plain seconds float/int
        self.assertEqual(segmenter._parse_time("4.25"), 4.25)
        self.assertEqual(segmenter._parse_time("10"), 10.0)
        # Test malformed / empty strings
        self.assertEqual(segmenter._parse_time(""), 0.0)
        self.assertEqual(segmenter._parse_time("invalid"), 0.0)

    @patch("google.generativeai.configure")
    @patch("google.generativeai.GenerativeModel")
    def test_parse_json_response(self, mock_model, mock_configure):
        """Test parsing of clean JSON response and markdown blocks."""
        os.environ["GEMINI_API_KEY"] = "mock-api-key-value"
        segmenter = GeminiActionSegmenter(ActionSegmenterConfig())
        
        json_input = """
        ```json
        [
          {
            "name": "pick_up",
            "start_time": "0:02.5",
            "end_time": "0:08.0",
            "object_name": "spoon",
            "hand_used": "left",
            "description": "picking up the silver spoon"
          }
        ]
        ```
        """
        results = segmenter._parse_json_response(json_input)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "pick_up")
        self.assertEqual(results[0].start_time, 2.5)
        self.assertEqual(results[0].end_time, 8.0)
        self.assertEqual(results[0].object_name, "spoon")
        self.assertEqual(results[0].hand_used, "left")
        self.assertEqual(results[0].description, "picking up the silver spoon")

    @patch("google.generativeai.configure")
    @patch("google.generativeai.GenerativeModel")
    def test_parse_text_fallback(self, mock_model, mock_configure):
        """Test regex fallback parser on plain text descriptions."""
        os.environ["GEMINI_API_KEY"] = "mock-api-key-value"
        segmenter = GeminiActionSegmenter(ActionSegmenterConfig())
        
        fallback_text = """
        Below is the temporal segmentation of the video:
        00:02.5 - 00:08.0: pick_up holding the red cup with the right hand
        00:08.0 - 00:15.20: pour pouring water into the bowl using both hands
        """
        results = segmenter._parse_text_fallback(fallback_text)
        self.assertEqual(len(results), 2)
        
        # Check first segment
        self.assertEqual(results[0].name, "pick_up")
        self.assertEqual(results[0].start_time, 2.5)
        self.assertEqual(results[0].end_time, 8.0)
        self.assertEqual(results[0].object_name, "cup") # extracted via heuristic from "holding the red cup"
        self.assertEqual(results[0].hand_used, "right")
        self.assertEqual(results[0].description, "holding the red cup with the right hand")
        
        # Check second segment
        self.assertEqual(results[1].name, "pour")
        self.assertEqual(results[1].start_time, 8.0)
        self.assertAlmostEqual(results[1].end_time, 15.20, places=4)
        self.assertEqual(results[1].object_name, "bowl") # extracted via heuristic from "pouring water into the bowl"
        self.assertEqual(results[1].hand_used, "both")
        self.assertEqual(results[1].description, "pouring water into the bowl using both hands")

    @patch("google.generativeai.configure")
    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.upload_file")
    def test_default_fallback_segment_on_failure(self, mock_upload, mock_model, mock_configure):
        """Test that single default segment is returned if upload/generation fails."""
        os.environ["GEMINI_API_KEY"] = "mock-api-key-value"
        segmenter = GeminiActionSegmenter(ActionSegmenterConfig())
        
        # Make upload raise an error to trigger fallback path
        mock_upload.side_effect = Exception("Upload failed")
        
        results = segmenter.segment_video("dummy_path.mp4")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "manipulate")
        self.assertEqual(results[0].start_time, 0.0)
        self.assertEqual(results[0].end_time, 10.0)
        self.assertEqual(results[0].object_name, "unknown")
        self.assertEqual(results[0].hand_used, "right")


if __name__ == "__main__":
    unittest.main()
