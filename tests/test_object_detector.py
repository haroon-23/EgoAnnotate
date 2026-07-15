import os
import unittest
from unittest.mock import MagicMock, patch
import sys

# Add parent directory to path to allow import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datatypes import ObjectAnnotation
from src.object_detector import GeminiObjectDetector, ObjectDetectorConfig


class TestGeminiObjectDetector(unittest.TestCase):

    def test_config_initialization(self):
        """Test default config values."""
        config = ObjectDetectorConfig()
        self.assertEqual(config.keyframes_per_video, 3)
        self.assertEqual(config.gemini_model, "gemini-1.5-pro-latest")
        self.assertIn("Identify all objects", config.prompt)

    def test_api_key_missing_raises_error(self):
        """Test that ValueError is raised if GEMINI_API_KEY environment variable is missing."""
        # Ensure GEMINI_API_KEY is not in environment
        old_key = os.environ.pop("GEMINI_API_KEY", None)
        try:
            config = ObjectDetectorConfig()
            with self.assertRaises(ValueError) as context:
                GeminiObjectDetector(config)
            self.assertIn("GEMINI_API_KEY environment variable is not set", str(context.exception))
        finally:
            # Restore environment key if existed
            if old_key is not None:
                os.environ["GEMINI_API_KEY"] = old_key

    @patch("google.generativeai.configure")
    @patch("google.generativeai.GenerativeModel")
    def test_detector_initialization_success(self, mock_model, mock_configure):
        """Test detector instantiates successfully when API key is present."""
        os.environ["GEMINI_API_KEY"] = "mock-api-key-value"
        config = ObjectDetectorConfig()
        detector = GeminiObjectDetector(config)
        
        self.assertEqual(detector.config, config)
        mock_configure.assert_called_once_with(api_key="mock-api-key-value")
        mock_model.assert_called_once_with("gemini-1.5-pro-latest")

    @patch("google.generativeai.configure")
    @patch("google.generativeai.GenerativeModel")
    def test_parse_json_response(self, mock_model, mock_configure):
        """Test parsing of a clean JSON list response."""
        os.environ["GEMINI_API_KEY"] = "mock-api-key-value"
        detector = GeminiObjectDetector(ObjectDetectorConfig())
        
        json_input = """
        [
          {"name": "spatula", "location": "on the counter top", "touched": true},
          {"name": "pan", "location": "on the stove burner", "touched": false}
        ]
        """
        results = detector._parse_response(json_input)
        self.assertEqual(len(results), 2)
        
        self.assertEqual(results[0].name, "spatula")
        self.assertEqual(results[0].location_description, "on the counter top")
        self.assertTrue(results[0].touched)
        
        self.assertEqual(results[1].name, "pan")
        self.assertEqual(results[1].location_description, "on the stove burner")
        self.assertFalse(results[1].touched)

    @patch("google.generativeai.configure")
    @patch("google.generativeai.GenerativeModel")
    def test_parse_markdown_json_response(self, mock_model, mock_configure):
        """Test parsing of JSON response wrapped in markdown code blocks."""
        os.environ["GEMINI_API_KEY"] = "mock-api-key-value"
        detector = GeminiObjectDetector(ObjectDetectorConfig())
        
        markdown_input = """
        ```json
        [
          {"name": "pliers", "location": "in the workbench tray", "touched": false}
        ]
        ```
        """
        results = detector._parse_response(markdown_input)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "pliers")
        self.assertEqual(results[0].location_description, "in the workbench tray")
        self.assertFalse(results[0].touched)

    @patch("google.generativeai.configure")
    @patch("google.generativeai.GenerativeModel")
    def test_parse_dict_wrapped_json_response(self, mock_model, mock_configure):
        """Test parsing of JSON response wrapped in a dictionary."""
        os.environ["GEMINI_API_KEY"] = "mock-api-key-value"
        detector = GeminiObjectDetector(ObjectDetectorConfig())
        
        dict_input = """
        {
          "detected_objects": [
            {"object": "apple", "location_description": "in the bowl", "touched": true}
          ]
        }
        """
        results = detector._parse_response(dict_input)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "apple")
        self.assertEqual(results[0].location_description, "in the bowl")
        self.assertTrue(results[0].touched)

    @patch("google.generativeai.configure")
    @patch("google.generativeai.GenerativeModel")
    def test_parse_regex_fallback_response(self, mock_model, mock_configure):
        """Test parsing fallback using regex on broken JSON structures."""
        os.environ["GEMINI_API_KEY"] = "mock-api-key-value"
        detector = GeminiObjectDetector(ObjectDetectorConfig())
        
        broken_json = """
        Some introduction text.
        {"name": "mug", "location": "on table desk", "touched": false}
        and some trailing comments.
        """
        results = detector._parse_response(broken_json)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "mug")
        self.assertEqual(results[0].location_description, "on table desk")
        self.assertFalse(results[0].touched)

    @patch("google.generativeai.configure")
    @patch("google.generativeai.GenerativeModel")
    def test_parse_plain_text_line_fallback(self, mock_model, mock_configure):
        """Test parsing fallback for plain text line lists."""
        os.environ["GEMINI_API_KEY"] = "mock-api-key-value"
        detector = GeminiObjectDetector(ObjectDetectorConfig())
        
        plain_text = """
        Here is the list of objects I spotted:
        - Knife: on the cutting board (touched: true)
        - Carrot: in the prep bowl (touched: false)
        """
        results = detector._parse_response(plain_text)
        self.assertEqual(len(results), 2)
        
        self.assertEqual(results[0].name, "Knife")
        self.assertEqual(results[0].location_description, "on the cutting board")
        self.assertTrue(results[0].touched)
        
        self.assertEqual(results[1].name, "Carrot")
        self.assertEqual(results[1].location_description, "in the prep bowl")
        self.assertFalse(results[1].touched)


if __name__ == "__main__":
    unittest.main()
