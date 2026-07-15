import os
import unittest
from unittest.mock import MagicMock, patch
import sys

# Add parent directory to path to allow import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datatypes import ActionSegment
from src.language_generator import GeminiLanguageGenerator, LanguageGeneratorConfig


class TestGeminiLanguageGenerator(unittest.TestCase):

    def test_config_initialization(self):
        """Test default LanguageGeneratorConfig values."""
        config = LanguageGeneratorConfig()
        self.assertEqual(config.gemini_model, "gemini-1.5-pro-latest")
        self.assertIn("overall task", config.episode_prompt)
        self.assertIn("description", config.segment_prompt)

    def test_api_key_missing_raises_error(self):
        """Test that ValueError is raised if GEMINI_API_KEY environment variable is missing."""
        old_key = os.environ.pop("GEMINI_API_KEY", None)
        try:
            config = LanguageGeneratorConfig()
            with self.assertRaises(ValueError) as context:
                GeminiLanguageGenerator(config)
            self.assertIn("GEMINI_API_KEY environment variable is not set", str(context.exception))
        finally:
            if old_key is not None:
                os.environ["GEMINI_API_KEY"] = old_key

    @patch("google.generativeai.configure")
    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.upload_file")
    def test_generate_episode_description_success(self, mock_upload, mock_model, mock_configure):
        """Test generating episode description and word truncation to 50 words."""
        os.environ["GEMINI_API_KEY"] = "mock-api-key-value"
        generator = GeminiLanguageGenerator(LanguageGeneratorConfig())
        
        # Mock file upload
        mock_file = MagicMock()
        mock_file.state.name = "ACTIVE"
        mock_upload.return_value = mock_file
        
        # Mock model response with >50 words
        long_response_text = "word " * 60
        mock_response = MagicMock()
        mock_response.text = long_response_text
        generator.model.generate_content.return_value = mock_response
        
        result = generator.generate_episode_description("dummy_path.mp4")
        
        # Verify result is truncated to exactly 50 words
        self.assertEqual(len(result.split()), 50)
        self.assertEqual(result, " ".join(["word"] * 50))

    @patch("google.generativeai.configure")
    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.upload_file")
    def test_generate_segment_descriptions_success(self, mock_upload, mock_model, mock_configure):
        """Test segment descriptions parsing and formatting."""
        os.environ["GEMINI_API_KEY"] = "mock-api-key-value"
        generator = GeminiLanguageGenerator(LanguageGeneratorConfig())
        
        mock_file = MagicMock()
        mock_file.state.name = "ACTIVE"
        mock_upload.return_value = mock_file
        
        # Mock VLM returning standard "Segment X: description" output
        vlm_response = """
        Segment 1: picking up a metal spoon
        Segment 2: pouring hot water into a cup
        """
        mock_response = MagicMock()
        mock_response.text = vlm_response
        generator.model.generate_content.return_value = mock_response
        
        segments = [
            ActionSegment(name="pick_up", start_time=2.5, end_time=5.0, object_name="spoon", hand_used="left"),
            ActionSegment(name="pour", start_time=5.0, end_time=12.5, object_name="cup", hand_used="right")
        ]
        
        results = generator.generate_segment_descriptions("dummy_path.mp4", segments)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], "picking up a metal spoon")
        self.assertEqual(results[1], "pouring hot water into a cup")

    @patch("google.generativeai.configure")
    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.upload_file")
    def test_generate_segment_descriptions_fallback(self, mock_upload, mock_model, mock_configure):
        """Test fallback to '{name} the {object}' if segment parsing fails."""
        os.environ["GEMINI_API_KEY"] = "mock-api-key-value"
        generator = GeminiLanguageGenerator(LanguageGeneratorConfig())
        
        mock_file = MagicMock()
        mock_file.state.name = "ACTIVE"
        mock_upload.return_value = mock_file
        
        # Mock VLM returning malformed/empty response
        mock_response = MagicMock()
        mock_response.text = "invalid output format"
        generator.model.generate_content.return_value = mock_response
        
        segments = [
            ActionSegment(name="pick_up", start_time=2.5, end_time=5.0, object_name="spoon", hand_used="left"),
            ActionSegment(name="place_down", start_time=5.0, end_time=12.5, object_name="cup", hand_used="right")
        ]
        
        results = generator.generate_segment_descriptions("dummy_path.mp4", segments)
        self.assertEqual(len(results), 2)
        # Should fallback to formatted names
        self.assertEqual(results[0], "pick up the spoon")
        self.assertEqual(results[1], "place down the cup")


if __name__ == "__main__":
    unittest.main()
