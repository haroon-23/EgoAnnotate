"""Tests for the GeminiLanguageGenerator (Phase D: VLM backend injected as a fake)."""
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datatypes import ActionSegment
from src.language_generator import GeminiLanguageGenerator, LanguageGeneratorConfig
from src.vlm_backend import VLMRequest, VLMResponse


class FakeVLMBackend:
    """In-memory VLMBackend: canned text, records requests."""

    name = "fake"

    def __init__(self, text=""):
        self._text = text
        self.requests = []

    def is_available(self):
        return True

    def generate(self, request):
        self.requests.append(request)
        return VLMResponse(text=self._text, backend="fake", latency_s=0.01)


def _make_generator(text="", **cfg_kwargs):
    return GeminiLanguageGenerator(LanguageGeneratorConfig(**cfg_kwargs),
                                   vlm=FakeVLMBackend(text))


# ---------------------------------------------------------------------------
# Error contract (unchanged)
# ---------------------------------------------------------------------------


def test_api_key_missing_raises_error():
    """Test that ValueError is raised if GEMINI_API_KEY environment variable is missing."""
    with patch.dict(os.environ, {}, clear=True):
        with patch.object(Path, "exists", return_value=False):
            with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable is not set"):
                GeminiLanguageGenerator(LanguageGeneratorConfig())


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_initialization():
    config = LanguageGeneratorConfig()
    assert config.gemini_model == "gemini-3.8-flash"
    assert "overall task" in config.episode_prompt
    assert "description" in config.segment_prompt


# ---------------------------------------------------------------------------
# Generation (fake backend)
# ---------------------------------------------------------------------------


def test_generate_episode_description_success():
    """Test generating episode description and word truncation to 50 words."""
    long_response_text = "word " * 60
    fake = FakeVLMBackend(long_response_text)
    generator = GeminiLanguageGenerator(LanguageGeneratorConfig(), vlm=fake)

    result = generator.generate_episode_description("dummy_path.mp4")

    assert len(result.split()) == 50
    assert result == " ".join(["word"] * 50)
    req = fake.requests[0]
    assert isinstance(req, VLMRequest)
    assert req.video_path == "dummy_path.mp4"
    assert req.video_upload_timeout_s == 10.0


def test_generate_segment_descriptions_success():
    """Test segment descriptions parsing and formatting."""
    vlm_response = """
    Segment 1: picking up a metal spoon
    Segment 2: pouring hot water into a cup
    """
    generator = _make_generator(vlm_response)

    segments = [
        ActionSegment(name="pick_up", start_time=2.5, end_time=5.0, object_name="spoon", hand_used="left"),
        ActionSegment(name="pour", start_time=5.0, end_time=12.5, object_name="cup", hand_used="right")
    ]

    results = generator.generate_segment_descriptions("dummy_path.mp4", segments)
    assert len(results) == 2
    assert results[0] == "picking up a metal spoon"
    assert results[1] == "pouring hot water into a cup"


def test_generate_segment_descriptions_fallback():
    """Test fallback to '{name} the {object}' if segment parsing fails."""
    generator = _make_generator("invalid output format")

    segments = [
        ActionSegment(name="pick_up", start_time=2.5, end_time=5.0, object_name="spoon", hand_used="left", hands=["left"]),
        ActionSegment(name="place_down", start_time=5.0, end_time=12.5, object_name="cup", hand_used="right", hands=["right"])
    ]

    results = generator.generate_segment_descriptions("dummy_path.mp4", segments)
    assert len(results) == 2
    assert results[0] == "pick_up spoon with left hand"
    assert results[1] == "place_down cup with right hand"


def test_generate_episode_description_failure_returns_fallback():
    """Backend failure -> the 'manipulating object' fallback."""
    class Boom(FakeVLMBackend):
        def generate(self, request):
            raise RuntimeError("boom")

    generator = GeminiLanguageGenerator(LanguageGeneratorConfig(), vlm=Boom(""))
    result = generator.generate_episode_description("dummy_path.mp4")
    assert result == "manipulating object"
