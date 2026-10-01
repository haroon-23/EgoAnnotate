"""Tests for the GeminiActionSegmenter (Phase D: VLM backend injected as a fake)."""
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datatypes import ActionSegment
from src.action_segmenter import GeminiActionSegmenter, ActionSegmenterConfig
from src.vlm_backend import VLMRequest, VLMResponse


class FakeVLMBackend:
    """In-memory VLMBackend: canned text, records requests, optional error."""

    name = "fake"

    def __init__(self, text="{}", exc=None):
        self._text = text
        self._exc = exc
        self.requests = []

    def is_available(self):
        return True

    def generate(self, request):
        self.requests.append(request)
        if self._exc is not None:
            raise self._exc
        return VLMResponse(text=self._text, backend="fake", latency_s=0.01)


def _make_segmenter(text="{}", exc=None, **cfg_kwargs):
    return GeminiActionSegmenter(ActionSegmenterConfig(**cfg_kwargs),
                                 vlm=FakeVLMBackend(text, exc=exc))


def _canned():
    return (
        '[{"name": "pick_up", "start_time": "0:02.5", "end_time": "0:08.0", '
        '"object_name": "spoon", "hand_used": "left", '
        '"description": "picking up the silver spoon"}]'
    )


# ---------------------------------------------------------------------------
# Error contract (unchanged)
# ---------------------------------------------------------------------------


def test_api_key_missing_raises_error():
    """Test that ValueError is raised if GEMINI_API_KEY environment variable is missing."""
    with patch.dict(os.environ, {}, clear=True):
        with patch.object(Path, "exists", return_value=False):
            with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable is not set"):
                GeminiActionSegmenter(ActionSegmenterConfig())


# ---------------------------------------------------------------------------
# Pure parsing (unchanged behavior)
# ---------------------------------------------------------------------------


def test_time_parsing_formats():
    segmenter = _make_segmenter()
    assert segmenter._parse_time("01:30.50") == 90.50
    assert segmenter._parse_time("02:15") == 135.0
    assert segmenter._parse_time("01:00:10") == 3610.0
    assert segmenter._parse_time("4.25") == 4.25
    assert segmenter._parse_time("10") == 10.0
    assert segmenter._parse_time("") == 0.0
    assert segmenter._parse_time("invalid") == 0.0


def test_parse_json_response():
    segmenter = _make_segmenter()
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
    assert len(results) == 1
    assert results[0].name == "pick_up"
    assert results[0].start_time == 2.5
    assert results[0].end_time == 8.0
    assert results[0].object_name == "spoon"
    assert results[0].hand_used == "left"
    assert results[0].description == "picking up the silver spoon"


def test_parse_text_fallback():
    segmenter = _make_segmenter()
    fallback_text = """
    Below is the temporal segmentation of the video:
    00:02.5 - 00:08.0: pick_up holding the red cup with the right hand
    00:08.0 - 00:15.20: pour pouring water into the bowl using both hands
    """
    results = segmenter._parse_text_fallback(fallback_text)
    assert len(results) == 2
    assert results[0].name == "pick_up"
    assert results[0].start_time == 2.5
    assert results[0].end_time == 8.0
    assert results[0].object_name == "cup"
    assert results[0].hand_used == "right"
    assert results[0].description == "holding the red cup with the right hand"
    assert results[1].name == "pour"
    assert results[1].start_time == 8.0
    assert results[1].end_time == pytest.approx(15.20)
    assert results[1].object_name == "bowl"
    assert results[1].hand_used == "both"
    assert results[1].description == "pouring water into the bowl using both hands"


# ---------------------------------------------------------------------------
# Video upload path (fake backend)
# ---------------------------------------------------------------------------


def test_segment_video_success_sends_video_path():
    fake = FakeVLMBackend(_canned())
    segmenter = GeminiActionSegmenter(ActionSegmenterConfig(), vlm=fake)
    results = segmenter.segment_video("dummy_path.mp4")
    assert len(results) == 1
    assert results[0].name == "pick_up"
    assert results[0].start_time == 2.5
    req = fake.requests[0]
    assert isinstance(req, VLMRequest)
    assert req.video_path == "dummy_path.mp4"
    assert "temporal segments" in req.prompt


def test_default_fallback_segment_on_transient_failure(monkeypatch):
    """Transient (non-fatal) backend errors retry, then fall back to one
    default segment — the old upload-failure behavior."""
    import src.action_segmenter as seg_mod
    monkeypatch.setattr(seg_mod, "RETRY_DELAY", 0)  # keep the test fast
    segmenter = _make_segmenter(exc=ConnectionError("Upload failed"))
    results = segmenter.segment_video("dummy_path.mp4")
    assert len(results) == 1
    assert results[0].name == "manipulate"
    assert results[0].start_time == 0.0
    assert results[0].end_time == 10.0
    assert results[0].object_name == "unknown"
    assert results[0].hand_used == "right"


def test_fatal_backend_error_propagates():
    """RuntimeError from the backend is fatal — same as the old SDK contract."""
    segmenter = _make_segmenter(exc=RuntimeError("auth blew up"))
    with pytest.raises(RuntimeError, match="auth blew up"):
        segmenter.segment_video("dummy_path.mp4")
