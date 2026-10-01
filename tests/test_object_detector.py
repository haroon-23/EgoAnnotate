"""Tests for the GeminiObjectDetector (Phase D: VLM backend injected as a fake)."""
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.object_detector import GeminiObjectDetector, ObjectDetectorConfig
from src.vlm_backend import VLMRequest, VLMResponse


class FakeVLMBackend:
    """In-memory VLMBackend: canned text, records requests."""

    name = "fake"

    def __init__(self, text="{}", fail=False):
        self._text = text
        self._fail = fail
        self.requests = []

    def is_available(self):
        return True

    def generate(self, request):
        self.requests.append(request)
        if self._fail:
            raise RuntimeError("fake backend failure")
        return VLMResponse(text=self._text, backend="fake", latency_s=0.01)


def _make_detector(text="{}", fail=False, **cfg_kwargs):
    return GeminiObjectDetector(ObjectDetectorConfig(**cfg_kwargs), vlm=FakeVLMBackend(text, fail=fail))


def _make_video(path, n_frames=3, w=64, h=48, fps=10):
    import cv2
    import numpy as np
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for i in range(n_frames):
        writer.write(np.full((h, w, 3), (i * 40) % 256, dtype=np.uint8))
    writer.release()
    return str(path)


def _canned():
    return (
        '[{"name": "red cup", "location": "center", "touched": true}, '
        '{"name": "blue book", "location": "right", "touched": false}]'
    )


# ---------------------------------------------------------------------------
# Error contract (unchanged)
# ---------------------------------------------------------------------------


def test_api_key_missing_raises_error():
    with patch.dict(os.environ, {}, clear=True):
        with patch.object(Path, "exists", return_value=False):
            with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable is not set"):
                GeminiObjectDetector(ObjectDetectorConfig())


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_detector_initialization_success():
    config = ObjectDetectorConfig()
    detector = GeminiObjectDetector(config, vlm=FakeVLMBackend(_canned()))
    assert detector.config is config
    assert detector._vlm.name == "fake"


def test_detector_uses_injected_backend():
    fake = FakeVLMBackend(_canned())
    detector = GeminiObjectDetector(ObjectDetectorConfig(), vlm=fake)
    assert detector._vlm is fake


# ---------------------------------------------------------------------------
# Parsing (pure, unchanged behavior)
# ---------------------------------------------------------------------------


def test_parse_response_valid_json():
    detector = _make_detector()
    annotations = detector._parse_response(_canned())
    assert len(annotations) == 2
    assert annotations[0].name == "red cup"
    assert annotations[0].touched is True
    assert annotations[1].name == "blue book"
    assert annotations[1].location_description == "right"


def test_parse_response_invalid_json_falls_back():
    detector = _make_detector()
    annotations = detector._parse_response("not valid json at all")
    assert annotations == []


def test_parse_response_markdown_fence():
    detector = _make_detector()
    text = '```json\n[{"name": "cup", "location": "left", "touched": false}]\n```'
    annotations = detector._parse_response(text)
    assert len(annotations) == 1
    assert annotations[0].name == "cup"


# ---------------------------------------------------------------------------
# VLM call behavior (fake backend)
# ---------------------------------------------------------------------------


def test_call_vlm_fast_sends_image_and_prompt():
    fake = FakeVLMBackend(_canned())
    detector = GeminiObjectDetector(ObjectDetectorConfig(), vlm=fake)
    img = Image.new("RGB", (64, 48))
    annotations = detector._call_vlm_fast(img)
    assert annotations is not None and len(annotations) == 2
    req = fake.requests[0]
    assert isinstance(req, VLMRequest)
    assert req.images == [img]
    assert "JSON" in req.prompt


def test_call_vlm_fast_garbage_returns_none():
    detector = _make_detector(text="not json at all")
    img = Image.new("RGB", (64, 48))
    assert detector._call_vlm_fast(img) is None


def test_detect_objects_api_error_propagates(tmp_path):
    detector = _make_detector(fail=True)
    video = _make_video(tmp_path / "v.mp4")
    with pytest.raises(RuntimeError, match="fake backend failure"):
        detector.detect_objects(video)


def test_detect_objects_success_from_video(tmp_path):
    detector = _make_detector(text=_canned())
    video = _make_video(tmp_path / "v.mp4")
    objects = detector.detect_objects(video)
    # 3 keyframes x 2 objects, deduplicated by name -> 2 unique objects.
    assert {o.name for o in objects} == {"red cup", "blue book"}


def test_quota_pacing_only_for_gemini_backend(tmp_path):
    video = _make_video(tmp_path / "v.mp4")

    gemini_like = FakeVLMBackend(_canned())
    gemini_like.name = "gemini"
    detector = GeminiObjectDetector(ObjectDetectorConfig(), vlm=gemini_like)
    with patch("time.sleep") as mock_sleep:
        detector.detect_objects(video)
    assert mock_sleep.call_count == 3  # one 10s pace per keyframe
    mock_sleep.assert_called_with(10)

    local_like = FakeVLMBackend(_canned())
    detector = GeminiObjectDetector(ObjectDetectorConfig(), vlm=local_like)
    with patch("time.sleep") as mock_sleep:
        detector.detect_objects(video)
    mock_sleep.assert_not_called()
