"""Parsing parity between the gemini and local VLM backends (Phase D).

Identical canned responses delivered in the two shapes (gemini native vs
local 500M chatter) must produce identical stage outputs — the abstraction
changes nothing. All backends are in-memory fakes.
"""
import os
import sys

import pytest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.action_segmenter import ActionSegmenterConfig, GeminiActionSegmenter
from src.language_generator import LanguageGeneratorConfig, GeminiLanguageGenerator
from src.object_detector import ObjectDetectorConfig, GeminiObjectDetector
from src.segment_labeler import SegmentLabeler, SegmentLabelerConfig
from src.vlm_backend import VLMRequest, VLMResponse


# ---------------------------------------------------------------------------
# Fake
# ---------------------------------------------------------------------------


class FakeVLMBackend:
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


def _img():
    return Image.new("RGB", (64, 48), color=(10, 20, 30))


# Gemini native shape: clean JSON.
GEMINI_OD = (
    '[{"name": "red cup", "location": "left", "touched": true}, '
    '{"name": "blue book", "location": "right", "touched": false}]'
)
# Local 500M shape: prose + markdown-wrapped JSON (same data).
LOCAL_OD = (
    "Here are the objects I found in the frame:\n"
    '```json\n[{"name": "red cup", "location": "left", "touched": true},\n'
    '{"name": "blue book", "location": "right", "touched": false}]\n```\n'
    "That's everything I can see."
)

GEMINI_SEG = (
    '[{"name": "pick_up", "start_time": 1.0, "end_time": 3.5, '
    '"object_name": "cup", "hand_used": "right", "description": "pick up cup"}, '
    '{"name": "place", "start_time": 3.5, "end_time": 6.0, '
    '"object_name": "cup", "hand_used": "right", "description": "place cup"}]'
)
LOCAL_SEG = (
    "Segments:\n```json\n"
    '[{"name": "pick_up", "start_time": 1.0, "end_time": 3.5, '
    '"object_name": "cup", "hand_used": "right", "description": "pick up cup"}, '
    '{"name": "place", "start_time": 3.5, "end_time": 6.0, '
    '"object_name": "cup", "hand_used": "right", "description": "place cup"}]\n```'
)

GEMINI_LABEL = '{"action": "grasp", "description": "grasping the red cup"}'
LOCAL_LABEL = (
    "My best guess:\n```json\n"
    '{"action": "grasp", "description": "grasping the red cup"}\n```'
)


# ---------------------------------------------------------------------------
# Object detector
# ---------------------------------------------------------------------------


def _od_annotations(text):
    detector = GeminiObjectDetector(ObjectDetectorConfig(), vlm=FakeVLMBackend(text))
    anns = detector._call_vlm_fast(_img())
    assert anns is not None
    return [(a.name, a.location_description, a.touched) for a in anns]


def test_object_detector_gemini_vs_local_shape_identical():
    gemini_out = _od_annotations(GEMINI_OD)
    local_out = _od_annotations(LOCAL_OD)
    assert gemini_out == local_out == [
        ("red cup", "left", True),
        ("blue book", "right", False),
    ]


def test_object_detector_local_shape_needs_no_repair():
    """Markdown-wrapped JSON parses on the strict pass — the repair re-prompt
    is only a last resort for real 500M chatter."""
    fake = FakeVLMBackend(LOCAL_OD)
    detector = GeminiObjectDetector(ObjectDetectorConfig(), vlm=fake)
    anns = detector._call_vlm_fast(_img())
    assert anns is not None and len(anns) == 2
    # Only one VLM call: the repair re-prompt was not needed.
    assert len(fake.requests) == 1


# ---------------------------------------------------------------------------
# Action segmenter
# ---------------------------------------------------------------------------


def _seg_segments(text):
    segmenter = GeminiActionSegmenter(ActionSegmenterConfig(), vlm=FakeVLMBackend(text))
    return segmenter._try_video_upload("dummy.mp4")


def test_action_segmenter_gemini_vs_local_shape_identical():
    gemini_out = _seg_segments(GEMINI_SEG)
    local_out = _seg_segments(LOCAL_SEG)
    assert gemini_out is not None and local_out is not None
    gemini_tpl = [(s.start_time, s.end_time, s.name) for s in gemini_out]
    local_tpl = [(s.start_time, s.end_time, s.name) for s in local_out]
    assert gemini_tpl == local_tpl == [
        (1.0, 3.5, "pick_up"),
        (3.5, 6.0, "place"),
    ]


# ---------------------------------------------------------------------------
# Segment labeler
# ---------------------------------------------------------------------------


def _label(text):
    labeler = SegmentLabeler(SegmentLabelerConfig(), vlm=FakeVLMBackend(text))
    return labeler._call_vlm_with_retry(_img(), "label prompt")


def test_segment_labeler_gemini_vs_local_shape_identical():
    gemini_out = _label(GEMINI_LABEL)
    local_out = _label(LOCAL_LABEL)
    assert gemini_out == local_out == {
        "action": "grasp",
        "description": "grasping the red cup",
    }


# ---------------------------------------------------------------------------
# Language generator
# ---------------------------------------------------------------------------


def test_language_generator_gemini_vs_local_shape_identical():
    gemini_canned = "Segment 1: The person reaches for the cup.\nSegment 2: The person places the cup down."
    local_canned = (
        "Descriptions:\nSegment 1: The person reaches for the cup.\n"
        "Segment 2: The person places the cup down.\nThat's all."
    )
    gen_g = GeminiLanguageGenerator(LanguageGeneratorConfig(), vlm=FakeVLMBackend(gemini_canned))
    gen_l = GeminiLanguageGenerator(LanguageGeneratorConfig(), vlm=FakeVLMBackend(local_canned))
    from src.datatypes import ActionSegment

    segs = [ActionSegment(name="reach", start_time=0.0, end_time=2.0,
                         object_name="cup", hand_used="right"),
            ActionSegment(name="place", start_time=2.0, end_time=4.0,
                         object_name="cup", hand_used="right")]
    out_g = gen_g.generate_segment_descriptions("dummy.mp4", segs)
    out_l = gen_l.generate_segment_descriptions("dummy.mp4", segs)
    assert out_g == out_l
    assert out_g[0] == "The person reaches for the cup."
    assert out_l[1] == "The person places the cup down."
