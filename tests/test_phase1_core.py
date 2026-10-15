"""Phase 1 tests: Fail-loud behavior, no 5-finger fallback, fixed-anchor metric scaling, camera motion, caching."""
from __future__ import annotations

from pathlib import Path
import sys

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pytest

from src.atomic_io import RunManifest, atomic_write_json
from src.cache_manager import StageCache, VLMDiskCache
from src.camera_motion import estimate_camera_motion
from src.contact_detector import ContactDetector, ContactDetectorConfig
from src.datatypes import AnnotationFrame, HandLandmarks, ObjectAnnotation
from src.language_generator import GeminiLanguageGenerator, LanguageGeneratorConfig
from src.retargeting.pose_mapper import PoseMapper, PoseMapperConfig
from src.vlm_backend import VLMRequest, VLMResponse


def _make_hand(cx: float = 0.5, cy: float = 0.5, cz: float = 0.0, palm_size: float = 0.12) -> HandLandmarks:
    x = np.full(21, cx, dtype=np.float32)
    y = np.full(21, cy, dtype=np.float32)
    z = np.full(21, cz, dtype=np.float32)
    # Wrist at (cx, cy), middle MCP (9) at (cx, cy - palm_size), index MCP (5), pinky MCP (17)
    x[5], y[5] = cx - palm_size * 0.5, cy - palm_size
    x[9], y[9] = cx, cy - palm_size
    x[17], y[17] = cx + palm_size * 0.5, cy - palm_size
    return HandLandmarks(x=x, y=y, z=z, confidence=0.95, handedness="Right")


def test_contact_detector_no_bbox_never_marks_five_fingertips():
    """Rule 4/5: An object with touched=True but bbox=None must NOT mark all 5 fingertips in contact."""
    det = ContactDetector(ContactDetectorConfig())
    hand = _make_hand()
    obj_no_bbox = ObjectAnnotation(name="mug", location_description="table", touched=True, bbox=None)
    state = det.detect_contact(hand, [obj_no_bbox])
    assert state is not None
    assert state.in_contact is False
    assert int(np.sum(state.fingers)) == 0
    assert state.object_name is None


class _FailingVLM:
    name = "failing_mock"

    def is_available(self) -> bool:
        return True

    def generate(self, request: VLMRequest) -> VLMResponse:
        raise RuntimeError("Simulated 503 VLM outage")


class _EmptyVLM:
    name = "empty_mock"

    def is_available(self) -> bool:
        return True

    def generate(self, request: VLMRequest) -> VLMResponse:
        return VLMResponse(text="   ", backend="mock", latency_s=0.01)


def test_language_generator_fails_loud_never_returns_canned_text():
    """Rule 4/5: LanguageGenerator must raise RuntimeError on VLM failure, never 'manipulating object'."""
    cfg = LanguageGeneratorConfig()
    gen_fail = GeminiLanguageGenerator(cfg, vlm=_FailingVLM())
    with pytest.raises(RuntimeError):
        gen_fail.generate_episode_description("dummy.mp4")

    gen_empty = GeminiLanguageGenerator(cfg, vlm=_EmptyVLM())
    with pytest.raises(RuntimeError):
        gen_empty.generate_episode_description("dummy.mp4")


def test_pose_mapper_fixed_anchor_preserves_small_motion_magnitude():
    """Phase 1: Small 2% normalized wrist micro-motion must NOT stretch across the entire 35cm workspace."""
    cfg = PoseMapperConfig(enable_trajectory_smoothing=False, enable_metric_calibration=False)
    mapper = PoseMapper(cfg)

    frames = []
    for i in range(10):
        # Move wrist by only +0.02 normalized units (~1.4 cm) across the clip
        hand = _make_hand(cx=0.5 + 0.002 * i, cy=0.5, cz=0.0, palm_size=0.12)
        frames.append(
            AnnotationFrame(
                frame_idx=i,
                timestamp=i / 30.0,
                image_path=f"frame_{i}.jpg",
                left_hand=None,
                right_hand=hand,
            )
        )

    targets = mapper.map_frames(frames)
    positions = np.stack([t.position for t in targets], axis=0)
    span_m = np.ptp(positions, axis=0)
    # Under old min-max stretching, span_m[1] would be 0.60m (full y workspace)!
    # Under fixed-anchor metric scaling, 0.018 norm units * ~0.7m/unit is ~0.013m (< 0.05m).
    assert float(np.max(span_m)) < 0.05, f"Micro-motion stretched to {span_m}m!"
    assert targets[0].scaling_metadata["method"] == "fixed_anchor_anthropometric_metric"


def test_manifest_and_stage_cache(tmp_path: Path):
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"fake_video_bytes_12345")

    cache = StageCache(vid, cache_root=tmp_path / "cache", enabled=True)
    calls = {"n": 0}

    def _expensive():
        calls["n"] += 1
        return [1, 2, 3]

    res1, cached1 = cache.load_or_compute("stage_a", _expensive)
    res2, cached2 = cache.load_or_compute("stage_a", _expensive)
    assert res1 == [1, 2, 3] and not cached1
    assert res2 == [1, 2, 3] and cached2
    assert calls["n"] == 1

    m = RunManifest(episode_id="ep1", video_path=str(vid), allow_partial=True)
    m.record_stage("stage_a", "ok", 0.12)
    m.record_stage("stage_b", "degraded", 0.05, "optional VLM timeout")
    assert m.exit_code == 2
    m_path = tmp_path / "manifest.json"
    m.save(m_path)
    assert m_path.exists()
