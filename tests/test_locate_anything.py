"""Tests for the LocateAnything-3B detector backend (src/perception/detector.py).

Hermetic: fakes for the model/query path; sys.modules blocking for the
missing-transformers case. No weights, no downloads.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.perception.detector as det_mod
from src.perception.detector import (
    Detector2DConfig,
    LocateAnythingDetector,
    create_detector_2d,
    parse_locate_anything_boxes,
)


@pytest.fixture
def no_transformers_la(monkeypatch):
    """Simulate an env without transformers (works whether or not installed)."""
    for name in list(sys.modules):
        if name == "src.perception" or name.startswith("src.perception."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "transformers", None)
    yield


# ---------------------------------------------------------------------------
# Box-token parsing
# ---------------------------------------------------------------------------


def test_parse_single_box():
    boxes = parse_locate_anything_boxes("Found it <box> 100, 200, 300, 400 </box> here.")
    assert len(boxes) == 1
    np.testing.assert_allclose(boxes[0], [0.1, 0.2, 0.3, 0.4], rtol=1e-6)


def test_parse_multiple_boxes_preserve_order():
    text = "<box> 0, 0, 100, 100 </box> and <box> 500, 500, 900, 900 </box>"
    boxes = parse_locate_anything_boxes(text)
    assert len(boxes) == 2
    np.testing.assert_allclose(boxes[0], [0.0, 0.0, 0.1, 0.1], rtol=1e-6)
    np.testing.assert_allclose(boxes[1], [0.5, 0.5, 0.9, 0.9], rtol=1e-6)


def test_parse_point_token_ignored():
    # 2-int spans are points, not boxes -> no detections.
    assert parse_locate_anything_boxes("<box> 500, 500 </box>") == []


def test_parse_malformed_and_empty():
    assert parse_locate_anything_boxes("") == []
    assert parse_locate_anything_boxes("no boxes here") == []
    assert parse_locate_anything_boxes("<box> 1, 2, 3 </box>") == []  # 3 ints
    assert parse_locate_anything_boxes("<box> 1, 2, 3, 4, 5 </box>") == []  # 5 ints
    assert parse_locate_anything_boxes("<box> abc </box>") == []  # no ints


def test_parse_clamps_and_reorders():
    boxes = parse_locate_anything_boxes("<box> 1200, -50, 800, 500 </box>")
    assert len(boxes) == 1
    np.testing.assert_allclose(boxes[0], [0.8, 0.0, 1.0, 0.5], rtol=1e-6)
    assert boxes[0].min() >= 0.0 and boxes[0].max() <= 1.0


# ---------------------------------------------------------------------------
# Per-label query fan-out (fake model)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_locate_anything(monkeypatch):
    """Fake the query path: canned text per label, no model load."""
    monkeypatch.setattr(det_mod, "_LOCATE_ANYTHING_IMPORT_OK", True)
    canned = {
        "mug": "Sure <box> 100, 100, 200, 200 </box>",
        "bottle": "<box> 300, 300, 400, 400 </box> and <box> 500, 500, 600, 600 </box>",
    }
    calls = []

    def fake_ensure(self):
        return True

    def fake_generate(self, image_pil, label):
        calls.append(label)
        return canned.get(label, "")

    monkeypatch.setattr(LocateAnythingDetector, "_ensure_model", fake_ensure)
    monkeypatch.setattr(LocateAnythingDetector, "_generate_text", fake_generate)
    return calls


def test_detect_queries_once_per_label(fake_locate_anything):
    calls = fake_locate_anything
    det = LocateAnythingDetector(Detector2DConfig(backend="locate_anything"))
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    dets = det.detect(img, ["mug", "bottle"])
    assert calls == ["mug", "bottle"]  # one query per label
    assert len(dets) == 3
    assert [d.label for d in dets] == ["mug", "bottle", "bottle"]
    assert all(d.score == 1.0 for d in dets)  # model emits no scores
    for d in dets:
        assert d.bbox_xyxy_norm.min() >= 0.0 and d.bbox_xyxy_norm.max() <= 1.0
    np.testing.assert_allclose(dets[0].bbox_xyxy_norm, [0.1, 0.1, 0.2, 0.2], rtol=1e-6)


def test_detect_empty_and_garbage_never_raise(fake_locate_anything, monkeypatch):
    det = LocateAnythingDetector(Detector2DConfig(backend="locate_anything"))
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    monkeypatch.setattr(
        LocateAnythingDetector, "_generate_text", lambda self, p, l: ""
    )
    assert det.detect(img, ["mug"]) == []
    monkeypatch.setattr(
        LocateAnythingDetector,
        "_generate_text",
        lambda self, p, l: "not a box <box> xyz </box>",
    )
    assert det.detect(img, ["mug"]) == []
    assert det.detect(img, []) == []
    assert det.detect(img, ["  "]) == []  # blank labels skipped


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


def test_unavailable_without_transformers(no_transformers_la):
    from src.perception.detector import (
        Detector2DConfig,
        LocateAnythingDetector,
        create_detector_2d,
    )

    det = LocateAnythingDetector(Detector2DConfig(backend="locate_anything"))
    assert det.is_available() is False
    assert det.detect(np.zeros((32, 32, 3), np.uint8), ["mug"]) == []
    assert create_detector_2d(Detector2DConfig(backend="locate_anything")) is None


def test_factory_returns_none_when_weights_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(det_mod, "_LOCATE_ANYTHING_IMPORT_OK", True)
    cfg = Detector2DConfig(
        backend="locate_anything",
        locate_anything_weights=str(tmp_path / "nope"),
    )
    assert create_detector_2d(cfg) is None


def test_version_mismatch_warns_but_does_not_raise(monkeypatch, tmp_path):
    monkeypatch.setattr(det_mod, "_LOCATE_ANYTHING_IMPORT_OK", True)
    monkeypatch.setattr(
        det_mod,
        "_locate_anything_transformers_status",
        lambda: (True, "4.60.0", False),
    )
    det = LocateAnythingDetector(
        Detector2DConfig(
            backend="locate_anything",
            locate_anything_weights=str(tmp_path / "nope"),
        )
    )
    # Missing weights -> unavailable; the version mismatch only warns.
    assert det.is_available() is False


def test_default_backend_unchanged():
    assert Detector2DConfig().backend == "owlvit"


def test_factory_unknown_backend_still_raises():
    with pytest.raises(ValueError, match="Unknown detector backend"):
        create_detector_2d(Detector2DConfig(backend="not_a_backend"))
