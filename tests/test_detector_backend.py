"""Tests for the unified 2D detector interface (src/perception/detector.py).

Uses fakes for the heavy model objects — no weights, no downloads.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.perception.detector as det_mod
from src.perception.detector import (
    Detector2DConfig,
    GroundingDinoDetector,
    OwlViTDetector,
    create_detector_2d,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeInputs(dict):
    def to(self, device):
        return self


class _FakeOwlProcessor:
    """Fake transformers AutoProcessor: canned pixel-space box."""

    def __call__(self, images, text, return_tensors):
        return _FakeInputs()

    def post_process_object_detection(self, outputs, threshold, target_sizes):
        return [
            {
                "boxes": torch.tensor([[10.0, 20.0, 110.0, 120.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
        ]


class _FakeOwlModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        return object()


@pytest.fixture
def fake_owlvit(monkeypatch):
    monkeypatch.setattr(det_mod, "_OWL_VIT_IMPORT_OK", True)
    monkeypatch.setattr(
        det_mod, "AutoProcessor",
        type("AP", (), {"from_pretrained": staticmethod(lambda *a, **k: _FakeOwlProcessor())}),
        raising=False,
    )
    monkeypatch.setattr(
        det_mod, "OwlViTForObjectDetection",
        type("OM", (), {"from_pretrained": staticmethod(lambda *a, **k: _FakeOwlModel())}),
        raising=False,
    )
    monkeypatch.setattr(det_mod, "torch", torch, raising=False)


@pytest.fixture
def fake_gdino(monkeypatch, tmp_path):
    """Fake groundingdino package surface + real weight/config files."""
    weights = tmp_path / "groundingdino_swint_ogc.pth"
    weights.write_bytes(b"fake")
    cfg = tmp_path / "GroundingDINO_SwinT_OGC.py"
    cfg.write_text("# fake config")

    def fake_load_model(cfg_path, weights_path, device="cpu"):
        return object()

    def fake_predict(model, image, caption, box_threshold, text_threshold, device):
        boxes = torch.tensor([[0.5, 0.5, 0.4, 0.4]])  # normalized cxcywh
        logits = torch.tensor([0.9])
        return boxes, logits, ["mug"]

    class FakeBoxOps:
        @staticmethod
        def box_cxcywh_to_xyxy(boxes):
            cx, cy, w, h = boxes.unbind(-1)
            return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

    monkeypatch.setattr(det_mod, "_GDINO_IMPORT_OK", True)
    monkeypatch.setattr(det_mod, "load_model", fake_load_model, raising=False)
    monkeypatch.setattr(det_mod, "predict", fake_predict, raising=False)
    monkeypatch.setattr(det_mod, "box_ops", FakeBoxOps(), raising=False)
    return str(weights), str(cfg)


@pytest.fixture
def no_transformers(monkeypatch):
    """Simulate an env without transformers, whether or not it's installed.

    Evicts the perception modules so their guarded imports re-run with
    ``transformers`` blocked (``sys.modules["transformers"] = None`` makes
    ``from transformers import ...`` raise ImportError).
    """
    for name in list(sys.modules):
        if (
            name == "src.perception"
            or name.startswith("src.perception.")
            or name == "src.grounding_detector"
        ):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "transformers", None)
    yield



# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_owlvit_detect_normalized_boxes(fake_owlvit):
    det = OwlViTDetector(Detector2DConfig(backend="owlvit"))
    assert det.is_available()
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    dets = det.detect(img, ["mug"])
    assert len(dets) == 1
    d = dets[0]
    # Fake box was [10, 20, 110, 120] px on a 200x200 image.
    np.testing.assert_allclose(
        d.bbox_xyxy_norm, [0.05, 0.10, 0.55, 0.60], rtol=1e-5
    )
    assert d.bbox_xyxy_norm.min() >= 0.0 and d.bbox_xyxy_norm.max() <= 1.0
    assert d.label == "mug"
    assert d.score == pytest.approx(0.9)


def test_owlvit_unavailable_without_deps(no_transformers):
    # transformers blocked -> graceful degradation, no raise.
    from src.perception.detector import (
        Detector2DConfig,
        OwlViTDetector,
        create_detector_2d,
    )

    det = OwlViTDetector(Detector2DConfig(backend="owlvit"))
    assert det.is_available() is False
    assert det.detect(np.zeros((32, 32, 3), np.uint8), ["mug"]) == []
    assert create_detector_2d(Detector2DConfig(backend="owlvit")) is None


def test_gdino_detect_converts_pixel_to_normalized(fake_gdino):
    weights, cfg = fake_gdino
    det = GroundingDinoDetector(
        Detector2DConfig(
            backend="grounding_dino",
            grounding_dino_weights=weights,
            grounding_dino_config=cfg,
        )
    )
    assert det.is_available()
    img = np.zeros((100, 200, 3), dtype=np.uint8)  # h=100, w=200
    dets = det.detect(img, ["mug"])
    assert len(dets) == 1
    d = dets[0]
    # Fake cxcywh [0.5, 0.5, 0.4, 0.4] -> px [60, 30, 140, 70] -> norm [0.3, 0.3, 0.7, 0.7]
    np.testing.assert_allclose(d.bbox_xyxy_norm, [0.3, 0.3, 0.7, 0.7], rtol=1e-5)
    assert d.bbox_xyxy_norm.min() >= 0.0 and d.bbox_xyxy_norm.max() <= 1.0
    assert d.label == "mug"


def test_gdino_missing_weights_is_unavailable_no_raise(tmp_path):
    det = GroundingDinoDetector(
        Detector2DConfig(
            backend="grounding_dino",
            grounding_dino_weights=str(tmp_path / "nope.pth"),
            grounding_dino_config=str(tmp_path / "nope.py"),
        )
    )
    assert det.is_available() is False
    assert det.detect(np.zeros((32, 32, 3), np.uint8), ["mug"]) == []
    assert (
        create_detector_2d(
            Detector2DConfig(
                backend="grounding_dino",
                grounding_dino_weights=str(tmp_path / "nope.pth"),
            )
        )
        is None
    )


def test_factory_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown detector backend"):
        create_detector_2d(Detector2DConfig(backend="not_a_backend"))


def test_legacy_shim_still_works(no_transformers):
    """src/grounding_detector.py keeps the old OWL-ViT API alive."""
    from src.grounding_detector import (
        GroundingDINOConfig,
        GroundingDINODetector,
        create_grounding_detector,
    )

    cfg = GroundingDINOConfig()
    assert cfg.model_name == "google/owlvit-base-patch32"
    # transformers blocked -> factory returns None gracefully
    assert create_grounding_detector(cfg) is None
    det = GroundingDINODetector(cfg)
    assert det.is_available() is False
    assert det.detect(np.zeros((32, 32, 3), np.uint8), ["mug"]) == []
