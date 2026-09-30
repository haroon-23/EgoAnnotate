"""Tests for src/perception/sam2_segmenter.py — fakes for sam2, real numpy RLE."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.perception.sam2_segmenter as sam2_mod
from src.perception.sam2_segmenter import (
    Sam2Config,
    Sam2Segmenter,
    create_sam2_segmenter,
    decode_mask_rle,
    encode_mask_rle,
)


class _FakePredictor:
    def __init__(self, h, w):
        self.h, self.w = h, w

    def set_image(self, image_rgb):
        self.h, self.w = image_rgb.shape[:2]

    def predict(self, box, multimask_output):
        x1, y1, x2, y2 = (int(v) for v in box[0])
        masks = np.zeros((1, 1, self.h, self.w), dtype=bool)
        masks[0, 0, y1:y2, x1:x2] = True
        scores = np.array([0.99])
        logits = np.zeros((1, 1, self.h, self.w), dtype=np.float32)
        return masks, scores, logits


@pytest.fixture
def fake_sam2(monkeypatch, tmp_path):
    ckpt = tmp_path / "sam2.1_hiera_tiny.pt"
    ckpt.write_bytes(b"fake")
    monkeypatch.setattr(sam2_mod, "_SAM2_IMPORT_OK", True)
    monkeypatch.setattr(
        sam2_mod, "build_sam2", lambda cfg, ckpt_p, device="cpu": object(), raising=False
    )
    monkeypatch.setattr(
        sam2_mod,
        "SAM2ImagePredictor",
        lambda model: _FakePredictor(48, 64),
        raising=False,
    )
    return str(ckpt)


def test_missing_package_is_unavailable_no_raise(tmp_path):
    # sam2 is not installed in this sandbox.
    seg = Sam2Segmenter(
        Sam2Config(enabled=True, checkpoint=str(tmp_path / "x.pt"))
    )
    assert seg.is_available() is False
    assert create_sam2_segmenter(Sam2Config(enabled=True)) is None


def test_disabled_returns_none():
    assert create_sam2_segmenter(Sam2Config(enabled=False)) is None


def test_predict_masks_with_fake(fake_sam2, tmp_path):
    cfg_yaml = tmp_path / "sam2_hiera_t.yaml"
    cfg_yaml.write_text("# fake sam2 config")
    seg = Sam2Segmenter(
        Sam2Config(enabled=True, checkpoint=fake_sam2, config=str(cfg_yaml))
    )
    assert seg.is_available()
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    masks = seg.predict_masks(img, [np.array([0.25, 0.25, 0.75, 0.75])])
    assert len(masks) == 1
    m = masks[0]
    assert m.dtype == bool and m.shape == (48, 64)
    # Fake fills the box interior: [16:48, 12:48] approx.
    assert m[24, 32]  # inside
    assert not m[0, 0]  # outside


def test_rle_roundtrip():
    rng = np.random.default_rng(5)
    for mask in [
        rng.random((37, 53)) > 0.7,
        np.zeros((10, 10), dtype=bool),
        np.ones((10, 10), dtype=bool),
        np.zeros((8, 8), dtype=bool),
    ]:
        mask[0, 0] = True  # exercise the leading-1 run edge case
        rle = encode_mask_rle(mask)
        assert rle["size"] == [mask.shape[0], mask.shape[1]]
        assert all(isinstance(c, int) for c in rle["counts"])
        np.testing.assert_array_equal(decode_mask_rle(rle), mask)


def test_rle_is_json_safe():
    import json

    mask = np.zeros((16, 16), dtype=bool)
    mask[4:8, 4:8] = True
    rle = encode_mask_rle(mask)
    json.dumps(rle)  # must not raise
