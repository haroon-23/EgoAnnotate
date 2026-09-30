"""Tests for src/perception/depth.py — fakes for onnxruntime, synthetic depth maps."""
import logging
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.perception.depth as depth_mod
from src.perception.depth import (
    DepthConfig,
    DisabledDepth,
    UniDepthEstimator,
    create_depth_estimator,
)
from src.perception.pnp_pose import CameraIntrinsics, backproject_to_3d


class _FakeInput:
    name = "image"
    shape = [1, 3, 448, 448]


class _FakeSession:
    def get_inputs(self):
        return [_FakeInput()]

    def run(self, output_names, input_feed):
        depth_mm = np.ones((448, 448), dtype=np.float32) * 1500.0  # 1.5 m in mm
        return [np.expand_dims(depth_mm, 0)]


class _FakeOrt:
    @staticmethod
    def InferenceSession(path, providers=None):
        return _FakeSession()


@pytest.fixture
def fake_ort(monkeypatch, tmp_path):
    weights = tmp_path / "unidepth_v1.onnx"
    weights.write_bytes(b"fake-onnx")
    monkeypatch.setattr(depth_mod, "_ORT_IMPORT_OK", True)
    monkeypatch.setattr(depth_mod, "ort", _FakeOrt(), raising=False)
    return str(weights)


def test_none_backend_is_explicit_noop():
    est = create_depth_estimator(DepthConfig(backend="none"))
    assert isinstance(est, DisabledDepth)
    assert est.is_available() is False
    with pytest.raises(RuntimeError, match="backend is 'none'"):
        est.estimate_depth(np.zeros((16, 16, 3), np.uint8))


def test_unidepth_estimate_meters(fake_ort):
    est = UniDepthEstimator(DepthConfig(model_path=fake_ort))
    assert est.is_available()
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    depth = est.estimate_depth(img)
    assert depth.shape == (48, 64)
    assert depth.dtype == np.float32
    # Fake session returns 1500 mm -> 1.5 m.
    assert np.median(depth) == pytest.approx(1.5)


def test_unidepth_missing_weights_names_download_url(tmp_path, caplog, monkeypatch):
    # ort importable but weights absent -> the weights branch must name the URL.
    monkeypatch.setattr(depth_mod, "_ORT_IMPORT_OK", True)
    est = UniDepthEstimator(
        DepthConfig(model_path=str(tmp_path / "missing.onnx"))
    )
    with caplog.at_level(logging.WARNING):
        assert est.is_available() is False
    assert "huggingface.co/ibaiGorordo/ONNX-UniDepth-V1" in caplog.text


def test_unidepth_estimate_raises_when_unavailable(tmp_path):
    est = UniDepthEstimator(DepthConfig(model_path=str(tmp_path / "missing.onnx")))
    with pytest.raises(RuntimeError, match="not available"):
        est.estimate_depth(np.zeros((16, 16, 3), np.uint8))


def test_backproject_uses_real_intrinsics():
    depth = np.full((100, 100), 2.0, dtype=np.float32)
    intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0)
    p = backproject_to_3d(depth, 60.0, 50.0, intr)
    np.testing.assert_allclose(p, [0.2, 0.0, 2.0], rtol=1e-6)


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown depth backend"):
        create_depth_estimator(DepthConfig(backend="zoedepth"))
