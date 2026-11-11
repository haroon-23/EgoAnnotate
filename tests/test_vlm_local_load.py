"""Local transformers backend must load without the optional ``accelerate``."""
import builtins
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vlm_backend import LocalVLMBackend, VLMBackendConfig


class _FakeModel:
    def __init__(self):
        self.moved_to = None
        self.evaluated = False

    def to(self, device):
        self.moved_to = device
        return self

    def eval(self):
        self.evaluated = True


def _install_fake_transformers(monkeypatch, captured):
    mod = types.ModuleType("transformers")

    class Proc:
        @staticmethod
        def from_pretrained(*a, **k):
            return object()

    class Auto:
        @staticmethod
        def from_pretrained(path, **kwargs):
            captured.update(kwargs)
            return _FakeModel()

    mod.AutoProcessor = Proc
    mod.AutoModelForImageTextToText = Auto
    monkeypatch.setitem(sys.modules, "transformers", mod)


def _block_accelerate(monkeypatch):
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "accelerate":
            raise ImportError("No module named 'accelerate'")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)


def test_cpu_load_needs_no_accelerate(tmp_path, monkeypatch):
    captured = {}
    _install_fake_transformers(monkeypatch, captured)
    _block_accelerate(monkeypatch)
    b = LocalVLMBackend(VLMBackendConfig(backend="local", local_weights=str(tmp_path), device="cpu"))
    b._ensure_model()
    assert "device_map" not in captured
    assert b._model.moved_to == "cpu" and b._model.evaluated


def test_non_cpu_without_accelerate_names_the_package(tmp_path, monkeypatch):
    _install_fake_transformers(monkeypatch, {})
    _block_accelerate(monkeypatch)
    b = LocalVLMBackend(VLMBackendConfig(backend="local", local_weights=str(tmp_path), device="cuda"))
    with pytest.raises(RuntimeError, match="accelerate"):
        b._ensure_model()
