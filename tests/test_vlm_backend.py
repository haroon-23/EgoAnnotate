"""Tests for src/vlm_backend.py — dataclasses, factories, backend protocol.

Heavy deps are faked throughout: no weights, no downloads, no transformers,
no google-genai needed. Follows the Phase-C fake pattern (cf. fake_owlvit).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.vlm_backend as vlm_mod
from src.vlm_backend import (
    SMOLVLM_URL,
    GeminiBackend,
    LocalVLMBackend,
    VLMBackendConfig,
    VLMRequest,
    VLMResponse,
    build_chat_messages,
    build_video_prompt,
    create_vlm_backend,
    parse_json_strict,
    resolve_stage_backend,
    sample_video_frames,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


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


class _FakeGenAIResponse:
    def __init__(self, text):
        self.text = text


class _FakeModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return _FakeGenAIResponse("canned answer")


class _FakeGenAIClient:
    def __init__(self):
        self.models = _FakeModels()


class _FakeProcessor:
    """Fake SmolVLM processor: records the chat messages, returns canned text."""

    def __init__(self):
        self.messages = None
        self.kwargs = None

    def apply_chat_template(self, messages, add_generation_prompt=True):
        self.messages = messages
        return "<chat>"

    def __call__(self, text, images, return_tensors):
        self.kwargs = {"text": text, "n_images": None if images is None else len(images)}
        return {"input_ids": torch.tensor([[1, 2]])}

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["local canned"]


class _FakeLocalModel:
    def __init__(self):
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return torch.tensor([[1, 2, 7, 8, 9]])


def _fake_ensure_model(self):
    self._processor = _FakeProcessor()
    self._model = _FakeLocalModel()


def _img():
    return Image.new("RGB", (64, 48), color=(10, 20, 30))


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


def test_request_response_defaults():
    req = VLMRequest(images=[_img()], prompt="hi")
    assert req.video_path is None
    assert req.max_new_tokens == 256
    assert req.temperature == pytest.approx(0.1)
    resp = VLMResponse(text="t", backend="gemini", latency_s=0.5)
    assert resp.backend == "gemini"


def test_backend_config_defaults():
    cfg = VLMBackendConfig()
    assert cfg.backend == "gemini"
    assert cfg.local_weights == "models/smolvlm-500m-instruct"
    assert cfg.device == "cpu"
    assert cfg.video_frames == 8


# ---------------------------------------------------------------------------
# create_vlm_backend — never raises, None when unavailable
# ---------------------------------------------------------------------------


def test_create_unknown_backend_returns_none():
    assert create_vlm_backend(VLMBackendConfig(backend="not_a_backend")) is None


def test_create_gemini_unavailable_without_sdk(monkeypatch):
    monkeypatch.setattr(vlm_mod, "GEMINI_AVAILABLE", False)
    monkeypatch.setattr(vlm_mod, "ensure_api_key_from_dotenv", lambda: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert create_vlm_backend(VLMBackendConfig(backend="gemini")) is None


def test_create_gemini_unavailable_without_key(monkeypatch):
    monkeypatch.setattr(vlm_mod, "GEMINI_AVAILABLE", True)
    monkeypatch.setattr(vlm_mod, "ensure_api_key_from_dotenv", lambda: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert create_vlm_backend(VLMBackendConfig(backend="gemini", gemini_api_key=None)) is None


def test_create_local_unavailable_missing_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm_mod, "_TRANSFORMERS_OK", True)
    cfg = VLMBackendConfig(backend="local", local_weights=str(tmp_path / "nope"))
    assert create_vlm_backend(cfg) is None


def test_create_never_raises_on_broken_config(monkeypatch):
    monkeypatch.setattr(vlm_mod, "ensure_api_key_from_dotenv", lambda: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # backend=None -> falls back to gemini -> unavailable here -> None, no raise
    assert create_vlm_backend(VLMBackendConfig(backend=None)) is None


# ---------------------------------------------------------------------------
# resolve_stage_backend — strict legacy contract
# ---------------------------------------------------------------------------


def test_resolve_unknown_backend_raises_valueerror():
    with pytest.raises(ValueError, match="Unknown VLM backend"):
        resolve_stage_backend("not_a_backend", "gemini-3.8-flash")


def test_resolve_gemini_no_sdk_raises_runtimeerror(monkeypatch):
    monkeypatch.setattr(vlm_mod, "GEMINI_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="google-genai not installed"):
        resolve_stage_backend(None, "gemini-3.8-flash")


def test_resolve_gemini_no_key_raises_valueerror(monkeypatch):
    monkeypatch.setattr(vlm_mod, "GEMINI_AVAILABLE", True)
    monkeypatch.setattr(vlm_mod, "ensure_api_key_from_dotenv", lambda: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable is not set"):
        resolve_stage_backend(None, "gemini-3.8-flash")


def test_resolve_local_missing_weights_raises_with_url(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm_mod, "_TRANSFORMERS_OK", True)
    cfg = VLMBackendConfig(backend="local", local_weights=str(tmp_path / "nope"))
    with pytest.raises(RuntimeError) as exc:
        resolve_stage_backend("local", "gemini-3.8-flash", base=cfg)
    assert SMOLVLM_URL in str(exc.value)


# ---------------------------------------------------------------------------
# GeminiBackend.generate — contents shapes
# ---------------------------------------------------------------------------


def _fake_generate_text_factory(calls):
    def fake_generate_text(client, model_name, contents, temperature=0.1, timeout_s=30.0):
        calls.append({"model": model_name, "contents": contents,
                      "temperature": temperature})
        return "canned answer"
    return fake_generate_text


def test_gemini_backend_image_contents(monkeypatch):
    monkeypatch.setattr(vlm_mod, "GEMINI_AVAILABLE", True)
    fake_client = _FakeGenAIClient()
    monkeypatch.setattr(vlm_mod, "create_client", lambda key, timeout_s=30.0: fake_client)
    monkeypatch.setattr(vlm_mod, "pil_to_part", lambda img: ("part", img))
    calls = []
    monkeypatch.setattr(vlm_mod, "generate_text", _fake_generate_text_factory(calls))

    backend = GeminiBackend(VLMBackendConfig(gemini_api_key="k"))
    assert backend.is_available()

    img = _img()
    resp = backend.generate(VLMRequest(images=[img], prompt="describe"))
    assert resp.backend == "gemini"
    assert resp.text == "canned answer"
    assert resp.latency_s >= 0.0
    call = calls[0]
    assert call["model"] == "gemini-3.8-flash"
    assert call["temperature"] == pytest.approx(0.1)
    # Prompt first, then image parts (legacy order).
    assert call["contents"][0] == "describe"
    assert call["contents"][1] == ("part", img)


def test_gemini_backend_video_contents_and_cleanup(monkeypatch):
    monkeypatch.setattr(vlm_mod, "GEMINI_AVAILABLE", True)
    fake_client = _FakeGenAIClient()
    monkeypatch.setattr(vlm_mod, "create_client", lambda key, timeout_s=30.0: fake_client)
    calls = []
    monkeypatch.setattr(vlm_mod, "generate_text", _fake_generate_text_factory(calls))
    remote = type("R", (), {"name": "files/abc"})()
    uploaded = {}
    deleted = []

    def fake_upload(client, path, timeout_s):
        uploaded["path"] = path
        return remote

    monkeypatch.setattr(vlm_mod, "upload_video_file", fake_upload)
    monkeypatch.setattr(vlm_mod, "delete_remote_file",
                        lambda client, name: deleted.append(name))

    backend = GeminiBackend(VLMBackendConfig(gemini_api_key="k"))
    resp = backend.generate(
        VLMRequest(video_path="vid.mp4", prompt="segment", video_upload_timeout_s=10.0)
    )
    assert resp.text == "canned answer"
    assert uploaded["path"] == "vid.mp4"
    call = calls[0]
    # Legacy order: [remote_file, prompt].
    assert call["contents"][0] is remote
    assert call["contents"][1] == "segment"
    assert deleted == ["files/abc"]


def test_gemini_backend_video_upload_failure_raises(monkeypatch):
    monkeypatch.setattr(vlm_mod, "GEMINI_AVAILABLE", True)
    monkeypatch.setattr(vlm_mod, "create_client", lambda key, timeout_s=30.0: _FakeGenAIClient())
    monkeypatch.setattr(vlm_mod, "upload_video_file", lambda client, path, timeout_s: None)

    backend = GeminiBackend(VLMBackendConfig(gemini_api_key="k"))
    with pytest.raises(RuntimeError, match="did not reach ACTIVE"):
        backend.generate(VLMRequest(video_path="vid.mp4", prompt="x"))


# ---------------------------------------------------------------------------
# LocalVLMBackend
# ---------------------------------------------------------------------------


def test_local_unavailable_without_transformers(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm_mod, "_TRANSFORMERS_OK", False)
    cfg = VLMBackendConfig(backend="local", local_weights=str(tmp_path))
    assert LocalVLMBackend(cfg).is_available() is False


def test_local_missing_weights_raises_with_url(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm_mod, "_TRANSFORMERS_OK", True)
    backend = LocalVLMBackend(
        VLMBackendConfig(backend="local", local_weights=str(tmp_path / "nope"))
    )
    with pytest.raises(RuntimeError) as exc:
        backend._ensure_model()
    assert SMOLVLM_URL in str(exc.value)
    assert "no auto-download" in str(exc.value)


def test_local_generate_with_fake_model(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm_mod, "_TRANSFORMERS_OK", True)
    monkeypatch.setattr(LocalVLMBackend, "_ensure_model", _fake_ensure_model)
    cfg = VLMBackendConfig(backend="local", local_weights=str(tmp_path))
    backend = LocalVLMBackend(cfg)
    assert backend.is_available()

    img = _img()
    resp = backend.generate(VLMRequest(images=[img], prompt="hi", temperature=0.0))
    assert resp.backend == "local"
    assert resp.text == "local canned"
    # Chat template: image parts first (temporal order), then text.
    content = backend._processor.messages[0]["content"]
    assert content[0] == {"type": "image"}
    assert content[1] == {"type": "text", "text": "hi"}
    # temperature=0 -> greedy
    assert backend._model.kwargs["do_sample"] is False


def test_build_chat_messages_order():
    msgs = build_chat_messages([_img(), _img()], "prompt text")
    content = msgs[0]["content"]
    assert [c["type"] for c in content] == ["image", "image", "text"]
    assert content[-1]["text"] == "prompt text"
    assert msgs[0]["role"] == "user"


def test_build_video_prompt_timestamps():
    frames = [(0.0, _img()), (1.5, _img()), (3.0, _img())]
    prompt = build_video_prompt("do the thing", frames)
    assert "3 frames sampled evenly" in prompt
    assert "Frame 1 of 3, t\u22480.0s." in prompt
    assert "Frame 2 of 3, t\u22481.5s." in prompt
    assert "Frame 3 of 3, t\u22483.0s." in prompt
    assert prompt.endswith("do the thing")
