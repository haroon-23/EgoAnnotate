"""Prompt shaping for the local VLM backend (Phase D): image order, timestamp
prefixes, chat-template layout. Model-free — only build_chat_messages and
build_video_prompt are exercised (plus the wire format the local backend
hands to the chat template via a fake processor).
"""
import os
import sys

import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vlm_backend import (
    LocalVLMBackend,
    VLMBackendConfig,
    VLMRequest,
    build_chat_messages,
    build_video_prompt,
)


def _img(color=(10, 20, 30)):
    return Image.new("RGB", (64, 48), color=color)


class _FakeProcessor:
    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, add_generation_prompt=True):
        self.messages = messages
        return "<chat>"

    def __call__(self, text, images, return_tensors):
        return {"input_ids": torch.tensor([[1, 2]])}

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["ok"]


class _FakeLocalModel:
    def generate(self, **kwargs):
        return torch.tensor([[1, 2, 3]])


def _local_backend(monkeypatch, tmp_path):
    from src import vlm_backend as vlm_mod
    monkeypatch.setattr(vlm_mod, "_TRANSFORMERS_OK", True)

    def fake_ensure(self):
        self._processor = _FakeProcessor()
        self._model = _FakeLocalModel()

    monkeypatch.setattr(LocalVLMBackend, "_ensure_model", fake_ensure)
    return LocalVLMBackend(VLMBackendConfig(backend="local", local_weights=str(tmp_path)))


def test_chat_messages_image_order_preserved():
    imgs = [_img((1, 2, 3)), _img((4, 5, 6))]
    msgs = build_chat_messages(imgs, "do it")
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    content = msgs[0]["content"]
    assert [c["type"] for c in content] == ["image", "image", "text"]
    assert content[-1] == {"type": "text", "text": "do it"}


def test_chat_messages_with_timestamp_prefixes():
    frames = [(0.0, _img()), (1.5, _img())]
    prompt = build_video_prompt("base prompt", frames)
    msgs = build_chat_messages([f[1] for f in frames], prompt)
    text = msgs[0]["content"][-1]["text"]
    assert "2 frames sampled evenly" in text
    assert "Frame 1 of 2, t\u22480.0s." in text
    assert "Frame 2 of 2, t\u22481.5s." in text
    # Image order in the message matches the sampled frame order.
    assert len(msgs[0]["content"]) == 3
    assert text.endswith("base prompt")


def test_local_backend_wire_format_images(monkeypatch, tmp_path):
    """The local backend hands the processor images in request order,
    then the text — the processor fake records the chat messages."""
    backend = _local_backend(monkeypatch, tmp_path)
    imgs = [_img((1, 2, 3)), _img((4, 5, 6))]
    backend.generate(VLMRequest(images=imgs, prompt="wire"))
    content = backend._processor.messages[0]["content"]
    assert [c["type"] for c in content] == ["image", "image", "text"]
    assert content[-1]["text"] == "wire"


def test_build_video_prompt_single_frame():
    prompt = build_video_prompt("p", [(1.5, _img())])
    assert "1 frame sampled evenly" in prompt
    assert "Frame 1 of 1, t\u22481.5s." in prompt
