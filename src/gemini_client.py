"""Shared Gemini client helpers (google-genai SDK).

The legacy ``google-generativeai`` package is deprecated and its 1.5-era
model names now return HTTP 404. Every VLM stage (object_detector,
action_segmenter, segment_labeler, language_generator) builds its client
through this module so the SDK migration lives in exactly one place.

Verified against google-genai 2.25.0:
  - client = genai.Client(api_key=..., http_options=types.HttpOptions(timeout_ms))
  - client.models.generate_content(model=..., contents=..., config=types.GenerateContentConfig(temperature=...))
  - client.files.upload(file=...) / .get(name=...) / .delete(name=...)
  - types.Part.from_bytes(data=..., mime_type=...) for images
  - File.state is a FileState enum (PROCESSING / ACTIVE / FAILED)
"""
from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types

    GEMINI_AVAILABLE = True
except ImportError:  # pragma: no cover - SDK simply not installed
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]
    GEMINI_AVAILABLE = False

#: Current flash-tier model. 1.5-era names 404 as of 2026.
DEFAULT_MODEL = "gemini-3.8-flash"

#: Model names from the google-generativeai era. All retired; remapped below.
_LEGACY_MODEL_ALIASES = {
    "gemini-1.5-flash",
    "gemini-flash-latest",
    "gemini-1.5-pro-latest",
    "gemini-flash-lite-latest",
    "models/gemini-1.5-flash",
    "models/gemini-flash-latest",
    "models/gemini-1.5-pro-latest",
    "models/gemini-flash-lite-latest",
}


def resolve_model_name(name: Optional[str]) -> str:
    """Return a usable model name for the google-genai SDK.

    Retired/empty names map to DEFAULT_MODEL; the new SDK takes bare model
    IDs (no ``models/`` prefix).
    """
    name = (name or "").strip()
    if not name or name in _LEGACY_MODEL_ALIASES:
        if name:
            logger.info("[gemini_client] Remapping retired model %r -> %r", name, DEFAULT_MODEL)
        return DEFAULT_MODEL
    return name[7:] if name.startswith("models/") else name


def create_client(api_key: Optional[str] = None, timeout_s: float = 30.0):
    """Create a google-genai Client.

    Raises:
        RuntimeError: if the google-genai package is not installed.
        ValueError: if no API key is available.
    """
    if not GEMINI_AVAILABLE:
        raise RuntimeError("google-genai not installed. Install with: pip install google-genai")
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    http_options = types.HttpOptions(timeout=int(timeout_s * 1000))  # ms
    return genai.Client(api_key=key, http_options=http_options)


def pil_to_part(pil_image):
    """Encode a PIL image as a JPEG-bytes Part for the new SDK.

    The old SDK accepted PIL images directly in ``contents``; the new one
    requires explicit bytes, so we convert once here.
    """
    buf = io.BytesIO()
    img = pil_image.convert("RGB") if pil_image.mode != "RGB" else pil_image
    img.save(buf, format="JPEG")
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


def generate_text(client, model_name: str, contents: list, temperature: float = 0.1) -> str:
    """One generate_content call. Returns ``response.text`` (``""`` if empty)."""
    resp = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(temperature=temperature),
    )
    return resp.text or ""


def _file_state_name(remote_file) -> str:
    state = getattr(remote_file, "state", None)
    return str(getattr(state, "name", state) or "").upper()


def upload_video_file(client, video_path: Union[str, Path], timeout_s: float = 60.0):
    """Upload a video and wait until it is ACTIVE. Returns the File or None."""
    remote = client.files.upload(file=str(video_path))
    waited = 0.0
    while _file_state_name(remote) == "PROCESSING" and waited < timeout_s:
        time.sleep(2.0)
        waited += 2.0
        try:
            remote = client.files.get(name=remote.name)
        except Exception:
            pass
    if _file_state_name(remote) != "ACTIVE":
        logger.warning(
            "[gemini_client] Video upload did not reach ACTIVE (state=%s)",
            _file_state_name(remote),
        )
        return None
    return remote


def delete_remote_file(client, name: str) -> None:
    """Best-effort deletion of an uploaded remote file."""
    try:
        client.files.delete(name=name)
    except Exception:
        pass
