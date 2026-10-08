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
import random
import re
import threading
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
DEFAULT_MODEL = "gemini-2.5-flash"

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


def create_client(api_key: Optional[str] = None, timeout_s: float = 120.0):
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


# ---------------------------------------------------------------------------
# API resilience: pacing + retry on transient quota/overload errors
# ---------------------------------------------------------------------------
# The free Gemini tier caps Flash at ~10-15 requests/minute (and a few
# hundred/day). The pipeline issues several VLM calls per episode (segmentation,
# one label call per segment, language), so un-paced bursts trip 429s within
# seconds — and generate_text previously had ZERO retry, so a single 429 failed
# that segment's label outright. These helpers sit in generate_text, the single
# funnel every VLM stage goes through, so one fix covers all callers.

_RETRYABLE_CODES = {429, 500, 502, 503, 504}
_RETRYABLE_STATUSES = {"RESOURCE_EXHAUSTED", "UNAVAILABLE", "OVERLOADED"}
_SERVER_HINT_RE = re.compile(r"retry in ([0-9]+(?:\.[0-9]+)?)\s*s", re.IGNORECASE)


class _GeminiPacer:
    """Process-wide minimum spacing between generate_content calls.

    Thread-safe. The minimum interval defaults to 4s (safe under the free
    tier's ~15 RPM cap) and is overridden by the GEMINI_MIN_INTERVAL_S env
    var (seconds, float). Set GEMINI_MIN_INTERVAL_S=0 once billing is enabled
    (Tier 1 raises Flash to ~300 RPM) to remove the pacing delay.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_call_s = 0.0
        try:
            self.min_interval_s = float(os.environ.get("GEMINI_MIN_INTERVAL_S", "4.0"))
        except ValueError:
            self.min_interval_s = 4.0
        if self.min_interval_s < 0:
            self.min_interval_s = 0.0

    def wait(self) -> None:
        """Block until at least min_interval_s since the previous call."""
        if self.min_interval_s <= 0:
            return
        with self._lock:
            now = time.monotonic()
            delay = self.min_interval_s - (now - self._last_call_s)
            if delay > 0:
                time.sleep(delay)
            self._last_call_s = time.monotonic()


_PACER = _GeminiPacer()


def _max_attempts() -> int:
    """Total generate_content attempts per call (env GEMINI_MAX_ATTEMPTS)."""
    try:
        return max(1, int(os.environ.get("GEMINI_MAX_ATTEMPTS", "5")))
    except ValueError:
        return 5


def _is_retryable_error(exc: Exception) -> bool:
    """True for transient Gemini quota/overload errors worth retrying.

    Matches the google-genai APIError shape (``code`` int, ``status`` str),
    transport-level timeouts/connect errors, plus a message-substring
    fallback for SDKs that wrap differently. Permanent errors (400 bad
    request, 401/403 auth, 404 model) return False so they surface
    immediately instead of burning retries.
    """
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code in _RETRYABLE_CODES
    status = str(getattr(exc, "status", "") or "").upper()
    if status in _RETRYABLE_STATUSES:
        return True
    name = type(exc).__name__
    if "Timeout" in name or "ConnectError" in name:
        return True
    msg = str(exc)
    return any(tok in msg for tok in ("429", "503", "RESOURCE_EXHAUSTED",
                                      "UNAVAILABLE", "overloaded"))


def _retry_delay_s(exc: Exception, attempt: int) -> float:
    """Seconds to wait before the next attempt.

    Honors the server's "Please retry in Xs" hint when present (capped at
    120s); otherwise exponential backoff 2s, 4s, 8s... capped at 60s, with
    ±25% jitter to de-synchronise parallel callers.
    """
    m = _SERVER_HINT_RE.search(str(exc))
    if m:
        return min(120.0, float(m.group(1)) + 1.0)
    backoff = min(60.0, 2.0 * (2 ** attempt))
    return backoff * (1.0 + random.uniform(-0.25, 0.25))


def generate_text(client, model_name: str, contents: list, temperature: float = 0.1) -> str:
    """One generate_content call, with pacing + retry on transient errors.

    Calls are spaced by the shared pacer (GEMINI_MIN_INTERVAL_S, default 4s)
    so per-segment bursts stay under the free-tier RPM cap. Transient 429
    (RESOURCE_EXHAUSTED) / 503 (UNAVAILABLE) errors are retried up to
    GEMINI_MAX_ATTEMPTS times (default 5) with backoff that honors the
    server's retry hint; after the attempts are exhausted the original
    exception propagates, so callers' existing fallback behavior (e.g. the
    SmolVLM fallback / default segment labels) is unchanged.
    """
    attempts = _max_attempts()
    for attempt in range(attempts):
        _PACER.wait()
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(temperature=temperature),
            )
            return resp.text or ""
        except Exception as exc:  # noqa: BLE001 - re-raised unless retryable
            if attempt == attempts - 1 or not _is_retryable_error(exc):
                raise
            # A server retry hint of several minutes means the daily (RPD)
            # quota is exhausted — retrying inside this run cannot fix it.
            # Fail fast so the caller's fallback takes over immediately
            # instead of burning minutes of wall-clock per label call.
            hint = _SERVER_HINT_RE.search(str(exc))
            if hint and float(hint.group(1)) > 240.0:
                logger.warning(
                    "[gemini_client] Gemini quota exhausted (server asks to "
                    "retry in %ss — daily limit). Failing fast to fallback; "
                    "enable billing (Tier 1) to remove the daily cap.",
                    hint.group(1),
                )
                raise
            delay = _retry_delay_s(exc, attempt)
            logger.warning(
                "[gemini_client] Transient API error (%s: %s). "
                "Retrying in %.1fs (attempt %d/%d).",
                type(exc).__name__, exc, delay, attempt + 1, attempts,
            )
            time.sleep(delay)
    return ""  # unreachable: the loop either returns or raises


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
