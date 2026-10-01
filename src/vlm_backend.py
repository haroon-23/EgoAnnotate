"""Dual VLM backend: Gemini API (production) + local SmolVLM (offline/dev).

Phase D. Mirrors the Phase-C ``Detector2D`` pattern:

  - :class:`VLMRequest` / :class:`VLMResponse` dataclasses
  - :class:`VLMBackend` protocol (``name``, ``is_available()``, ``generate()``)
  - :class:`GeminiBackend`: thin wrapper over :mod:`src.gemini_client`
    (all google-genai SDK logic lives there — untouched)
  - :class:`LocalVLMBackend`: SmolVLM-500M-Instruct via transformers,
    CPU-only, lazy model load
  - :func:`create_vlm_backend`: returns a working backend or ``None``;
    never raises

Plus the shared output-contract layer (:func:`parse_json_strict` with a
one-shot repair re-prompt hook) and the video frame sampler used by the
video stages instead of Gemini's Files-API upload.

Rules (same as Phase C):

  - No weight auto-download anywhere. Missing weights -> loud ``RuntimeError``
    naming the manual URL: https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct
  - Import-safe: ``import src.vlm_backend`` never raises when google-genai /
    transformers / torch are absent (all heavy imports are lazy/guarded).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, List, Optional, Protocol, Tuple

import cv2
import numpy as np
from PIL import Image

from .gemini_client import (
    GEMINI_AVAILABLE,
    create_client,
    delete_remote_file,
    generate_text,
    pil_to_part,
    resolve_model_name,
    upload_video_file,
)

logger = logging.getLogger(__name__)

#: Manual download for the local backend (no auto-download by design).
SMOLVLM_URL = "https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct"
SMOLVLM_MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"

try:  # transformers is optional; the module must import without it.
    import transformers  # noqa: F401

    _TRANSFORMERS_OK = True
except ImportError:
    _TRANSFORMERS_OK = False


# ---------------------------------------------------------------------------
# Config / request / response
# ---------------------------------------------------------------------------


@dataclass
class VLMBackendConfig:
    """Configuration for both VLM backends (mirrors ``configs/default.yaml``)."""

    backend: str = "gemini"  # "gemini" | "local"
    gemini_model: str = "gemini-3.8-flash"
    gemini_api_key: Optional[str] = None  # None -> GEMINI_API_KEY env var
    local_model: str = SMOLVLM_MODEL_ID
    local_weights: str = "models/smolvlm-500m-instruct"  # manual download, never auto
    device: str = "cpu"
    max_new_tokens: int = 256
    temperature: float = 0.1
    video_frames: int = 8  # stills sampled to replace Gemini video upload
    timeout_s: float = 300.0  # local CPU inference is slow; fail loud, don't hang
    on_unavailable: str = "degrade"  # "degrade" = use per-stage fallbacks


@dataclass
class VLMRequest:
    """One VLM call. Images are PIL, temporal order; video_path for video stages."""

    images: List[Image.Image] = field(default_factory=list)
    prompt: str = ""
    video_path: Optional[str] = None
    max_new_tokens: int = 256
    temperature: float = 0.1
    # Gemini-only hint: how long upload_video_file may poll for ACTIVE.
    video_upload_timeout_s: float = 30.0


@dataclass
class VLMResponse:
    """Raw model text; parsing stays in the calling stage."""

    text: str
    backend: str  # "gemini" | "local"
    latency_s: float


class VLMBackend(Protocol):
    """Structural interface every VLM backend implements."""

    name: str

    def is_available(self) -> bool: ...
    def generate(self, request: VLMRequest) -> VLMResponse: ...


# ---------------------------------------------------------------------------
# .env bootstrap (legacy behavior, centralized)
# ---------------------------------------------------------------------------


def ensure_api_key_from_dotenv() -> None:
    """Best-effort ``GEMINI_API_KEY`` load from ``.env`` (legacy stage behavior).

    The stage modules used to do this scan at import time; it now lives here
    so every backend path shares it. Never raises.
    """
    if os.environ.get("GEMINI_API_KEY"):
        return
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / ".env",
        Path(".env"),
        Path.home() / "sia_agent" / ".env",
    ]
    try:
        for env_path in candidates:
            if env_path.exists():
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GEMINI_API_KEY="):
                            val = line.split("=", 1)[1].strip("'\"")
                            if val:
                                os.environ["GEMINI_API_KEY"] = val
                                return
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Gemini backend (thin wrapper — SDK logic stays in gemini_client.py)
# ---------------------------------------------------------------------------


class GeminiBackend:
    """Gemini API backend. Wraps :mod:`src.gemini_client`; adds no SDK logic."""

    name = "gemini"

    def __init__(self, config: VLMBackendConfig):
        self.config = config
        self._client = None
        self._model_name = resolve_model_name(config.gemini_model)

    @property
    def model_name(self) -> str:
        return self._model_name

    def is_available(self) -> bool:
        if not GEMINI_AVAILABLE:
            return False
        key = self.config.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        return bool(key)

    def _ensure_client(self):
        if self._client is None:
            key = self.config.gemini_api_key or os.environ.get("GEMINI_API_KEY")
            self._client = create_client(key, timeout_s=30.0)
        return self._client

    def generate(self, request: VLMRequest) -> VLMResponse:
        t0 = time.time()
        # Raises RuntimeError/ValueError exactly like the old stage _init_model
        # did when the SDK/key was missing.
        client = self._ensure_client()
        if request.video_path:
            remote = upload_video_file(
                client, str(request.video_path),
                timeout_s=request.video_upload_timeout_s,
            )
            if remote is None:
                raise RuntimeError(
                    f"Gemini video upload did not reach ACTIVE: {request.video_path}"
                )
            try:
                text = generate_text(
                    client, self._model_name, [remote, request.prompt],
                    temperature=request.temperature,
                )
            finally:
                delete_remote_file(client, remote.name)
        else:
            contents = [request.prompt] + [pil_to_part(img) for img in request.images]
            text = generate_text(
                client, self._model_name, contents, temperature=request.temperature
            )
        return VLMResponse(text=text or "", backend="gemini", latency_s=time.time() - t0)


# ---------------------------------------------------------------------------
# Local backend (SmolVLM-500M-Instruct via transformers, CPU-only)
# ---------------------------------------------------------------------------


def build_chat_messages(
    images: List[Image.Image], prompt_text: str
) -> List[dict]:
    """Build a SmolVLM/Idefics3 chat message list (testable without a model).

    Image parts come first, in temporal order, then the text part.
    """
    content: List[dict] = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": prompt_text})
    return [{"role": "user", "content": content}]


def build_video_prompt(base_prompt: str, frames: List[Tuple[float, Image.Image]]) -> str:
    """Prefix a prompt with per-frame timestamp lines for sampled stills."""
    n = len(frames)
    word = "frame" if n == 1 else "frames"
    lines = [
        f"These are {n} {word} sampled evenly from a video, in temporal order."
    ]
    for i, (t, _) in enumerate(frames):
        lines.append(f"Frame {i + 1} of {n}, t\u2248{t:.1f}s.")
    return "\n".join(lines) + "\n\n" + base_prompt


def sample_video_frames(
    video_path: str, k: int = 8, max_size: int = 512
) -> List[Tuple[float, Image.Image]]:
    """Sample K evenly-spaced frames.

    Returns ``[(t_seconds, PIL.Image)]`` in temporal order, each frame
    downscaled so its longest side is ``<= max_size``. Deterministic for a
    given video and K. Raises ``ValueError`` when the video cannot be read.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if n <= 0:
            raise ValueError(f"Video has no frames: {video_path}")
        k = max(1, min(int(k), n))
        idxs = np.linspace(0, n - 1, k, dtype=int)
        out: List[Tuple[float, Image.Image]] = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size))
            out.append((float(int(idx)) / float(fps), img))
        if not out:
            raise ValueError(f"Could not read any frames from: {video_path}")
        return out
    finally:
        cap.release()


class LocalVLMBackend:
    """Local SmolVLM-500M-Instruct backend (transformers, CPU-only).

    The model loads lazily on the first :meth:`generate` call and is then
    cached. Weights are NEVER downloaded automatically — a missing weights
    directory raises a loud ``RuntimeError`` naming the manual URL.
    """

    name = "local"

    def __init__(self, config: VLMBackendConfig):
        self.config = config
        self._processor = None
        self._model = None
        self._load_error: Optional[BaseException] = None

    def is_available(self) -> bool:
        return _TRANSFORMERS_OK and Path(self.config.local_weights).is_dir()

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if self._load_error is not None:
            raise self._load_error
        weights = Path(self.config.local_weights)
        if not weights.is_dir():
            err = RuntimeError(
                f"Local VLM weights not found at '{weights}'. Download manually from "
                f"{SMOLVLM_URL} into '{weights}/' (no auto-download by design)."
            )
            self._load_error = err
            raise err
        try:
            try:
                from transformers import AutoModelForImageTextToText, AutoProcessor

                model_cls = AutoModelForImageTextToText
            except ImportError:  # older transformers (<4.46-ish)
                from transformers import AutoModelForVision2Seq, AutoProcessor

                model_cls = AutoModelForVision2Seq
            import torch

            self._processor = AutoProcessor.from_pretrained(
                str(weights), trust_remote_code=False, local_files_only=True
            )
            self._model = model_cls.from_pretrained(
                str(weights),
                torch_dtype=torch.float32,
                device_map=self.config.device or "cpu",
                trust_remote_code=False,
                local_files_only=True,  # never reach the network: manual weights only
            )
            self._model.eval()
        except Exception as e:
            err = RuntimeError(f"Failed to load local VLM from '{weights}': {e}")
            self._load_error = err
            raise err from e

    def generate(self, request: VLMRequest) -> VLMResponse:
        t0 = time.time()
        self._ensure_model()
        images = list(request.images)
        prompt_text = request.prompt
        if request.video_path:
            frames = sample_video_frames(str(request.video_path), k=self.config.video_frames)
            images = [img for _, img in frames]
            prompt_text = build_video_prompt(request.prompt, frames)
        text = self._generate_multi_image(images, prompt_text, request)
        return VLMResponse(text=text, backend="local", latency_s=time.time() - t0)

    def _generate_multi_image(
        self, images: List[Image.Image], prompt_text: str, request: VLMRequest
    ) -> str:
        messages = build_chat_messages(images, prompt_text)
        chat_text = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(
            text=chat_text, images=images if images else None, return_tensors="pt"
        )
        gen_kwargs: dict = {
            "max_new_tokens": request.max_new_tokens or self.config.max_new_tokens
        }
        if request.temperature and request.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = request.temperature
        else:
            gen_kwargs["do_sample"] = False
        # Fail loud on timeout instead of hanging the pipeline on old CPUs.
        # The worker thread cannot be killed; it is abandoned, not awaited.
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            fut = executor.submit(self._model.generate, **inputs, **gen_kwargs)
            try:
                out = fut.result(timeout=self.config.timeout_s)
            except FuturesTimeoutError:
                raise TimeoutError(
                    f"Local VLM inference timed out after {self.config.timeout_s}s "
                    "(old Intel CPUs are slow; raise vlm.timeout_s, lower "
                    "vlm.video_frames, or use the gemini backend)."
                )
        finally:
            executor.shutdown(wait=False)
        input_len = inputs["input_ids"].shape[-1]
        text = self._processor.batch_decode(
            out[:, input_len:], skip_special_tokens=True
        )[0]
        return text.strip()


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def create_vlm_backend(config: Optional[VLMBackendConfig] = None) -> Optional[VLMBackend]:
    """Build the configured backend. Returns ``None`` when unavailable.

    Never raises — use :func:`resolve_stage_backend` when the legacy strict
    (raise-on-missing) constructor contract is wanted.
    """
    cfg = config or VLMBackendConfig()
    try:
        ensure_api_key_from_dotenv()
        name = (cfg.backend or "gemini").strip().lower()
        if name == "gemini":
            backend: VLMBackend = GeminiBackend(cfg)
        elif name == "local":
            backend = LocalVLMBackend(cfg)
        else:
            logger.warning(
                "[vlm_backend] Unknown backend %r (want 'gemini'|'local')", cfg.backend
            )
            return None
        if backend.is_available():
            logger.debug("[vlm_backend] Using backend=%s", backend.name)
            return backend
        if name == "local":
            logger.warning(
                "[vlm_backend] Local backend unavailable "
                "(transformers_ok=%s, weights dir '%s' present=%s). Download from %s.",
                _TRANSFORMERS_OK,
                cfg.local_weights,
                Path(cfg.local_weights).is_dir(),
                SMOLVLM_URL,
            )
        else:
            logger.warning(
                "[vlm_backend] Gemini backend unavailable "
                "(google-genai missing or GEMINI_API_KEY unset)."
            )
        return None
    except Exception as e:  # never raise out of the factory
        logger.warning("[vlm_backend] create_vlm_backend failed: %s", e)
        return None


def resolve_stage_backend(
    stage_override: Optional[str],
    gemini_model: str = "gemini-3.8-flash",
    base: Optional[VLMBackendConfig] = None,
) -> VLMBackend:
    """Strict constructor helper: return a working backend or raise.

    Preserves the legacy stage ``_init_model`` contract:

      - unknown backend name -> ``ValueError``
      - ``gemini`` requested but SDK missing -> ``RuntimeError("google-genai not installed.")``
      - ``gemini`` requested but no API key -> ``ValueError("GEMINI_API_KEY environment variable is not set...")``
      - ``local`` requested but transformers/weights missing -> ``RuntimeError``
        naming the manual download URL
    """
    cfg = (
        replace(base, gemini_model=gemini_model)
        if base is not None
        else VLMBackendConfig(gemini_model=gemini_model)
    )
    name = (stage_override or cfg.backend or "gemini").strip().lower()
    cfg = replace(cfg, backend=name)
    backend = create_vlm_backend(cfg)
    if backend is not None:
        return backend
    if name == "local":
        raise RuntimeError(
            "Local VLM backend requested but unavailable "
            f"(transformers_ok={_TRANSFORMERS_OK}, weights dir "
            f"'{cfg.local_weights}' present={Path(cfg.local_weights).is_dir()}). "
            f"Download weights manually from {SMOLVLM_URL} — no auto-download by design."
        )
    if name != "gemini":
        raise ValueError(f"Unknown VLM backend {stage_override!r} (want 'gemini'|'local').")
    if not GEMINI_AVAILABLE:
        raise RuntimeError("google-genai not installed.")
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set. "
        "Set it (or vlm.gemini_api_key) to use the Gemini backend, "
        "or switch to vlm.backend='local' with downloaded SmolVLM weights."
    )


# ---------------------------------------------------------------------------
# Output-contract layer: strict JSON parse + one repair re-prompt
# ---------------------------------------------------------------------------

_MISSING = object()


def _json_candidates(text: str):
    """Yield JSON-parseable candidate substrings, most-likely first."""
    t = text.strip()
    yield t
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", t, re.DOTALL)
    if m:
        yield m.group(1).strip()
    opens = [i for i, ch in enumerate(t) if ch in "[{"]
    if opens:
        first = opens[0]
        closer = "]" if t[first] == "[" else "}"
        last = t.rfind(closer)
        if last > first:
            yield t[first : last + 1]


def _try_parse_json(text: str):
    for cand in _json_candidates(text):
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
    return _MISSING


def build_repair_prompt(original_text: str, expect: str = "JSON") -> str:
    """One-shot repair prompt: ask for ONLY the JSON, no fences, no prose."""
    return (
        "Your last reply was not valid JSON. Reply with ONLY the JSON "
        f"{expect} \u2014 no markdown fences, no commentary, no extra text.\n\n"
        f"Previous reply:\n{original_text.strip()[:2000]}"
    )


def parse_json_strict(
    text: str,
    repair_fn: Optional[Callable[[str], str]] = None,
    expect: str = "JSON",
):
    """Strict JSON extraction with an optional one-shot repair re-prompt.

    Args:
        text: raw model output.
        repair_fn: optional callable taking the repair prompt and returning
            the model's second reply (one attempt only).
        expect: human description of the expected shape, e.g. "JSON list".

    Returns the parsed object, or ``None`` when unparseable. Never raises.
    """
    if not text:
        return None
    parsed = _try_parse_json(text)
    if parsed is not _MISSING:
        return parsed
    if repair_fn is not None:
        try:
            repaired = repair_fn(build_repair_prompt(text, expect=expect))
        except Exception:
            logger.debug("[vlm_backend] repair re-prompt failed", exc_info=True)
            return None
        if repaired:
            parsed = _try_parse_json(repaired)
            if parsed is not _MISSING:
                return parsed
    return None
