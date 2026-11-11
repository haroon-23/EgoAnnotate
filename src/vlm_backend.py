"""VLM backends: Gemini, OpenAI-compatible HTTPS, local SmolVLM, or none.

  - :class:`VLMRequest` / :class:`VLMResponse` dataclasses
  - :class:`VLMBackend` protocol (``name``, ``is_available()``, ``generate()``)
  - :class:`GeminiBackend`: thin wrapper over :mod:`src.gemini_client`
  - :class:`OpenAICompatibleBackend`: plain chat-completions over HTTPS; one
    backend for NVIDIA's hosted API (integrate.api.nvidia.com), a self-hosted
    vLLM/NIM server, or Ollama
  - :class:`LocalVLMBackend`: SmolVLM via transformers, CPU, lazy load
  - :class:`NullVLMBackend`: the ``none`` choice (template language only)
  - :class:`VLMBudget` + :class:`ManagedBackend`: the single place that
    enforces the per-episode request budget (retries included) and the
    on-disk response cache
  - :func:`select_backend` / :func:`create_managed_backend`: the factory for
    ``vlm.backend: auto | gemini | openai_compatible | local | none``
  - :func:`create_vlm_backend`: legacy factory, returns a backend or ``None``

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

import base64
import hashlib
import io
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

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
    """Configuration for all VLM backends (mirrors the ``vlm:`` block of ``configs/default.yaml``)."""

    # "auto" | "gemini" | "openai_compatible" | "local" | "none"
    backend: str = "gemini"
    gemini_model: str = "gemini-2.5-flash"
    gemini_api_key: Optional[str] = None  # None -> GEMINI_API_KEY env var
    local_model: str = SMOLVLM_MODEL_ID
    local_weights: str = "models/smolvlm-500m-instruct"  # manual download, never auto
    device: str = "cpu"
    max_new_tokens: int = 256
    temperature: float = 0.1
    video_frames: int = 8  # stills sampled to replace Gemini video upload
    timeout_s: float = 300.0  # local CPU inference is slow; fail loud, don't hang
    on_unavailable: str = "degrade"  # "degrade" = use per-stage fallbacks
    # --- openai_compatible (NVIDIA hosted API, vLLM/NIM, Ollama) ---
    base_url: str = ""  # e.g. https://integrate.api.nvidia.com/v1 or http://localhost:8000/v1
    model: str = ""  # model id sent to the endpoint
    api_key_env: str = "NVIDIA_API_KEY"  # env var NAME holding the key; "" = no auth
    request_timeout_s: float = 60.0
    max_retries: int = 2  # extra attempts on 429/5xx/network errors; each counts against the budget
    json_mode: bool = False  # send response_format={"type":"json_object"} (not all servers accept it)
    # --- cost control ---
    max_requests_per_episode: int = 3
    keyframes_per_request: int = 5
    keyframe_max_side: int = 448
    cache_enabled: bool = True
    cache_path: str = "cache/vlm_cache.json"

    @classmethod
    def from_config(cls, config: Optional[dict]) -> "VLMBackendConfig":
        """Build from a full pipeline config dict (reads ``vlm`` and ``gemini``).

        Unlike the bare dataclass (legacy default ``gemini``), the config
        default is ``auto`` so a machine without a key still runs.
        """
        config = config or {}
        v = config.get("vlm") or {}
        g = config.get("gemini") or {}
        d = cls()

        def pick(key: str):
            val = v.get(key)
            default = getattr(d, key)
            return default if val is None else type(default)(val)

        return cls(
            backend=str(v.get("backend") or "auto"),
            gemini_model=str(v.get("gemini_model") or g.get("model") or d.gemini_model),
            gemini_api_key=v.get("gemini_api_key") or g.get("api_key"),
            local_model=pick("local_model"),
            local_weights=pick("local_weights"),
            device=pick("device"),
            max_new_tokens=pick("max_new_tokens"),
            temperature=pick("temperature"),
            video_frames=pick("video_frames"),
            timeout_s=pick("timeout_s"),
            on_unavailable=pick("on_unavailable"),
            base_url=pick("base_url"),
            model=pick("model"),
            api_key_env=pick("api_key_env"),
            request_timeout_s=pick("request_timeout_s"),
            max_retries=pick("max_retries"),
            json_mode=bool(v.get("json_mode", d.json_mode)),
            max_requests_per_episode=pick("max_requests_per_episode"),
            keyframes_per_request=pick("keyframes_per_request"),
            keyframe_max_side=pick("keyframe_max_side"),
            cache_enabled=bool(v.get("cache_enabled", d.cache_enabled)),
            cache_path=pick("cache_path"),
        )


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
    # Called before every network attempt (retries included); raises to veto.
    # Set by ManagedBackend; excluded from the cache key.
    on_attempt: Optional[Callable[[], None]] = None


@dataclass
class VLMResponse:
    """Raw model text; parsing stays in the calling stage."""

    text: str
    backend: str  # "gemini" | "openai_compatible" | "local"
    latency_s: float
    cached: bool = False
    model: str = ""


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

    @property
    def model_id(self) -> str:
        return self._model_name

    def is_available(self) -> bool:
        if not GEMINI_AVAILABLE:
            return False
        key = self.config.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        return bool(key)

    def _ensure_client(self):
        if self._client is None:
            key = self.config.gemini_api_key or os.environ.get("GEMINI_API_KEY")
            self._client = create_client(key, timeout_s=120.0)
        return self._client

    def generate(self, request: VLMRequest) -> VLMResponse:
        # Response caching and the request budget live in ManagedBackend.
        t0 = time.time()
        gen_kwargs: Dict[str, Any] = {"temperature": request.temperature}
        if request.on_attempt is not None:
            gen_kwargs["on_attempt"] = request.on_attempt
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
                    client, self._model_name, [remote, request.prompt], **gen_kwargs
                )
            finally:
                delete_remote_file(client, remote.name)
        else:
            contents = [request.prompt] + [pil_to_part(img) for img in request.images]
            text = generate_text(client, self._model_name, contents, **gen_kwargs)
        return VLMResponse(
            text=text or "", backend="gemini", latency_s=time.time() - t0,
            model=self._model_name,
        )


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

    @property
    def model_id(self) -> str:
        return self.config.local_model

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
            device = (self.config.device or "cpu").strip()
            load_kwargs: Dict[str, Any] = dict(
                torch_dtype=torch.float32,
                trust_remote_code=False,
                local_files_only=True,  # never reach the network: manual weights only
            )
            # ``device_map`` requires the optional ``accelerate`` package; the
            # default CPU path loads plainly and moves the module afterwards.
            if device != "cpu":
                try:
                    import accelerate  # noqa: F401
                except ImportError as ie:
                    raise RuntimeError(
                        f"vlm.device={device!r} needs the 'accelerate' package "
                        "(pip install accelerate); use device: cpu to run without it."
                    ) from ie
                load_kwargs["device_map"] = device
            self._model = model_cls.from_pretrained(str(weights), **load_kwargs)
            if device == "cpu":
                self._model = self._model.to("cpu")
            self._model.eval()
        except Exception as e:
            err = RuntimeError(f"Failed to load local VLM from '{weights}': {e}")
            self._load_error = err
            raise err from e

    def generate(self, request: VLMRequest) -> VLMResponse:
        t0 = time.time()
        if request.on_attempt is not None:
            request.on_attempt()
        self._ensure_model()
        images = list(request.images)
        prompt_text = request.prompt
        if request.video_path:
            frames = sample_video_frames(str(request.video_path), k=self.config.video_frames)
            images = [img for _, img in frames]
            prompt_text = build_video_prompt(request.prompt, frames)
        text = self._generate_multi_image(images, prompt_text, request)
        return VLMResponse(
            text=text, backend="local", latency_s=time.time() - t0,
            model=self.config.local_model,
        )

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
# OpenAI-compatible backend (NVIDIA hosted API, vLLM/NIM, Ollama)
# ---------------------------------------------------------------------------

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
_RETRY_HTTP = {408, 429, 500, 502, 503, 504}
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Drop ``<think>`` blocks and unwrap ``<answer>`` (reasoning VLMs such as Cosmos-Reason)."""
    if not text:
        return ""
    t = _THINK_RE.sub("", text)
    if "</think>" in t:  # opening tag stripped by the chat template
        t = t.rsplit("</think>", 1)[1]
    m = _ANSWER_RE.search(t)
    if m:
        t = m.group(1)
    return t.strip()


def image_to_data_url(img: Image.Image, max_side: int = 1024, quality: int = 85) -> str:
    img = img.convert("RGB") if img.mode != "RGB" else img
    if max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class OpenAICompatibleBackend:
    """Plain HTTPS ``/chat/completions`` with image content parts.

    No SDK dependency. Works with NVIDIA's hosted API
    (``https://integrate.api.nvidia.com/v1``), a self-hosted vLLM / NIM
    server, or Ollama (``http://localhost:11434/v1``). Retries only on
    429/5xx/network errors, at most ``max_retries`` extra times; every attempt
    goes through ``request.on_attempt`` so the budget counts retries.
    """

    name = "openai_compatible"

    def __init__(self, config: VLMBackendConfig, sleep: Callable[[float], None] = time.sleep):
        self.config = config
        self._sleep = sleep

    @property
    def model_id(self) -> str:
        return self.config.model

    @property
    def endpoint(self) -> str:
        return self.config.base_url.rstrip("/") + "/chat/completions"

    def _api_key(self) -> Optional[str]:
        env = (self.config.api_key_env or "").strip()
        return (os.environ.get(env) or None) if env else None

    def _is_loopback(self) -> bool:
        host = urllib.parse.urlparse(self.config.base_url).hostname or ""
        return host in _LOOPBACK_HOSTS

    def is_available(self) -> bool:
        if not (self.config.base_url and self.config.model):
            return False
        needs_key = bool((self.config.api_key_env or "").strip()) and not self._is_loopback()
        return (not needs_key) or self._api_key() is not None

    def build_payload(self, request: VLMRequest) -> dict:
        images = list(request.images)
        prompt = request.prompt
        if request.video_path:
            frames = sample_video_frames(str(request.video_path), k=self.config.video_frames)
            images = [img for _, img in frames]
            prompt = build_video_prompt(request.prompt, frames)
        content: List[dict] = [{"type": "text", "text": prompt}]
        for img in images:
            content.append(
                {"type": "image_url", "image_url": {"url": image_to_data_url(img)}}
            )
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": int(request.max_new_tokens or self.config.max_new_tokens),
            "temperature": float(request.temperature),
            "stream": False,
        }
        if self.config.json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _retry_delay(self, attempt: int, retry_after: Optional[str]) -> float:
        if retry_after:
            try:
                return min(30.0, max(0.0, float(retry_after)))
            except ValueError:
                pass
        return min(20.0, 1.0 * (2 ** attempt))

    def generate(self, request: VLMRequest) -> VLMResponse:
        t0 = time.time()
        body = json.dumps(self.build_payload(request)).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        key = self._api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        attempts = 1 + max(0, int(self.config.max_retries))
        last_err = "no attempt made"
        for attempt in range(attempts):
            if request.on_attempt is not None:
                request.on_attempt()
            req = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
            retry_after = None
            try:
                with urllib.request.urlopen(req, timeout=self.config.request_timeout_s) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                text = self._extract_text(data)
                return VLMResponse(
                    text=strip_reasoning(text), backend=self.name,
                    latency_s=time.time() - t0, model=self.config.model,
                )
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", "replace")[:300]
                except Exception:
                    pass
                last_err = f"HTTP {e.code} from {self.endpoint}: {detail}"
                if e.code not in _RETRY_HTTP:
                    raise RuntimeError(last_err) from None
                retry_after = e.headers.get("Retry-After") if e.headers else None
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last_err = f"network error calling {self.endpoint}: {e}"
            except (ValueError, KeyError, IndexError, TypeError) as e:
                raise RuntimeError(f"malformed response from {self.endpoint}: {e}") from e
            if attempt < attempts - 1:
                delay = self._retry_delay(attempt, retry_after)
                logger.warning("[vlm_backend] %s; retry %d/%d in %.1fs",
                               last_err, attempt + 1, attempts - 1, delay)
                self._sleep(delay)
        raise RuntimeError(f"openai_compatible request failed after {attempts} attempt(s): {last_err}")

    @staticmethod
    def _extract_text(data: dict) -> str:
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        return content or ""


# ---------------------------------------------------------------------------
# "none" backend, request budget, caching wrapper
# ---------------------------------------------------------------------------


class VLMUnavailable(RuntimeError):
    """Raised when a request is made on the ``none`` backend."""


class BudgetExceeded(RuntimeError):
    """The per-episode VLM request budget is used up."""


class NullVLMBackend:
    """The ``none`` choice: no model; language comes from templates."""

    name = "none"
    model_id = ""

    def is_available(self) -> bool:
        return False

    def generate(self, request: VLMRequest) -> VLMResponse:
        raise VLMUnavailable("no VLM backend configured (vlm.backend=none)")


class VLMBudget:
    """Hard per-episode request cap; the single place the limit is enforced.

    ``consume`` is called before every network attempt, including retries.
    ``limit <= 0`` means cache-only (no request may be made).
    """

    def __init__(self, limit: int = 3):
        self.limit = int(limit)
        self.used = 0
        self.cache_hits = 0
        self.denied = 0
        self._lock = threading.Lock()

    def consume(self) -> None:
        with self._lock:
            if self.used >= self.limit:
                self.denied += 1
                raise BudgetExceeded(
                    f"VLM request budget exhausted ({self.used}/{self.limit} requests this episode)"
                )
            self.used += 1

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def reset(self) -> None:
        with self._lock:
            self.used = self.cache_hits = self.denied = 0


_DISK_CACHES: Dict[str, Any] = {}


def _disk_cache_for(path: str):
    from .cache_manager import VLMDiskCache

    key = str(Path(path).resolve())
    if key not in _DISK_CACHES:
        _DISK_CACHES[key] = VLMDiskCache(path)
    return _DISK_CACHES[key]


class ManagedBackend:
    """Wraps a backend with the on-disk response cache and the request budget.

    A cache hit costs zero requests. Every real attempt (retries included) is
    counted by ``budget``; once it is exhausted ``generate`` raises
    :class:`BudgetExceeded` instead of calling the model.
    """

    def __init__(
        self,
        inner: VLMBackend,
        budget: Optional[VLMBudget] = None,
        cache: Any = None,
        selection_reason: str = "",
    ):
        self.inner = inner
        self.budget = budget if budget is not None else VLMBudget()
        self.cache = cache
        self.selection_reason = selection_reason

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def model_id(self) -> str:
        return str(getattr(self.inner, "model_id", "") or "")

    @property
    def is_null(self) -> bool:
        return isinstance(self.inner, NullVLMBackend)

    def is_available(self) -> bool:
        return self.inner.is_available()

    def request_key(self, request: VLMRequest) -> str:
        from .cache_manager import VLMDiskCache, file_sha256

        fps = [f"temp:{request.temperature}", f"max_tokens:{request.max_new_tokens}"]
        if request.video_path and os.path.exists(request.video_path):
            fps.append("video:" + file_sha256(request.video_path))
        for img in request.images:
            h = hashlib.sha256(img.tobytes())
            h.update(repr(img.size).encode())
            fps.append("img:" + h.hexdigest())
        return VLMDiskCache.make_key(f"{self.name}:{self.model_id}", request.prompt, fps)

    def generate(self, request: VLMRequest) -> VLMResponse:
        key = self.request_key(request) if self.cache is not None else ""
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached:
                self.budget.cache_hits += 1
                return VLMResponse(
                    text=cached, backend=self.name, latency_s=0.0,
                    cached=True, model=self.model_id,
                )
        resp = self.inner.generate(replace(request, on_attempt=self.budget.consume))
        if self.cache is not None and resp.text:
            self.cache.put(key, resp.text)
        return resp

    def discard(self, request: VLMRequest) -> None:
        """Drop a cached response that turned out to be unusable (e.g. bad JSON)."""
        if self.cache is not None:
            self.cache.discard(self.request_key(request))

    def new_episode(self) -> None:
        self.budget.reset()

    def stats(self) -> Dict[str, Any]:
        """Per-run counters for the manifest."""
        return {
            "backend": self.name,
            "model": self.model_id,
            "selection_reason": self.selection_reason,
            "requests_used": self.budget.used,
            "max_requests_per_episode": self.budget.limit,
            "cache_hits": self.budget.cache_hits,
            "budget_exhausted": self.budget.exhausted,
            "requests_denied": self.budget.denied,
        }


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

BACKEND_CHOICES = ("auto", "gemini", "openai_compatible", "local", "none")


@dataclass
class BackendSelection:
    backend: VLMBackend  # never None; NullVLMBackend when nothing is usable
    requested: str
    reason: str


def _unavailable_reason(name: str, cfg: VLMBackendConfig) -> str:
    if name == "gemini":
        return "gemini unavailable (google-genai missing or GEMINI_API_KEY unset)"
    if name == "openai_compatible":
        return (
            "openai_compatible unavailable (needs vlm.base_url and vlm.model, and "
            f"the key env var {cfg.api_key_env!r} unless the host is local)"
        )
    if name == "local":
        return (
            f"local unavailable (transformers_ok={_TRANSFORMERS_OK}, weights dir "
            f"'{cfg.local_weights}' present={Path(cfg.local_weights).is_dir()})"
        )
    return f"unknown backend {name!r}"


def select_backend(config: Optional[VLMBackendConfig] = None) -> BackendSelection:
    """Pick a backend for ``vlm.backend``; never raises, never returns ``None``.

    ``auto``: Gemini if the key is present, else an OpenAI-compatible endpoint
    if configured, else ``none`` (template language).
    """
    cfg = config or VLMBackendConfig()
    try:
        ensure_api_key_from_dotenv()
    except Exception:
        pass
    requested = (cfg.backend or "gemini").strip().lower()
    builders = {
        "gemini": GeminiBackend,
        "openai_compatible": OpenAICompatibleBackend,
        "local": LocalVLMBackend,
    }
    if requested == "none":
        return BackendSelection(NullVLMBackend(), requested, "vlm.backend=none")
    if requested == "auto":
        for name in ("gemini", "openai_compatible"):
            try:
                b = builders[name](cfg)
                if b.is_available():
                    return BackendSelection(b, requested, f"auto -> {name}")
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("[vlm_backend] auto probe %s failed: %s", name, e)
        return BackendSelection(
            NullVLMBackend(), requested,
            "auto -> none (no GEMINI_API_KEY and no openai_compatible endpoint configured)",
        )
    if requested not in builders:
        return BackendSelection(
            NullVLMBackend(), requested,
            f"unknown backend {cfg.backend!r} (want one of {', '.join(BACKEND_CHOICES)})",
        )
    try:
        b = builders[requested](cfg)
        if b.is_available():
            return BackendSelection(b, requested, f"{requested} (configured)")
    except Exception as e:
        return BackendSelection(NullVLMBackend(), requested, f"{requested} failed to init: {e}")
    return BackendSelection(NullVLMBackend(), requested, _unavailable_reason(requested, cfg))


def create_managed_backend(
    config: Optional[VLMBackendConfig] = None,
    budget: Optional[VLMBudget] = None,
    cache: Any = None,
) -> ManagedBackend:
    """Factory for the pipeline: selection + response cache + request budget.

    Never returns ``None``. With ``vlm.on_unavailable: raise`` an explicitly
    requested backend that cannot be used raises ``RuntimeError`` instead of
    degrading to templates.
    """
    cfg = config or VLMBackendConfig()
    sel = select_backend(cfg)
    if isinstance(sel.backend, NullVLMBackend) and sel.requested not in ("none", "auto") \
            and (cfg.on_unavailable or "").lower() == "raise":
        raise RuntimeError(f"VLM backend unavailable: {sel.reason}")
    if cache is None and cfg.cache_enabled and not isinstance(sel.backend, NullVLMBackend):
        cache = _disk_cache_for(cfg.cache_path)
    if budget is None:
        budget = VLMBudget(cfg.max_requests_per_episode)
    return ManagedBackend(sel.backend, budget=budget, cache=cache, selection_reason=sel.reason)


def create_vlm_backend(config: Optional[VLMBackendConfig] = None) -> Optional[VLMBackend]:
    """Build the configured backend. Returns ``None`` when unavailable.

    Never raises — use :func:`resolve_stage_backend` when the legacy strict
    (raise-on-missing) constructor contract is wanted. For the new
    auto/none-aware path use :func:`create_managed_backend`.
    """
    cfg = config or VLMBackendConfig()
    try:
        ensure_api_key_from_dotenv()
        name = (cfg.backend or "gemini").strip().lower()
        if name in ("auto", "none"):
            sel = select_backend(cfg)
            return None if isinstance(sel.backend, NullVLMBackend) else sel.backend
        if name == "gemini":
            backend: VLMBackend = GeminiBackend(cfg)
        elif name == "openai_compatible":
            backend = OpenAICompatibleBackend(cfg)
        elif name == "local":
            backend = LocalVLMBackend(cfg)
        else:
            logger.warning(
                "[vlm_backend] Unknown backend %r (want one of %s)",
                cfg.backend, ", ".join(BACKEND_CHOICES),
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
            logger.warning("[vlm_backend] %s.", _unavailable_reason(name, cfg))
        return None
    except Exception as e:  # never raise out of the factory
        logger.warning("[vlm_backend] create_vlm_backend failed: %s", e)
        return None


def resolve_stage_backend(
    stage_override: Optional[str],
    gemini_model: str = "gemini-2.5-flash",
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
    if name == "none":
        return NullVLMBackend()
    backend = create_vlm_backend(cfg)
    if backend is not None:
        return backend
    if name == "auto":
        return NullVLMBackend()
    if name == "openai_compatible":
        raise RuntimeError(f"OpenAI-compatible VLM backend requested but {_unavailable_reason(name, cfg)}.")
    if name == "local":
        raise RuntimeError(
            "Local VLM backend requested but unavailable "
            f"(transformers_ok={_TRANSFORMERS_OK}, weights dir "
            f"'{cfg.local_weights}' present={Path(cfg.local_weights).is_dir()}). "
            f"Download weights manually from {SMOLVLM_URL} — no auto-download by design."
        )
    if name != "gemini":
        raise ValueError(
            f"Unknown VLM backend {stage_override!r} (want one of {', '.join(BACKEND_CHOICES)})."
        )
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
