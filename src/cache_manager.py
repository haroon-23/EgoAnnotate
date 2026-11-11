"""Deterministic disk caching for VLM responses and pipeline stages."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from .atomic_io import atomic_write_json

_VLM_CACHE_LOCK = threading.Lock()


def file_sha256(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_hash(obj: Any, length: int = 16) -> str:
    """Order-independent SHA-256 of a JSON-like object (non-JSON values via ``str``)."""
    blob = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


def make_stage_key(config: Any = None, model_id: str = "", prompt: str = "") -> str:
    """Cache-key component covering a stage's config, model id and prompt.

    Pass only the config sections the stage depends on so unrelated edits do
    not invalidate it.
    """
    return stable_hash({"config": config, "model": model_id, "prompt": prompt}, 24)


class VLMDiskCache:
    """Content-addressed JSON cache so no VLM request is repeated for identical inputs."""

    def __init__(self, cache_path: Path | str = "cache/vlm_cache.json") -> None:
        self.cache_path = Path(cache_path)
        self._data: dict[str, str] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self.cache_path.exists():
            try:
                self._data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}
        self._loaded = True

    @staticmethod
    def make_key(model: str, prompt: str, media_fingerprints: list[str]) -> str:
        h = hashlib.sha256()
        h.update(model.encode("utf-8"))
        h.update(b"\x00")
        h.update(prompt.encode("utf-8"))
        for fp in media_fingerprints:
            h.update(b"\x00")
            h.update(fp.encode("utf-8"))
        return h.hexdigest()

    def get(self, key: str) -> Optional[str]:
        if os.environ.get("EGO_DISABLE_VLM_CACHE") == "1":
            return None
        with _VLM_CACHE_LOCK:
            self._ensure_loaded()
            return self._data.get(key)

    def put(self, key: str, value: str) -> None:
        if not value:
            return
        with _VLM_CACHE_LOCK:
            self._ensure_loaded()
            self._data[key] = value
            atomic_write_json(self.cache_path, self._data)

    def discard(self, key: str) -> None:
        with _VLM_CACHE_LOCK:
            self._ensure_loaded()
            if self._data.pop(key, None) is not None:
                atomic_write_json(self.cache_path, self._data)


GLOBAL_VLM_CACHE = VLMDiskCache()


class StageCache:
    """Resumable per-video stage cache stored under cache/stages/<video>_<hash>/.

    A result is reused only when the video bytes, the stage name AND the stage
    key all match. The stage key (``load_or_compute(..., key=...)``, built with
    :func:`make_stage_key`) must cover the config, model id and prompt the stage
    depends on; ``salt`` adds the same for every stage of this cache.
    """

    def __init__(
        self,
        video_path: Path | str,
        cache_root: Path | str = "cache/stages",
        enabled: bool = True,
        salt: Any = None,
    ) -> None:
        self.enabled = enabled and (os.environ.get("EGO_DISABLE_STAGE_CACHE") != "1")
        self.video_path = Path(video_path)
        self.video_hash = file_sha256(self.video_path)[:16] if self.video_path.exists() else "no_video"
        self.salt = salt
        self.stage_dir = Path(cache_root) / f"{self.video_path.stem}_{self.video_hash}"
        if self.enabled:
            self.stage_dir.mkdir(parents=True, exist_ok=True)

    def stage_path(self, stage_name: str, key: Any = None) -> Path:
        digest = stable_hash({"stage": stage_name, "salt": self.salt, "key": key}, 12)
        return self.stage_dir / f"{stage_name}.{digest}.pkl"

    def load_or_compute(
        self,
        stage_name: str,
        compute_fn: Callable[[], Any],
        key: Any = None,
        cacheable: Optional[Callable[[Any], bool]] = None,
    ) -> tuple[Any, bool]:
        """Return (result, was_cached).

        ``cacheable(result) -> False`` skips persisting, e.g. a template
        fallback produced because the VLM was momentarily unavailable.
        """
        pkl_path = self.stage_path(stage_name, key)
        if self.enabled and pkl_path.exists():
            try:
                with open(pkl_path, "rb") as f:
                    return pickle.load(f), True
            except Exception:
                pass
        result = compute_fn()
        if self.enabled and (cacheable is None or cacheable(result)):
            tmp_path = pkl_path.with_suffix(".pkl.tmp")
            try:
                with open(tmp_path, "wb") as f:
                    pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp_path, pkl_path)
            except Exception:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
        return result, False
