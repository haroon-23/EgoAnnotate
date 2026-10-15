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


class VLMDiskCache:
    """Content-addressed JSON cache so no Gemini call is repeated for identical inputs."""

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


GLOBAL_VLM_CACHE = VLMDiskCache()


class StageCache:
    """Resumable per-video stage cache stored under cache/stages/<video_hash>/."""

    def __init__(self, video_path: Path | str, cache_root: Path | str = "cache/stages", enabled: bool = True) -> None:
        self.enabled = enabled and (os.environ.get("EGO_DISABLE_STAGE_CACHE") != "1")
        self.video_path = Path(video_path)
        self.video_hash = file_sha256(self.video_path)[:16] if self.video_path.exists() else "no_video"
        self.stage_dir = Path(cache_root) / f"{self.video_path.stem}_{self.video_hash}"
        if self.enabled:
            self.stage_dir.mkdir(parents=True, exist_ok=True)

    def load_or_compute(self, stage_name: str, compute_fn: Callable[[], Any]) -> tuple[Any, bool]:
        """Return (result, was_cached)."""
        pkl_path = self.stage_dir / f"{stage_name}.pkl"
        if self.enabled and pkl_path.exists():
            try:
                with open(pkl_path, "rb") as f:
                    return pickle.load(f), True
            except Exception:
                pass
        result = compute_fn()
        if self.enabled:
            tmp_path = pkl_path.with_suffix(".pkl.tmp")
            try:
                with open(tmp_path, "wb") as f:
                    pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp_path, pkl_path)
            except Exception:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
        return result, False
