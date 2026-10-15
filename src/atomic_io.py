"""Atomic file writing and run manifest tracking (Phase 1)."""
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


def atomic_write_text(path: Path | str, content: str, encoding: str = "utf-8") -> None:
    """Write text to a temporary file in the same directory and atomically replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise


def atomic_write_json(path: Path | str, data: Any, indent: int = 2) -> None:
    """Serialize data to JSON and write atomically."""
    text = json.dumps(data, indent=indent, sort_keys=False) + "\n"
    atomic_write_text(path, text)


@dataclass
class StageRecord:
    status: str  # "ok" | "cached" | "skipped" | "degraded" | "failed"
    duration_s: float = 0.0
    detail: Optional[str] = None


@dataclass
class RunManifest:
    episode_id: str
    video_path: str
    allow_partial: bool = False
    started_at: float = field(default_factory=time.time)
    stages: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    partial_failures: list[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def record_stage(self, name: str, status: str, duration_s: float = 0.0, detail: Optional[str] = None) -> None:
        self.stages[name] = {
            "status": status,
            "duration_s": round(float(duration_s), 4),
            "detail": detail,
        }
        if status in ("degraded", "failed"):
            self.partial_failures.append(f"{name}: {detail or status}")

    @property
    def exit_code(self) -> int:
        if any(s["status"] == "failed" for s in self.stages.values()):
            return 2 if self.allow_partial else 1
        if any(s["status"] == "degraded" for s in self.stages.values()):
            return 2
        return 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "video_path": self.video_path,
            "allow_partial": self.allow_partial,
            "overall_status": (
                "success" if self.exit_code == 0 else ("partial" if self.exit_code == 2 else "failed")
            ),
            "exit_code": self.exit_code,
            "elapsed_s": round(time.time() - self.started_at, 3),
            "stages": self.stages,
            "partial_failures": self.partial_failures,
            "metrics": self.metrics,
        }

    def save(self, path: Path | str) -> None:
        atomic_write_json(path, self.to_dict())
