from __future__ import annotations
"""Atomic file writing and run manifest tracking (Phase 1)."""

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


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
    internal_test_no_consent: bool = False
    started_at: float = field(default_factory=time.time)
    stages: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    partial_failures: list[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    accept_episode: bool = False
    quality_tier: str = "PENDING"
    failed_quality_checks: list[str] = field(default_factory=list)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)

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

    def evaluate_quality(self, episode: Any, quality_config_path: Optional[Path | str] = None) -> bool:
        """Evaluate episode against physical data-quality thresholds from configs/quality.yaml.

        Checks:
        - detected hand ratio >= min_detected_hand_ratio (default 0.25)
        - untruncated hand ratio >= min_untruncated_ratio (default 0.50)
        - max segment duration <= max_segment_duration_sec (default 10.0s)
        - min segment duration >= min_segment_duration_sec (default 0.50s)
        - left/right balance flag (if required)
        """
        import yaml
        cfg_path = Path(quality_config_path or "configs/quality.yaml")
        thresholds: Dict[str, Any] = {}
        if cfg_path.exists():
            try:
                raw_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                thresholds = raw_cfg.get("quality_thresholds") or raw_cfg.get("gates") or {}
            except Exception:
                thresholds = {}

        min_detected_ratio = float(thresholds.get("min_detected_hand_ratio", thresholds.get("min_hand_valid_ratio", 0.25)))
        min_untruncated_ratio = float(thresholds.get("min_untruncated_ratio", 0.50))
        max_segment_sec = float(thresholds.get("max_segment_duration_sec", 10.0))
        min_segment_sec = float(thresholds.get("min_segment_duration_sec", 0.50))
        require_balance = bool(thresholds.get("require_left_right_balance", False))

        frames = getattr(episode, "frames", [])
        n_frames = max(1, len(frames))

        detected_count = 0
        untruncated_count = 0
        left_count = 0
        right_count = 0

        for f in frames:
            lh = getattr(f, "left_hand", None)
            rh = getattr(f, "right_hand") if hasattr(f, "right_hand") else None

            def _is_detected(h: Any) -> bool:
                if not h:
                    return False
                src = str(getattr(h, "source", "interpolated" if getattr(h, "is_interpolated", False) else "detected")).lower()
                is_int = bool(getattr(h, "is_interpolated", False) or src in ("interpolated", "held"))
                return (src in ("detected", "rescued")) and not is_int

            l_det = _is_detected(lh)
            r_det = _is_detected(rh)
            if l_det:
                left_count += 1
            if r_det:
                right_count += 1
            if l_det or r_det:
                detected_count += 1

            # Check untruncated: landmarks within normalized bounds [0.02, 0.98]
            def _is_untruncated(h: Any) -> bool:
                if not h:
                    return False
                xs = getattr(h, "x", None)
                ys = getattr(h, "y", None)
                if xs is not None and ys is not None and len(xs) > 0:
                    min_x, max_x = float(np.min(xs)), float(np.max(xs))
                    min_y, max_y = float(np.min(ys)), float(np.max(ys))
                    return (min_x >= 0.01 and max_x <= 0.99 and min_y >= 0.01 and max_y <= 0.99)
                return True

            if (lh and _is_untruncated(lh)) or (rh and _is_untruncated(rh)):
                untruncated_count += 1

        detected_ratio = round(detected_count / n_frames, 4)
        untruncated_ratio = round(untruncated_count / n_frames, 4)
        left_ratio = round(left_count / n_frames, 4)
        right_ratio = round(right_count / n_frames, 4)

        # Segments
        segments = getattr(episode, "segments", [])
        seg_durations = [float(s.end_time - s.start_time) for s in segments] if segments else [0.0]
        max_seg_len = round(max(seg_durations), 3) if seg_durations else 0.0
        min_seg_len = round(min(seg_durations), 3) if seg_durations else 0.0

        # Balance flag
        balance_flag = bool(abs(left_ratio - right_ratio) < 0.65 or (left_ratio > 0.10 and right_ratio > 0.10))

        failed_checks: list[str] = []
        if detected_ratio < 0.25:
            quality_tier = "REJECT"
            failed_checks.append(f"low_detected_hand_ratio ({detected_ratio:.3f} < 0.25)")
        elif detected_ratio < 0.60:
            quality_tier = "REVIEW"
            failed_checks.append(f"review_tier_detected_hand_ratio ({detected_ratio:.3f} in [0.25, 0.60))")
        else:
            quality_tier = "ACCEPT"

        if untruncated_ratio < min_untruncated_ratio:
            failed_checks.append(f"low_untruncated_ratio ({untruncated_ratio:.3f} < {min_untruncated_ratio})")
        if max_seg_len > max_segment_sec:
            failed_checks.append(f"max_segment_duration_exceeded ({max_seg_len:.2f}s > {max_segment_sec}s)")
        if segments and min_seg_len < min_segment_sec:
            failed_checks.append(f"min_segment_duration_too_short ({min_seg_len:.2f}s < {min_segment_sec}s)")
        if require_balance and not balance_flag:
            failed_checks.append(f"left_right_unbalanced (left={left_ratio:.2f}, right={right_ratio:.2f})")

        self.quality_tier = quality_tier
        self.accept_episode = (quality_tier == "ACCEPT" and (len(failed_checks) == 0 or failed_checks == []))
        self.failed_quality_checks = failed_checks
        self.quality_metrics = {
            "quality_tier": quality_tier,
            "detected_hand_ratio": detected_ratio,
            "untruncated_ratio": untruncated_ratio,
            "max_segment_duration_sec": max_seg_len,
            "min_segment_duration_sec": min_seg_len,
            "left_hand_ratio": left_ratio,
            "right_hand_ratio": right_ratio,
            "left_right_balance_flag": balance_flag,
        }
        return self.accept_episode

    def to_dict(self) -> Dict[str, Any]:
        doc = {
            "episode_id": self.episode_id,
            "overall_status": "accepted" if self.accept_episode else ("review" if getattr(self, "quality_tier", "") == "REVIEW" else "rejected"),
            "quality_tier": getattr(self, "quality_tier", "ACCEPT" if self.accept_episode else "REJECT"),
            "accept_episode": self.accept_episode,
            "failed_quality_checks": self.failed_quality_checks,
            "quality_metrics": self.quality_metrics,
            "video_path": self.video_path,
            "allow_partial": self.allow_partial,
            "exit_code": self.exit_code,
            "elapsed_s": round(time.time() - self.started_at, 3),
            "stages": self.stages,
            "partial_failures": self.partial_failures,
            "metrics": self.metrics,
        }
        if self.internal_test_no_consent:
            doc["consent_status"] = "UNSET_INTERNAL_TEST"
            doc["distribution_restriction"] = "NOT_FOR_DISTRIBUTION"
            doc["distribution"] = "NOT_FOR_DISTRIBUTION"
        return doc

    def save(self, path: Path | str) -> None:
        atomic_write_json(path, self.to_dict())
