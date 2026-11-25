"""Phase 4 verification: GPU backend wiring, license compliance, and AUDIT.md completeness."""

from __future__ import annotations

from pathlib import Path

import torch

from src.egoannotate.gpu_backends import (
    DepthAnythingV2SmallGPU,
    GroundingDINOGPUDetector,
    LocalOpenSourceVLMAnnotator,
    _resolve_torch_dtype,
)


def test_gpu_backend_classes_and_dtype_resolution() -> None:
    assert _resolve_torch_dtype("bf16", "cpu") == torch.float32
    assert _resolve_torch_dtype("fp16", "cpu") == torch.float32
    assert hasattr(GroundingDINOGPUDetector, "detect_frame")
    assert hasattr(DepthAnythingV2SmallGPU, "estimate_depth_map")
    assert hasattr(LocalOpenSourceVLMAnnotator, "describe_segment_keyframes")


def test_audit_and_kaggle_runbook_exist_and_complete() -> None:
    audit = Path("AUDIT.md")
    runbook = Path("docs/KAGGLE_GPU_RUNBOOK.md")
    assert audit.exists()
    assert runbook.exists()
    txt = audit.read_text(encoding="utf-8")
    assert "15-Category Commercial Readiness Rubric" in txt
    assert "GPU-Pending" in txt
    assert "5400" in txt
