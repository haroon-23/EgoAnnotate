"""Unit tests for Phase 1A GPU model cache and VLA caption sanitizer."""

from __future__ import annotations

from src.egoannotate.gpu_backends import (
    _MODEL_CACHE,
    clear_gpu_model_cache,
    sanitize_vla_caption,
)


def test_sanitize_vla_caption_removes_robot_and_clothing_leakage() -> None:
    c1 = sanitize_vla_caption("The robot's right hand grasps a shovel.", "grasp")
    assert "robot" not in c1.lower()
    assert c1 == "Right hand grasps a shovel."

    c2 = sanitize_vla_caption(
        "A human hand wearing a brown and white striped long sleeve shirt and "
        "black flip flops manipulates a piece of metal with a welding torch.",
        "manipulate",
    )
    assert "shirt" not in c2.lower()
    assert "flip flops" not in c2.lower()
    assert "welding torch" in c2.lower()

    c3 = sanitize_vla_caption(
        "The right hand of a person wearing a yellow shirt is grasping a piece "
        "of white fabric.",
        "grasp",
    )
    assert "yellow shirt" not in c3.lower()
    assert "person" not in c3.lower()
    assert "white fabric" in c3.lower()


def test_sanitize_vla_caption_preserves_garment_workpieces() -> None:
    c = sanitize_vla_caption("Right hand grasp green shirt.", "grasp")
    assert c == "Right hand grasp green shirt."


def test_clear_gpu_model_cache_empties_registry() -> None:
    _MODEL_CACHE[("unit", "m", "cpu", "fp32")] = (object(), object())
    clear_gpu_model_cache()
    assert len(_MODEL_CACHE) == 0
