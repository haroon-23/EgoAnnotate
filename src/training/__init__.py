"""Training package (Phase B): dataset loader + vendored minimal ACT policy.

``torch`` is an OPTIONAL dependency: it is imported lazily so that pipeline,
export, merge and validation code paths never require it. Use
:func:`require_torch` at construction time of any training class.
"""

from __future__ import annotations

try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover - environment without torch
    torch = None  # type: ignore[assignment]


def require_torch():
    """Return the torch module, or raise a helpful error if not installed."""
    if torch is None:
        raise ImportError(
            "torch is required for training (LeRobotParquetDataset / TinyACTPolicy) "
            "but is not installed. Install the CPU-only wheel with:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )
    return torch


__all__ = ["require_torch"]
