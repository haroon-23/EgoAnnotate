"""Shared hermetic-test helpers: stub or block heavy third-party modules.

These helpers make retargeting/perception tests hermetic in environments where
heavy dependencies (scipy, mujoco, mink, ...) are not installed, WITHOUT
affecting machines where they are installed:

- :func:`ensure_scipy_importable` injects minimal ``scipy`` stubs ONLY if the
  real scipy cannot be imported. On a machine with real scipy it is a no-op.
- :func:`blocked_modules` context manager forces ``import <name>`` to raise
  ``ImportError`` (the Phase-C ``no_transformers`` technique).
- :func:`fake_module` context manager installs a fake module object.

This module is named ``hermetic_stubs.py`` (not ``test_*``) so pytest does not
collect it as a test file.
"""
from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from typing import Iterator


def ensure_scipy_importable() -> None:
    """Make ``import src.retargeting`` work without a real scipy install.

    Injects minimal ``scipy.signal.savgol_filter`` and
    ``scipy.spatial.transform.Rotation`` stubs ONLY when the real scipy is
    absent. No-op when real scipy imports fine (e.g. on the dev Mac).
    """
    try:
        import scipy.signal  # noqa: F401
        import scipy.spatial.transform  # noqa: F401
        return
    except ImportError:
        pass

    import numpy as np

    scipy = types.ModuleType("scipy")
    signal = types.ModuleType("scipy.signal")

    def savgol_filter(x, *args, **kwargs):  # noqa: D103
        return np.asarray(x)

    signal.savgol_filter = savgol_filter

    spatial = types.ModuleType("scipy.spatial")
    transform = types.ModuleType("scipy.spatial.transform")

    class Rotation:  # noqa: D106 - import-time stub only
        @classmethod
        def from_quat(cls, *args, **kwargs):
            raise NotImplementedError("scipy stub")

        @classmethod
        def from_matrix(cls, *args, **kwargs):
            raise NotImplementedError("scipy stub")

        def as_matrix(self, *args, **kwargs):
            raise NotImplementedError("scipy stub")

        def as_quat(self, *args, **kwargs):
            raise NotImplementedError("scipy stub")

    transform.Rotation = Rotation
    spatial.transform = transform
    scipy.signal = signal
    scipy.spatial = spatial
    # Marker so scipy_stub_for_import() only removes OUR stubs, never real modules.
    for mod in (scipy, signal, spatial, transform):
        mod.__hermetic_stub__ = True  # type: ignore[attr-defined]
    sys.modules["scipy"] = scipy
    sys.modules["scipy.signal"] = signal
    sys.modules["scipy.spatial"] = spatial
    sys.modules["scipy.spatial.transform"] = transform


@contextmanager
def scipy_stub_for_import() -> Iterator[None]:
    """Inject scipy stubs for a single import, then fully evict them.

    Use as::

        with scipy_stub_for_import():
            from src.retargeting import whatever  # noqa

    The stubs satisfy ``from scipy... import ...`` at import time. Afterwards
    the stubs AND every ``src.retargeting.*`` module imported under them are
    evicted from ``sys.modules`` — names the importing module already bound
    (classes, functions) keep working via direct references, but later test
    modules see the real environment (or the real ImportError), exactly as if
    the stub had never existed. Without this, a leaked stub changes
    pre-existing tests' failure modes in sessions without scipy.
    Only modules carrying the ``__hermetic_stub__`` marker are removed, plus
    the ``src.retargeting`` package tree they pulled in.
    """
    ensure_scipy_importable()
    injected = any(
        getattr(sys.modules.get(n), "__hermetic_stub__", False)
        for n in ("scipy", "scipy.signal", "scipy.spatial", "scipy.spatial.transform")
    )
    try:
        yield
    finally:
        if not injected:
            return  # real scipy present; nothing to clean up
        for name in list(sys.modules):
            if getattr(sys.modules[name], "__hermetic_stub__", False):
                del sys.modules[name]
        for name in [n for n in sys.modules
                     if n == "src.retargeting" or n.startswith("src.retargeting.")
                     or n == "src.pipeline"]:
            del sys.modules[name]


@contextmanager
def blocked_modules(*names: str) -> Iterator[None]:
    """Force ``import <name>`` to raise ImportError inside the context.

    Restores any previous ``sys.modules`` entries on exit.
    """
    sentinel = object()
    saved = {n: sys.modules.get(n, sentinel) for n in names}
    for n in names:
        sys.modules[n] = None  # type: ignore[assignment]
    try:
        yield
    finally:
        for n in names:
            if saved[n] is sentinel:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = saved[n]


@contextmanager
def fake_module(name: str, module: types.ModuleType) -> Iterator[None]:
    """Install a fake module object as ``sys.modules[name]`` inside the context."""
    sentinel = object()
    saved = sys.modules.get(name, sentinel)
    sys.modules[name] = module
    try:
        yield
    finally:
        if saved is sentinel:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved
