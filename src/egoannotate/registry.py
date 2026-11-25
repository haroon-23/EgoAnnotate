"""Modular plugin registry with strict commercial-license firewall."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

COMMERCIAL_PERMISSIVE_LICENSES = frozenset(
    {
        "Apache-2.0",
        "MIT",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "zlib",
        "CC-BY-4.0",
        "CC0-1.0",
        "Unlicense",
        "Internal-Math",
    }
)


class CommercialLicenseError(RuntimeError):
    """Raised when a non-commercial or paid plugin is requested without explicit opt-in."""


@dataclass(frozen=True)
class BackendSpec:
    stage: str
    name: str
    factory: Callable[..., Any]
    code_license: str
    weights_license: str
    commercial_ok: bool
    requires_gpu: bool = False
    is_paid_api: bool = False
    notes: str = ""


class PluginRegistry:
    """Central registry for swappable perception, geometry, VLM, retargeting, and export plugins."""

    def __init__(self) -> None:
        self._items: dict[str, dict[str, BackendSpec]] = {}

    def register(
        self,
        stage: str,
        name: str,
        factory: Callable[..., Any],
        *,
        code_license: str,
        weights_license: str = "N/A",
        commercial_ok: bool | None = None,
        requires_gpu: bool = False,
        is_paid_api: bool = False,
        notes: str = "",
    ) -> None:
        if commercial_ok is None:
            code_ok = code_license in COMMERCIAL_PERMISSIVE_LICENSES
            weights_ok = weights_license in ("N/A", "") or weights_license in COMMERCIAL_PERMISSIVE_LICENSES
            commercial_ok = bool(code_ok and weights_ok and not is_paid_api)
        spec = BackendSpec(
            stage=stage,
            name=name,
            factory=factory,
            code_license=code_license,
            weights_license=weights_license,
            commercial_ok=commercial_ok,
            requires_gpu=requires_gpu,
            is_paid_api=is_paid_api,
            notes=notes,
        )
        self._items.setdefault(stage, {})[name] = spec

    def get_spec(self, stage: str, name: str) -> BackendSpec:
        stage_map = self._items.get(stage, {})
        if name not in stage_map:
            avail = sorted(stage_map.keys())
            raise KeyError(f"Unknown backend '{name}' for stage '{stage}'. Available: {avail}")
        return stage_map[name]

    def create(
        self,
        stage: str,
        name: str,
        *,
        allow_non_commercial: bool = False,
        **kwargs: Any,
    ) -> Any:
        spec = self.get_spec(stage, name)
        if spec.is_paid_api:
            raise CommercialLicenseError(
                f"Backend '{stage}:{name}' uses a paid API and is forbidden in EgoAnnotate v3."
            )
        if not spec.commercial_ok and not allow_non_commercial:
            raise CommercialLicenseError(
                f"Backend '{stage}:{name}' is not commercially licensed "
                f"(code={spec.code_license}, weights={spec.weights_license}). "
                f"Set allow_non_commercial_plugins=True only for internal non-commercial R&D."
            )
        return spec.factory(**kwargs)

    def list_stage(self, stage: str) -> list[BackendSpec]:
        return list(self._items.get(stage, {}).values())

    def audit_table(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for stage in sorted(self._items.keys()):
            for name in sorted(self._items[stage].keys()):
                s = self._items[stage][name]
                rows.append(
                    {
                        "stage": s.stage,
                        "name": s.name,
                        "code_license": s.code_license,
                        "weights_license": s.weights_license,
                        "commercial_ok": s.commercial_ok,
                        "requires_gpu": s.requires_gpu,
                        "is_paid_api": s.is_paid_api,
                        "notes": s.notes,
                    }
                )
        return rows


REGISTRY = PluginRegistry()


def register_backend(
    stage: str,
    name: str,
    *,
    code_license: str,
    weights_license: str = "N/A",
    commercial_ok: bool | None = None,
    requires_gpu: bool = False,
    is_paid_api: bool = False,
    notes: str = "",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a backend factory with commercial license metadata."""

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        REGISTRY.register(
            stage=stage,
            name=name,
            factory=fn,
            code_license=code_license,
            weights_license=weights_license,
            commercial_ok=commercial_ok,
            requires_gpu=requires_gpu,
            is_paid_api=is_paid_api,
            notes=notes,
        )
        return fn

    return _decorator
