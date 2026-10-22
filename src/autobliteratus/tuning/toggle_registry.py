"""Runtime toggle registry for AUTOBLITERATUS."""

from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules
from typing import Any


def _iter_tuning_modules() -> list[str]:
    package_name = "autobliteratus.tuning"
    package = import_module(package_name)
    modules: list[str] = []
    for _, module_name, _ in iter_modules(package.__path__):
        if module_name.startswith("__"):
            continue
        if module_name in {"dispatch", "toggle_registry", "runtime_selector", "tuning_orchestrator"}:
            continue
        modules.append(module_name)
    return modules


def _build_registry() -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    for module_name in _iter_tuning_modules():
        full_name = f"autobliteratus.tuning.{module_name}"
        try:
            module = import_module(full_name)
        except Exception:
            continue

        toggle = getattr(module, "TOGGLE", None)
        if not isinstance(toggle, dict):
            continue

        toggle_id = toggle.get("id")
        if not toggle_id:
            continue

        registry[toggle_id] = {
            "default": bool(toggle.get("default", True)),
            "desc": str(toggle.get("desc", "")),
            "module": full_name,
            "name": module_name,
        }

    return registry


TOGGLES: dict[str, dict[str, Any]] = _build_registry()


def enabled_toggles(overrides: dict[str, bool] | None = None) -> dict[str, bool]:
    resolved: dict[str, bool] = {}
    overrides = overrides or {}
    for toggle_id, metadata in TOGGLES.items():
        value = bool(metadata.get("default", True))
        if toggle_id in overrides:
            value = bool(overrides[toggle_id])
        resolved[toggle_id] = value
    return resolved
