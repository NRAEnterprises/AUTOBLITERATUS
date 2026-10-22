"""Registry builder that discovers callable tuning stages by module metadata."""

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


def build_registry(adapter: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
    registry: dict[str, Any] = {}
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

        run_fn = getattr(module, "run", None)
        if callable(run_fn):
            registry[toggle_id] = run_fn
    return registry
