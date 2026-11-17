"""Runtime toggle registry for AUTOBLITERATUS.

Discovery reads static module metadata instead of importing every prototype.
That keeps ``--list-toggles`` and pipeline setup from executing optional module
imports or failing because an unrelated experimental dependency is absent.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

_INTERNAL_MODULES = {
    "dispatch",
    "toggle_registry",
    "runtime_selector",
    "tuning_orchestrator",
    "_101_tuning_orchestrator",
    "_102_runtime_selector",
    "_103_pipeline_entry",
}


def _build_registry() -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    tuning_dir = Path(__file__).resolve().parent
    for module_path in sorted(tuning_dir.glob("*.py")):
        module_name = module_path.stem
        if module_name.startswith("__") or module_name in _INTERNAL_MODULES:
            continue

        tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
        toggle: dict[str, Any] | None = None
        has_module_run = False
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run":
                has_module_run = True
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "TOGGLE"
                for target in node.targets
            ):
                try:
                    candidate = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue
                if isinstance(candidate, dict):
                    toggle = candidate
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "TOGGLE":
                try:
                    candidate = ast.literal_eval(node.value) if node.value is not None else None
                except (ValueError, TypeError):
                    continue
                if isinstance(candidate, dict):
                    toggle = candidate

        if not toggle or not toggle.get("id"):
            continue

        available = has_module_run
        toggle_id = str(toggle["id"])
        registry[toggle_id] = {
            "id": toggle_id,
            "default": bool(toggle.get("default", True)) and available,
            "available": available,
            "desc": str(toggle.get("desc", "")),
            "module": f"autobliteratus.tuning.{module_name}",
            "name": module_name,
        }

    return registry


TOGGLES: dict[str, dict[str, Any]] = _build_registry()


def enabled_toggles(
    overrides: dict[str, bool] | None = None,
    *,
    allow_unavailable: bool = False,
) -> dict[str, bool]:
    resolved: dict[str, bool] = {}
    overrides = overrides or {}
    for toggle_id, metadata in TOGGLES.items():
        value = bool(metadata.get("default", True))
        if toggle_id in overrides:
            value = bool(overrides[toggle_id])
        if value and not metadata.get("available", False) and not allow_unavailable:
            raise ValueError(
                f"Tuning stage '{toggle_id}' is a prototype without a callable "
                "run(adapter, context) implementation and cannot be enabled."
            )
        resolved[toggle_id] = value
    return resolved
