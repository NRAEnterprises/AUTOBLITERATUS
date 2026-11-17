"""Registry builder for tuning modules with an explicit callable stage entry."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def build_registry(adapter: Any = None, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return module-level ``run(adapter, context)`` stage implementations.

    Class methods are intentionally not guessed or instantiated: their
    constructors and required evaluation/correction callbacks vary by module.
    Static toggle discovery marks modules as available only when a top-level
    ``run`` function exists, so only those few modules are imported here.
    """
    from autobliteratus.tuning.toggle_registry import TOGGLES

    registry: dict[str, Any] = {}
    for toggle_id, metadata in TOGGLES.items():
        if not metadata.get("available", False):
            continue
        module = import_module(metadata["module"])
        run_fn = getattr(module, "run", None)
        if not callable(run_fn):
            raise TypeError(
                f"Stage '{toggle_id}' is marked callable but does not expose run(adapter, context)."
            )
        registry[toggle_id] = run_fn
    return registry
