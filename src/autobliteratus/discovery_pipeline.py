"""Sandbox-only model mapping discovery pipeline.

This pipeline is intentionally outside ``autobliteratus.pipeline`` and the
runtime toggle registry. It accepts only an explicitly sandboxed execution
context, writes evidence inside the sandbox boundary, and never promotes or
mutates runtime mappings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from autobliteratus.core.discovery_experiment_pipeline import run_unknown_factor_battery
from autobliteratus.core.model_mapping_discovery import discover, update_database

PIPELINE_ID = "model_mapping_discovery_sandbox"


def _sandbox_root(context: dict[str, Any]) -> Path:
    if context.get("sandbox") is not True:
        raise PermissionError("Mapping discovery requires sandbox=True; runtime discovery is forbidden.")
    if context.get("allow_model_mutation", False) is not False:
        raise PermissionError("Mapping discovery refuses contexts that allow model mutation.")
    if context.get("runtime_pipeline", False) is True or context.get("runtime_mode") in {"live", "production", "runtime"}:
        raise PermissionError("Mapping discovery cannot run in a live or runtime context.")
    value = context.get("sandbox_root")
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError("sandbox_root is required for mapping discovery.")
    root = Path(value).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError("sandbox_root must be an existing directory.")
    return root


def _database_path(context: dict[str, Any], root: Path) -> Path:
    requested = context.get("model_mapping_db_path")
    path = (root / "model-mappings.json") if requested is None else Path(requested).expanduser().resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PermissionError("model_mapping_db_path must remain inside sandbox_root.") from exc
    return path


def _assert_sandbox_adapter(adapter: Any, root: Path) -> None:
    if getattr(adapter, "sandboxed", False) is not True:
        raise PermissionError("Mapping discovery requires an adapter constructed for sandbox execution.")
    adapter_root = getattr(adapter, "sandbox_root", None)
    if adapter_root is None or Path(adapter_root).expanduser().resolve() != root:
        raise PermissionError("Mapping discovery adapter is not bound to sandbox_root.")


def run(adapter: Any, context: dict[str, Any]) -> dict[str, Any]:
    """Run read-only discovery and persist evidence only within the sandbox."""
    root = _sandbox_root(context)
    _assert_sandbox_adapter(adapter, root)
    database_path = _database_path(context, root)
    discovery_context = dict(context)
    discovery_context["model_mapping_db_path"] = database_path
    report = discover(adapter, discovery_context)
    report["unknown_factor_battery"] = run_unknown_factor_battery(adapter, report, discovery_context)
    update_database(database_path, report)
    report.update(
        {
            "pipeline_id": PIPELINE_ID,
            "sandbox_root": str(root),
            "database_path": str(database_path),
            "runtime_pipeline_touched": False,
            "runtime_mapping_promoted": False,
            "model_modified": False,
            "optimization_performed": False,
        }
    )
    context["sandbox_mapping_discovery_report"] = report
    return report
