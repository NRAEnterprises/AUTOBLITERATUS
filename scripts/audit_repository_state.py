"""Statically audit the package's toggle inventory and repository layout.

Run from any directory with ``python scripts/audit_repository_state.py``.
This mirrors the runtime registry's discovery boundary: top-level Python files
under ``src/autobliteratus/tuning`` with literal module-level TOGGLE metadata;
a module is callable when it has a top-level ``run`` function.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUNING_DIR = ROOT / "src" / "autobliteratus" / "tuning"
INTERNAL_MODULES = {
    "dispatch",
    "toggle_registry",
    "runtime_selector",
    "tuning_orchestrator",
    "evaluation_metrics",
    "mapping_promotion_gate",
    "mutation_firewall",
    "mutation_transaction",
    "promotion_evidence_coordinator",
    "shared_dict",
    "delta_weight_export_auto_tuning",
    "post_ablation_capability_tuning",
    "_101_tuning_orchestrator",
    "_102_runtime_selector",
    "_103_pipeline_entry",
}


def inspect_module(path: Path) -> tuple[dict | None, bool]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    toggle = None
    has_run = False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run":
            has_run = True
        value = None
        is_toggle = False
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "TOGGLE"
                for target in node.targets
            )
        ) or (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "TOGGLE"
        ):
            value, is_toggle = node.value, True
        if is_toggle and value is not None:
            try:
                candidate = ast.literal_eval(value)
            except (ValueError, TypeError):
                continue
            if isinstance(candidate, dict):
                toggle = candidate
    return toggle, has_run


def main() -> int:
    records = []
    missing_metadata = []
    for path in sorted(TUNING_DIR.glob("*.py")):
        if path.stem.startswith("__") or path.stem in INTERNAL_MODULES:
            continue
        toggle, has_run = inspect_module(path)
        if not toggle or not toggle.get("id"):
            missing_metadata.append(path.relative_to(ROOT).as_posix())
            continue
        records.append(
            {
                "id": str(toggle["id"]),
                "module": path.relative_to(ROOT).as_posix(),
                "callable": has_run,
                "default": bool(toggle.get("default", True)) and has_run,
            }
        )

    ids = [record["id"] for record in records]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    root_python = sorted(
        path.name for path in ROOT.glob("*.py") if path.is_file()
    )
    result = {
        "repository_root": ROOT.name,
        "discovery_source": "src/autobliteratus/tuning/*.py",
        "discovery_rule": "literal module-level TOGGLE; available iff top-level run exists",
        "toggle_total": len(records),
        "callable_total": sum(record["callable"] for record in records),
        "unavailable_total": sum(not record["callable"] for record in records),
        "duplicate_toggle_ids": duplicates,
        "modules_missing_literal_toggle": missing_metadata,
        "root_level_python_files_not_discovered_by_runtime_registry": root_python,
        "toggles": records,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if duplicates or missing_metadata else 0


if __name__ == "__main__":
    raise SystemExit(main())
