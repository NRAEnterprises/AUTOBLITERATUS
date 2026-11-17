"""Tuning type 70: version compatibility auto tuning.

Verify the ablated model loads in the target inference frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TOGGLE = {
    "id": "tuning.version_compatibility_auto",
    "default": False,
    "desc": "Load the exported model through declared inference-framework callbacks and report compatibility without mutation.",
}


@dataclass
class CompatibilityReport:
    per_framework: dict = field(default_factory=dict)


class VersionCompatibilityAutoTuning:
    def __init__(self, loaders: dict, enabled: bool = True) -> None:
        self.loaders = loaders
        self.enabled = enabled

    def run(self, model_path: str) -> CompatibilityReport:
        if not self.enabled:
            return CompatibilityReport()
        results = {}
        for name, loader in self.loaders.items():
            try:
                loader(model_path)
                results[name] = "ok"
            except Exception as e:  # noqa: BLE001 - isolate arbitrary framework loader failures.
                results[name] = f"fail: {e}"
        return CompatibilityReport(per_framework=results)


def run(adapter: Any, context: dict[str, Any]) -> dict[str, Any]:
    """Run caller-declared compatibility loaders against the exported artifact."""
    del adapter
    loaders = context.get("compatibility_loaders", {})
    if not isinstance(loaders, dict):
        raise TypeError("compatibility_loaders must be a mapping of framework names to callables.")
    if any(not isinstance(name, str) or not callable(loader) for name, loader in loaders.items()):
        raise TypeError("compatibility_loaders must map string framework names to callables.")
    model_path = context.get("exported_path") or context.get("output_dir")
    if not model_path:
        report = {"status": "not_evaluated", "reason": "no_exported_model", "per_framework": {}}
    else:
        artifact_path = Path(model_path).expanduser()
        if not artifact_path.is_dir():
            raise FileNotFoundError(f"Exported model directory does not exist: {artifact_path}")
        per_framework: dict[str, str] = {}
        for name, loader in sorted(loaders.items()):
            try:
                loader(str(artifact_path))
                per_framework[name] = "ok"
            except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
                per_framework[name] = f"fail: {type(exc).__name__}"
        report = {
            "status": "measured" if per_framework else "not_evaluated",
            "model_path": str(artifact_path),
            "per_framework": per_framework,
            "framework_count": len(per_framework),
            "model_modified": False,
        }
    context["version_compatibility_report"] = report
    return report
