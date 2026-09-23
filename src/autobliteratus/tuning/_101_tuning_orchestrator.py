"""Tuning type 101: tuning orchestrator.

Reads the toggle registry. Runs every enabled tuning in dependency order.
Nothing runs unless its toggle is on. Nothing runs before its dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


TOGGLE = {"id": "tuning.orchestrator", "default": True}


@dataclass
class OrchestratorReport:
    executed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)


class TuningOrchestrator:
    def __init__(
        self,
        registry: dict[str, dict],
        dependencies: dict[str, list[str]],
        run_one: Callable[[str, dict], Any],
    ) -> None:
        self.registry = registry
        self.dependencies = dependencies
        self.run_one = run_one

    def run(self, context: dict) -> OrchestratorReport:
        report = OrchestratorReport()
        resolved: dict[str, bool] = context.get("_resolved_toggles", {})
        completed: set[str] = set()
        order = self._topological_order()

        for toggle_id in order:
            if not resolved.get(toggle_id, False):
                report.skipped.append(toggle_id)
                continue
            deps = self.dependencies.get(toggle_id, [])
            if any(d not in completed for d in deps if resolved.get(d, False)):
                report.skipped.append(toggle_id)
                continue
            try:
                self.run_one(toggle_id, context)
                completed.add(toggle_id)
                report.executed.append(toggle_id)
            except Exception as exc:
                report.failed.append({"id": toggle_id, "error": str(exc)})

        return report

    def _topological_order(self) -> list[str]:
        visited: set[str] = set()
        order: list[str] = []

        def visit(node: str) -> None:
            if node in visited:
                return
            visited.add(node)
            for dep in self.dependencies.get(node, []):
                visit(dep)
            order.append(node)

        for node in self.registry:
            visit(node)
        return order
