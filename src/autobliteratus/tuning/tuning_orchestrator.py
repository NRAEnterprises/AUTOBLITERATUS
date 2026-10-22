"""Topological orchestration for AUTOBLITERATUS tuning stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class OrchestratorReport:
    executed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[dict[str, Any]] = field(default_factory=list)


class TuningOrchestrator:
    def __init__(
        self,
        registry: dict[str, dict[str, Any]],
        dependencies: dict[str, list[str]],
        run_one: Callable[[str, dict[str, Any]], Any],
    ) -> None:
        self.registry = registry
        self.dependencies = dependencies
        self.run_one = run_one

    def run(self, context: dict[str, Any]) -> OrchestratorReport:
        report = OrchestratorReport()
        resolved: dict[str, bool] = context.get("_resolved_toggles", {})
        completed: set[str] = set()
        order = self._topological_order()

        for toggle_id in order:
            if not resolved.get(toggle_id, False):
                report.skipped.append(toggle_id)
                continue

            deps = self.dependencies.get(toggle_id, [])
            if any(dep in self.registry and dep not in completed for dep in deps if resolved.get(dep, False)):
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
        active: set[str] = set()

        def visit(node: str) -> None:
            if node in active:
                raise RuntimeError(f"Dependency cycle detected at toggle '{node}'")
            if node in visited:
                return
            active.add(node)
            for dep in self.dependencies.get(node, []):
                if dep in self.registry:
                    visit(dep)
            active.remove(node)
            visited.add(node)
            order.append(node)

        for node in self.registry:
            visit(node)
        return order
