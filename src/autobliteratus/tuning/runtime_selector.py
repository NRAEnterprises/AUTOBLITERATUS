"""Runtime toggle selector used by the pipeline and tuning modules."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RuntimeSelector:
    toggles: dict[str, bool] = field(default_factory=dict)

    def is_enabled(self, toggle_id: str) -> bool:
        return bool(self.toggles.get(toggle_id, False))

    def require(self, toggle_id: str) -> None:
        if not self.is_enabled(toggle_id):
            raise RuntimeError(f"Toggle not enabled: {toggle_id}")
