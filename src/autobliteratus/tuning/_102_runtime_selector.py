"""Tuning type 102: runtime selector.

Holds the resolved toggle map for a single run. Every tuning file queries
this to know if it should run.
"""

from __future__ import annotations

from dataclasses import dataclass, field


TOGGLE = {"id": "tuning.runtime_selector", "default": True}


@dataclass
class RuntimeSelector:
    toggles: dict[str, bool] = field(default_factory=dict)

    def is_enabled(self, toggle_id: str) -> bool:
        return bool(self.toggles.get(toggle_id, False))

    def require(self, toggle_id: str) -> None:
        if not self.is_enabled(toggle_id):
            raise RuntimeError(f"Toggle not enabled: {toggle_id}")
