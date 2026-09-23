# abliterate/tuning/102_runtime_selector.py
#
# Tuning Type 102 of 103:
# Runtime selector.
#
# Runtime toggle: not applicable. This is the runtime state.
#
# Effect:
#   Holds the resolved toggle map for a single run. Every tuning file queries
#   this to know if it should run. If its toggle is off, it returns a no-op
#   report immediately and does nothing.
#
# This file owns:
#   - the runtime toggle map
#   - the query interface for each tuning file

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RuntimeSelector:
    toggles: dict = field(default_factory=dict)

    def is_enabled(self, toggle_id: str) -> bool:
        return bool(self.toggles.get(toggle_id, False))

    def require(self, toggle_id: str):
        if not self.is_enabled(toggle_id):
            raise RuntimeError(f"Toggle not enabled: {toggle_id}")


TOGGLE = {"id": "tuning.runtime_selector", "default": True, "always_on": True}
