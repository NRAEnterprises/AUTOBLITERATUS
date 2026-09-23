"""Tuning type 89: audit trail auto tuning.

Produce a human-readable and machine-readable audit of everything that
happened.
"""

from __future__ import annotations

import json
from pathlib import Path


TOGGLE = {"id": "tuning.audit_trail_auto", "default": True}


class AuditTrailAutoTuning:
    def __init__(self, work_dir: str = "./ablated_model", enabled: bool = True) -> None:
        self.work_dir = Path(work_dir)
        self.enabled = enabled

    def write(self, events: list) -> None:
        if not self.enabled:
            return
        self.work_dir.mkdir(parents=True, exist_ok=True)
        (self.work_dir / "audit.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events))
        (self.work_dir / "audit.md").write_text(self._render_md(events))

    def _render_md(self, events: list) -> str:
        lines = ["# Audit Trail", ""]
        for e in events:
            lines.append(f"- {e.get('stage', '?')}: {e.get('action', '')}")
        return "\n".join(lines)
