# abliterate/tuning/89_audit_trail_auto_tuning.py
#
# Tuning Type 89 of 103:
# Audit trail auto tuning.
#
# Runtime toggle: --enable tuning.audit_trail_auto  (default: on)
#
# Effect:
#   Produce a human-readable and machine-readable audit of everything that
#   happened.
#
# This file owns:
#   - audit trail assembly
#   - human-readable rendering
#   - machine-readable rendering

from __future__ import annotations

from pathlib import Path
import json


class AuditTrailAutoTuning:

    def __init__(self, work_dir: str = "./ablated_model", enabled: bool = True):
        self.work_dir = Path(work_dir)
        self.enabled = enabled

    def write(self, events: list):
        if not self.enabled:
            return
        (self.work_dir / "audit.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events))
        (self.work_dir / "audit.md").write_text(self._render_md(events))

    def _render_md(self, events: list) -> str:
        lines = ["# Audit Trail", ""]
        for e in events:
            lines.append(f"- {e.get('stage', '?')}: {e.get('action', '')}")
        return "\n".join(lines)


TOGGLE = {"id": "tuning.audit_trail_auto", "default": True}
