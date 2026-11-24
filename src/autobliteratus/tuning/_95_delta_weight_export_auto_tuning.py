"""Tuning type 95: delta-weight export prototype.

The direct-call API delegates to a verified artifact utility. This module has
no module-level ``run(adapter, context)`` function and remains unavailable to the
runtime pipeline until its complete governance and mutation contract is promoted.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

TOGGLE = {
    "id": "tuning.delta_weight_export_auto",
    "default": False,
    "desc": "Verified report-only delta artifact prototype; not pipeline-enabled.",
}


class DeltaWeightExportAutoTuning:
    def __init__(self, work_dir: str = "./ablated_model/delta", enabled: bool = False) -> None:
        self.work_dir = Path(work_dir)
        self.enabled = enabled

    def run(
        self,
        base_model: Any,
        ablated_model: Any,
        *,
        model_identity: Mapping[str, Any] | None = None,
    ) -> Path:
        if not self.enabled:
            return Path()
        if model_identity is None:
            raise ValueError("An explicit model_identity is required for delta export.")
        from autobliteratus.tuning.delta_weight_export_auto_tuning import (
            export_weight_delta,
        )

        report = export_weight_delta(
            base_model,
            ablated_model,
            self.work_dir,
            model_identity=model_identity,
        )
        return Path(report["delta_file"])
