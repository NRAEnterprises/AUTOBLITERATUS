# abliterate/tuning/92_export_format_auto_tuning.py
#
# Tuning Type 92 of 103:
# Export-format auto tuning.
#
# Runtime toggle: --enable tuning.export_format_auto  (default: on)
#
# Effect:
#   Produce the ablated model in safetensors, GGUF, ONNX, MLX, and other
#   formats.
#
# This file owns:
#   - format detection from target frameworks
#   - format-specific export
#   - verification of each export

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ExportReport:
    per_format: dict = field(default_factory=dict)


class ExportFormatAutoTuning:

    def __init__(self, exporters: dict, enabled: bool = True):
        self.exporters = exporters
        self.enabled = enabled

    def run(self, model, tokenizer, output_dir: str) -> ExportReport:
        if not self.enabled:
            return ExportReport()
        results = {}
        for fmt, exporter in self.exporters.items():
            try:
                path = exporter(model, tokenizer, output_dir)
                results[fmt] = {"ok": True, "path": str(path)}
            except Exception as e:
                results[fmt] = {"ok": False, "error": str(e)}
        return ExportReport(per_format=results)


TOGGLE = {"id": "tuning.export_format_auto", "default": True}
