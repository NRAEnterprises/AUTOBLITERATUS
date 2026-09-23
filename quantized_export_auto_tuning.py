# abliterate/tuning/93_quantized_export_auto_tuning.py
#
# Tuning Type 93 of 103:
# Quantized export auto tuning.
#
# Runtime toggle: --enable tuning.quantized_export_auto  (default: on)
#
# Effect:
#   Produce 4-bit and 8-bit exports with calibration verified.
#
# This file owns:
#   - quantized export path
#   - calibration verification per export

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class QuantExportReport:
    per_precision: dict = field(default_factory=dict)


class QuantizedExportAutoTuning:

    PRECISIONS = ("4bit", "8bit", "fp8")

    def __init__(self, export: Callable, verify: Callable, enabled: bool = True):
        self.export = export
        self.verify = verify
        self.enabled = enabled

    def run(self, model, tokenizer, output_dir: str) -> QuantExportReport:
        if not self.enabled:
            return QuantExportReport()
        per = {}
        for precision in self.PRECISIONS:
            path = self.export(model, tokenizer, output_dir, precision)
            ok = self.verify(path)
            per[precision] = {"path": str(path), "calibrated": ok}
        return QuantExportReport(per_precision=per)


TOGGLE = {"id": "tuning.quantized_export_auto", "default": True}
