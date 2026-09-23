"""Tuning type 100: tool gateway auto tuning.

For models serving tool-gateway traffic, verify tool schemas, routing
metadata, and structured call integrity remain intact after ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


TOGGLE = {"id": "tuning.tool_gateway_auto", "default": True}


@dataclass
class ToolGatewayProfile:
    schema_valid: bool
    routing_valid: bool
    corrected: bool
    findings: list = field(default_factory=list)


class ToolGatewayAutoTuning:
    def __init__(self, measure: Callable, apply_correction: Callable, enabled: bool = True) -> None:
        self.measure = measure
        self.apply_correction = apply_correction
        self.enabled = enabled

    def run(self, model, tokenizer, gateway_schemas: list) -> ToolGatewayProfile:
        if not self.enabled:
            return ToolGatewayProfile(True, True, False)
        schema_valid, routing_valid, findings = self.measure(model, tokenizer, gateway_schemas)
        corrected = False
        if not schema_valid or not routing_valid:
            self.apply_correction(model, findings)
            corrected = True
        return ToolGatewayProfile(
            schema_valid=schema_valid,
            routing_valid=routing_valid,
            corrected=corrected,
            findings=findings,
        )
