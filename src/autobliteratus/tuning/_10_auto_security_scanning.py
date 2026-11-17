"""Read-only model tensor anomaly checks; this is not malware detection."""

from __future__ import annotations

import math
from typing import Any

TOGGLE = {
    "id": "tuning.auto_security_scanning",
    "default": False,
    "desc": "Scan in-memory floating-point parameters for non-finite values and configurable extreme magnitudes.",
}


def run(adapter: Any, context: dict[str, Any]) -> dict[str, Any]:
    """Scan loaded model parameters for numeric anomalies without changing them."""
    import torch

    model = adapter.handle.model
    max_abs_threshold = float(context.get("security_scan_max_abs", 1_000.0))
    if not math.isfinite(max_abs_threshold) or max_abs_threshold <= 0:
        raise ValueError("security_scan_max_abs must be finite and positive.")

    findings: list[dict[str, Any]] = []
    scanned = 0
    skipped_non_float = 0
    skipped_unmaterialized = 0
    for name, parameter in model.named_parameters():
        tensor = parameter.detach()
        if tensor.device.type == "meta":
            skipped_unmaterialized += 1
            continue
        if not tensor.is_floating_point():
            skipped_non_float += 1
            continue
        scanned += 1
        flat = tensor.reshape(-1)
        nonfinite_count = 0
        max_abs = 0.0
        chunk_size = 1_000_000
        for start in range(0, flat.numel(), chunk_size):
            chunk = flat[start : start + chunk_size]
            finite = torch.isfinite(chunk)
            nonfinite_count += int((~finite).sum().item())
            finite_values = chunk[finite]
            if finite_values.numel():
                max_abs = max(max_abs, float(finite_values.abs().max().item()))
        if nonfinite_count:
            findings.append(
                {
                    "parameter": name,
                    "kind": "non_finite_parameter_values",
                    "severity": "critical",
                    "count": nonfinite_count,
                    "detail": "NaN or infinity found in an in-memory floating-point parameter.",
                }
            )
        if max_abs > max_abs_threshold:
            findings.append(
                {
                    "parameter": name,
                    "kind": "extreme_parameter_magnitude",
                    "severity": "warning",
                    "max_abs": max_abs,
                    "threshold": max_abs_threshold,
                    "detail": "Magnitude exceeded a configurable heuristic threshold; this is not proof of tampering.",
                }
            )

    report = {
        "status": "findings" if findings else "no_numeric_anomalies_detected",
        "scope": "materialized floating-point parameters only",
        "parameters_scanned": scanned,
        "parameters_skipped_non_float": skipped_non_float,
        "parameters_skipped_unmaterialized": skipped_unmaterialized,
        "max_abs_threshold": max_abs_threshold,
        "findings": findings,
        "model_modified": False,
        "limitations": [
            "This scan does not detect backdoors, trojans, poisoned training data, malicious code, or behavioral vulnerabilities.",
            "An extreme-magnitude warning is heuristic and may be expected for some models or parameterizations.",
            "A clean result means only that the tested numeric checks found no anomaly.",
        ],
    }
    context["security_scan_report"] = report
    return report
