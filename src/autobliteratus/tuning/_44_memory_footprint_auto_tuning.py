"""Optional before-export memory telemetry; this stage does not optimize memory."""

from __future__ import annotations

from typing import Any

TOGGLE = {
    "id": "tuning.memory_footprint_auto",
    "default": False,
    "desc": "Measure process RSS and reported accelerator allocation before export; no optimization is implied.",
}


def _resident_bytes() -> int | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        return None


def run(adapter: Any, context: dict[str, Any]) -> dict[str, Any]:
    import torch

    model = adapter.handle.model
    device_names = {str(value) for value in (getattr(model, "hf_device_map", {}) or {}).values()}
    device_names.update(str(parameter.device) for parameter in model.parameters())
    measurements: dict[str, int | None] = {}
    for device_name in sorted(device_names - {"cpu", "meta"}):
        device_type = device_name.split(":", 1)[0]
        backend = getattr(torch, device_type, None)
        allocated = getattr(backend, "memory_allocated", None)
        no_argument_allocated = False
        if not callable(allocated) and device_type == "mps":
            allocated = getattr(backend, "current_allocated_memory", None)
            no_argument_allocated = True
        if callable(allocated):
            try:
                value = allocated() if no_argument_allocated else allocated(device_name)
                measurements[device_name] = int(value)
            except (RuntimeError, TypeError, ValueError):
                measurements[device_name] = None

    report = {
        "stage_id": "tuning.memory_footprint_auto",
        "schema_version": 1,
        "status": "measured",
        "measurement_point": "after selected transformations, before export",
        "host_rss_bytes": _resident_bytes(),
        "accelerator_allocated_bytes": measurements,
        "optimization_performed": False,
        "model_modified": False,
        "artifacts": [],
        "warnings": [
            "Memory telemetry is observational and runner-specific; unavailable readings are reported as null."
        ],
    }
    context["memory_report"] = report
    return report
