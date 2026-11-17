"""Atomic, prompt-redacted audit artifacts for completed pipeline runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOGGLE = {
    "id": "tuning.audit_trail_auto",
    "default": False,
    "desc": "Write atomic JSONL/Markdown audit artifacts with stage outcomes and prompt-redacted run metadata.",
}


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = stream.name
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _safe_report(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    safe: dict[str, Any] = {}
    for key in (
        "status",
        "metrics",
        "baseline_metrics",
        "post_metrics",
        "regressions",
        "tolerance_ratio",
        "tolerances",
        "sample_count",
        "scored_tokens",
        "evaluation_max_length",
    ):
        if key in value:
            safe[key] = value[key]
    return safe or None


def run(adapter: Any, context: dict[str, Any]) -> dict[str, Any]:
    """Mark audit finalization as requested; pipeline writes after all stages finish."""
    context["_audit_requested"] = True
    return {"status": "pending_finalization", "model_modified": False}


def finalize_pipeline_audit(adapter: Any, context: dict[str, Any], report: Any) -> dict[str, Any]:
    """Persist completed stage statuses, safe metric summaries, and runtime provenance."""
    run_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()
    model_path = str(context.get("model_path", ""))
    source_reference_hash = hashlib.sha256(model_path.encode("utf-8")).hexdigest() if model_path else None
    handle = adapter.handle

    failed = []
    for entry in getattr(report, "failed", []):
        failed.append(
            {
                "stage": entry.get("id", "unknown"),
                "error_type": entry.get("error_type", "StageError"),
            }
        )

    output_dir = Path(context.get("work_dir", "./abliterate_workspace")).expanduser() / "audit_trails"
    jsonl_path = output_dir / f"audit-{run_id}.jsonl"
    markdown_path = output_dir / f"audit-{run_id}.md"
    try:
        import torch
        torch_version = str(torch.__version__)
    except ImportError:
        torch_version = "unavailable"
    try:
        import transformers
        transformers_version = str(transformers.__version__)
    except ImportError:
        transformers_version = "unavailable"

    export = context.get("export_report")
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": created_at,
        "model": {
            "source_reference_sha256": source_reference_hash,
            "architecture": str(getattr(handle, "architecture", "unknown")),
            "architecture_class": str(getattr(handle, "architecture_class", "unknown")),
            "device": str(getattr(handle, "device", "unknown")),
            "dtype": str(getattr(handle, "dtype", "unknown")),
        },
        "settings": {
            "quantization": context.get("quantization"),
            "deployment_target": context.get("deployment_target"),
            "target_frameworks": context.get("target_frameworks"),
            "ablation_component": context.get("ablation_component"),
            "ablation_strength": context.get("ablation_strength"),
            "trust_remote_code": bool(context.get("trust_remote_code", False)),
            "model_revision": context.get("model_revision"),
            "device": context.get("device"),
            "dtype": context.get("dtype"),
            "device_map": context.get("device_map"),
            "regression_metric_directions": context.get("regression_metric_directions"),
            "regression_tolerance_ratio": context.get("regression_tolerance_ratio", 0.02),
            "regression_tolerances": context.get("regression_tolerances", {}),
        },
        "stages": {
            "executed": list(getattr(report, "executed", [])),
            "skipped": list(getattr(report, "skipped", [])),
            "failed": failed,
        },
        "evaluation": {
            "baseline": _safe_report(context.get("baseline_report")),
            "regression_gate": _safe_report(context.get("benchmark_regression_report")),
        },
        "artifact": {
            "path": (export or {}).get("path") if isinstance(export, dict) else None,
            "format": (export or {}).get("format") if isinstance(export, dict) else None,
            "validated": (export or {}).get("validated") if isinstance(export, dict) else None,
            "weight_fingerprint": context.get("weight_fingerprint"),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch_version,
            "transformers": transformers_version,
        },
        "privacy": "Evaluation texts, prompts, raw exception messages, and arbitrary stage_options are not recorded.",
        "files": {"jsonl": str(jsonl_path), "markdown": str(markdown_path)},
    }
    json_line = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    lines = [
        "# AUTOBLITERATUS run audit",
        "",
        f"- Run ID: `{run_id}`",
        f"- Created (UTC): `{created_at}`",
        f"- Architecture: `{payload['model']['architecture']}` ({payload['model']['architecture_class']})",
        f"- Quantization: `{context.get('quantization', 'none')}`",
        f"- Artifact validated: `{payload['artifact']['validated']}`",
        "",
        "## Stage outcomes",
        "",
    ]
    for stage in payload["stages"]["executed"]:
        lines.append(f"- `{stage}`: executed")
    for stage in payload["stages"]["skipped"]:
        lines.append(f"- `{stage}`: skipped")
    for entry in payload["stages"]["failed"]:
        lines.append(f"- `{entry['stage']}`: failed (`{entry['error_type']}`)")
    lines.extend(
        [
            "",
            "## Evaluation summary",
            "",
            f"- Baseline: `{(payload['evaluation']['baseline'] or {}).get('status', 'not run')}`",
            f"- Regression gate: `{(payload['evaluation']['regression_gate'] or {}).get('status', 'not run')}`",
            "",
            "Evaluation texts and raw error messages are intentionally omitted.",
            "",
        ]
    )
    _atomic_write(jsonl_path, json_line)
    _atomic_write(markdown_path, "\n".join(lines))
    result = {
        "status": "written",
        "run_id": run_id,
        "jsonl_path": str(jsonl_path),
        "markdown_path": str(markdown_path),
        "failed_stage_count": len(failed),
        "model_modified": False,
    }
    context["audit_report"] = result
    return result
