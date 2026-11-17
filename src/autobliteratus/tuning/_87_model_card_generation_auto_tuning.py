"""Tuning type 87: model card generation auto tuning.

Write a model card describing what was ablated, what was preserved, and
what changed.
"""

from __future__ import annotations

from pathlib import Path

TOGGLE = {
    "id": "tuning.model_card_generation_auto",
    "default": False,
    "desc": "Generate an artifact model card from observed architecture, transformations, and validation metadata.",
}


class ModelCardGenerationAutoTuning:
    def __init__(self, work_dir: str = "./ablated_model", enabled: bool = True) -> None:
        self.work_dir = Path(work_dir)
        self.enabled = enabled

    def write(self, report: dict) -> Path:
        if not self.enabled:
            return Path()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        path = self.work_dir / "README.md"
        path.write_text(self._render(report))
        return path

    def _render(self, report: dict) -> str:
        return render_model_card(report)


def render_model_card(report: dict) -> str:
    architecture = report.get("architecture") or {}
    quantization = report.get("quantization") or {}
    export = report.get("export") or {}
    reproducibility = report.get("reproducibility") or {}
    runtime = reproducibility.get("runtime") or {}
    memory = report.get("memory") or {}
    lines = [
        "# Model Artifact Card",
        "",
        "This card describes the generated artifact and records checks performed by AUTOBLITERATUS.",
        "It does not certify task quality, safety, or suitability for deployment.",
        "",
        "## Source and detected structure",
        "",
        f"- Source: `{report.get('source_model', 'unknown')}`",
        f"- Model type: `{architecture.get('model_type', 'unknown')}`",
        f"- Transformers class: `{architecture.get('model_class', 'unknown')}`",
        f"- Detected family: `{architecture.get('family', 'unknown')}`",
        f"- Decoder blocks inspected: {architecture.get('block_count', 'unknown')}",
        "",
        "## Transformations and target",
        "",
        f"- Ablation: `{report.get('ablation') or 'not run'}`",
        f"- Quantization: `{quantization.get('mode', 'none')}`",
        f"- Export format: `{export.get('format', 'Transformers')}`",
        f"- Requested deployment profile: `{report.get('deployment_target', 'server')}`",
        f"- Requested target frameworks: `{', '.join(report.get('target_frameworks', ['transformers']))}`",
        f"- Observed device: `{runtime.get('loaded_device_summary', 'not recorded')}`",
        f"- Observed dtype: `{runtime.get('dtype', 'not recorded')}`",
        f"- Host RSS at measurement: `{memory.get('host_rss_bytes', 'not recorded')}` bytes",
        "- Runtime metadata is observational; no bitwise reproducibility or memory optimization is claimed.",
        "",
        "## Validation",
        "",
        f"- Export validation: `{bool(export.get('validated', False))}`",
        f"- Source/output paths disjoint: `{bool(export.get('source_output_disjoint', False))}`",
        "- Quality benchmarking is not implied by a successful artifact export.",
        "",
    ]
    return "\n".join(lines)


def run(adapter, context):
    report = {
        "source_model": str(context.get("model_path")),
        "architecture": context.get("architecture_report"),
        "ablation": context.get("ablation_report"),
        "quantization": context.get("quantization_report"),
        "reproducibility": context.get("reproducibility_report"),
        "memory": context.get("memory_report"),
        "deployment_target": context.get("deployment_target", "server"),
        "target_frameworks": list(context.get("target_frameworks", ["transformers"])),
        "export": context.get("export_report"),
    }
    context["model_card"] = render_model_card(report)
    return {"generated": True, "artifact_file": "README.md"}
