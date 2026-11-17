"""Pipeline entry that sits inside the tuning package.

Same behavior as autobliteratus.pipeline.run_pipeline. This file exists so
the tuning package itself can be invoked directly without importing the
outer package.
"""

from __future__ import annotations

from typing import Any

TOGGLE = {"id": "tuning.pipeline_entry", "default": True}


def run(
    model_path: str,
    work_dir: str = "./abliterate_workspace",
    quantization: str = "none",
    target_frameworks: list[str] | None = None,
    deployment_target: str = "server",
    toggles: dict[str, bool] | None = None,
    dry_run: bool = False,
    **kwargs: Any,
) -> dict:
    enable = [k for k, v in (toggles or {}).items() if v is True]
    disable = [k for k, v in (toggles or {}).items() if v is False]
    from autobliteratus.pipeline import run_pipeline
    return run_pipeline(
        model_path=model_path,
        work_dir=work_dir,
        output_dir=kwargs.get("output_dir"),
        quantization=quantization,
        target_frameworks=target_frameworks,
        deployment_target=deployment_target,
        enable=enable,
        disable=disable,
        allow_mutation=kwargs.get("allow_mutation", False),
        dry_run=dry_run,
        ablation_strength=kwargs.get("ablation_strength", 1.0),
        ablation_component=kwargs.get("ablation_component", "attn.out"),
        trust_remote_code=kwargs.get("trust_remote_code", False),
        calibration_data=kwargs.get("calibration_data"),
        calibration_samples=kwargs.get("calibration_samples", 128),
        calibration_max_length=kwargs.get("calibration_max_length", 256),
        calibration_momentum=kwargs.get("calibration_momentum", 0.9),
        calibration_streamline=kwargs.get("calibration_streamline", True),
        device=kwargs.get("device", "auto"),
        dtype=kwargs.get("dtype", "auto"),
    )
