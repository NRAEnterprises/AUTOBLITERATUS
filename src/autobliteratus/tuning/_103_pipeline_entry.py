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
        quantization=quantization,
        target_frameworks=target_frameworks,
        deployment_target=deployment_target,
        enable=enable,
        disable=disable,
        dry_run=dry_run,
    )
