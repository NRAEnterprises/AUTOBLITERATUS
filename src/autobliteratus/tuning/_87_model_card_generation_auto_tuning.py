"""Tuning type 87: model card generation auto tuning.

Write a model card describing what was ablated, what was preserved, and
what changed.
"""

from __future__ import annotations

from pathlib import Path


TOGGLE = {"id": "tuning.model_card_generation_auto", "default": True}


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
        return "# Model Card\n\nAblation and tuning report attached.\n"
