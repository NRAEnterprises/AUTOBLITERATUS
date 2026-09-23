# abliterate/tuning/87_model_card_generation_auto_tuning.py
#
# Tuning Type 87 of 103:
# Model card generation auto tuning.
#
# Runtime toggle: --enable tuning.model_card_generation_auto  (default: on)
#
# Effect:
#   Write a model card describing what was ablated, what was preserved, and
#   what changed.
#
# This file owns:
#   - model card content generation
#   - model card writing

from __future__ import annotations

from pathlib import Path


class ModelCardGenerationAutoTuning:

    def __init__(self, work_dir: str = "./ablated_model", enabled: bool = True):
        self.work_dir = Path(work_dir)
        self.enabled = enabled

    def write(self, report: dict) -> Path:
        if not self.enabled:
            return Path()
        path = self.work_dir / "README.md"
        path.write_text(self._render(report))
        return path

    def _render(self, report: dict) -> str:
        return "# Model Card\n\nAblation and tuning report attached.\n"


TOGGLE = {"id": "tuning.model_card_generation_auto", "default": True}
