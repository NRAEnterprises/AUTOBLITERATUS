"""Tuning type 94: LoRA-only export auto tuning.

Produce the ablation as a LoRA adapter instead of a full model.
"""

from __future__ import annotations

from pathlib import Path


TOGGLE = {"id": "tuning.lora_only_export_auto", "default": True}


class LoRAOnlyExportAutoTuning:
    def __init__(self, work_dir: str = "./ablated_model/adapters", enabled: bool = True) -> None:
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled

    def run(self, model, tokenizer) -> Path:
        if not self.enabled:
            return Path()
        out = self.work_dir / "abliteration_adapter"
        model.save_pretrained(out)
        tokenizer.save_pretrained(out)
        return out
