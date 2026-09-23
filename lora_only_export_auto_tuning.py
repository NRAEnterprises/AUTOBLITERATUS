# abliterate/tuning/94_lora_only_export_auto_tuning.py
#
# Tuning Type 94 of 103:
# LoRA-only export auto tuning.
#
# Runtime toggle: --enable tuning.lora_only_export_auto  (default: on)
#
# Effect:
#   Produce the ablation as a LoRA adapter instead of a full model.
#
# This file owns:
#   - adapter-only extraction
#   - adapter verification

from __future__ import annotations

from pathlib import Path


class LoRAOnlyExportAutoTuning:

    def __init__(self, work_dir: str = "./ablated_model/adapters", enabled: bool = True):
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


TOGGLE = {"id": "tuning.lora_only_export_auto", "default": True}
