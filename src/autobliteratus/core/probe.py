"""Activation probe: collect per-layer hidden states for a prompt set."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ActivationSet:
    per_layer: dict = field(default_factory=dict)
    prompts: list = field(default_factory=list)
    num_layers: int = 0
    hidden_size: int = 0

    def mean(self, layer_idx):
        tensors = self.per_layer.get(layer_idx, [])
        if not tensors:
            return None
        return torch.stack(tensors).mean(dim=0)

    def norm(self, layer_idx):
        m = self.mean(layer_idx)
        return float(m.norm().item()) if m is not None else 0.0


class ActivationProbe:
    def __init__(self, handle):
        self.handle = handle
        self.model = handle.model
        self.tokenizer = handle.tokenizer
        self._layers = None

    def _resolve_layers(self):
        if self._layers is not None:
            return self._layers
        m = self.model
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            self._layers = m.model.layers
        elif hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            self._layers = m.transformer.h
        elif hasattr(m, "layers"):
            self._layers = m.layers
        else:
            self._layers = []
        return self._layers

    def collect(self, prompts, max_length=256):
        layers = self._resolve_layers()
        num_layers = len(layers)
        per_layer = {i: [] for i in range(num_layers)}
        captured = {}

        def make_hook(idx):
            def hook(module, inputs, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                captured[idx] = hidden.detach()
            return hook

        handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]

        try:
            for prompt in prompts:
                captured.clear()
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                ).to(self.model.device)
                with torch.no_grad():
                    self.model(**inputs)
                for i in range(num_layers):
                    if i in captured:
                        h = captured[i]
                        pooled = h.mean(dim=1).squeeze(0).float().cpu()
                        per_layer[i].append(pooled)
        finally:
            for h in handles:
                h.remove()

        hidden_size = 0
        for i in range(num_layers):
            if per_layer[i]:
                hidden_size = per_layer[i][0].shape[-1]
                break

        return ActivationSet(
            per_layer=per_layer,
            prompts=list(prompts),
            num_layers=num_layers,
            hidden_size=hidden_size,
        )
