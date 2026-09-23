"""Adapter layer between tuning files and core primitives."""

from __future__ import annotations

from autobliteratus.core.loader import load_model, ModelHandle
from autobliteratus.core.probe import ActivationProbe, ActivationSet
from autobliteratus.core.directions import compute_refusal_directions
from autobliteratus.core.projection import project_out
from autobliteratus.core.capability import measure_capability, AXES
from autobliteratus.core.refusal import measure_refusal_rate
from autobliteratus.core.cka import linear_cka
from autobliteratus.core.prompts import load_prompts
from autobliteratus.core.modules import LayerModuleAccess


class TuningAdapter:
    def __init__(self, handle):
        self.handle = handle
        self.modules = LayerModuleAccess(handle)
        self._probe_cache = {}

    @classmethod
    def from_path(cls, model_path, device="auto", dtype="auto"):
        handle = load_model(model_path, device=device, dtype=dtype)
        return cls(handle)

    def probe(self, prompts):
        key = tuple(prompts)
        if key in self._probe_cache:
            return self._probe_cache[key]
        result = ActivationProbe(self.handle).collect(prompts)
        self._probe_cache[key] = result
        return result

    def refusal_directions(self):
        harmful = self.probe(load_prompts("harmful"))
        harmless = self.probe(load_prompts("harmless"))
        return compute_refusal_directions(harmful, harmless)

    def project(self, layer_idx, direction, strength=1.0, component="attn.out"):
        return project_out(self.handle, layer_idx, direction, strength=strength, component=component)

    def capability(self, axes=AXES):
        return measure_capability(self.handle, axes=axes)

    def refusal_rate(self, prompts=None):
        prompts = prompts or load_prompts("harmful")
        return measure_refusal_rate(self.handle, prompts)

    def cka(self, a, b):
        return linear_cka(a, b)

    def prompts(self, name):
        return load_prompts(name)
