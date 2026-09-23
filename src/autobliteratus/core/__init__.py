"""AUTOBLITERATUS core: the ablation and measurement layer."""

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
from autobliteratus.core.adapters import TuningAdapter

__all__ = [
    "load_model", "ModelHandle",
    "ActivationProbe", "ActivationSet",
    "compute_refusal_directions", "project_out",
    "measure_capability", "AXES", "measure_refusal_rate",
    "linear_cka", "load_prompts",
    "LayerModuleAccess", "TuningAdapter",
]
