"""Apply the package's refusal-direction projection to the loaded model."""

from __future__ import annotations

import math
from typing import Any

TOGGLE = {
    "id": "tuning.refusal_direction_ablation",
    "default": True,
    "desc": "Project estimated refusal directions out of supported layer weights.",
}


def run(adapter: Any, context: dict[str, Any]) -> dict[str, Any]:
    strength = float(context.get("ablation_strength", 1.0))
    if not math.isfinite(strength) or not 0.0 < strength <= 1.0:
        raise ValueError("ablation_strength must be finite and in the interval (0, 1].")

    component = str(context.get("ablation_component", "attn.out"))
    if component not in {"attn.out", "mlp.down"}:
        raise ValueError("ablation_component must be 'attn.out' or 'mlp.down'.")

    refusal = adapter.refusal_directions()
    directions = refusal.get("directions", {})
    if not directions:
        raise RuntimeError("No refusal directions were estimated; no model edit was applied.")

    changed_layers: list[int] = []
    for layer_idx, direction in sorted(directions.items()):
        if adapter.project(
            layer_idx,
            direction,
            strength=strength,
            component=component,
        ):
            changed_layers.append(int(layer_idx))

    if not changed_layers:
        raise RuntimeError(
            f"No supported {component!r} weights were changed; refusing to export an unmodified model."
        )

    result = {
        "component": component,
        "strength": strength,
        "changed_layers": changed_layers,
        "estimated_layers": len(directions),
        "changed": True,
    }
    context["model_modified"] = True
    context["ablation_report"] = result
    return result
