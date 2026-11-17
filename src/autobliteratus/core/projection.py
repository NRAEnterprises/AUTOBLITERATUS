"""Projection: remove a refusal direction from a single layer's weights."""

from __future__ import annotations

import torch

from autobliteratus.core.modules import LayerModuleAccess


def _pick_target_module(layer, component):
    if component in ("attn.out", "attention", "self_attn"):
        for name in ("self_attn", "attention", "attn", "mixer", "self_attention"):
            if hasattr(layer, name):
                mod = getattr(layer, name)
                for out_name in ("o_proj", "out_proj", "dense", "proj", "c_proj", "wo"):
                    if hasattr(mod, out_name):
                        return getattr(mod, out_name)
                output = getattr(mod, "output", None)
                if output is not None and hasattr(output, "dense"):
                    return output.dense
        return None
    if component in ("mlp.down", "mlp", "ffn"):
        for name in ("mlp", "feed_forward", "ffn", "ff", "feedforward"):
            if hasattr(layer, name):
                mod = getattr(layer, name)
                for out_name in (
                    "down_proj", "fc2", "w2", "c_proj", "dense_4h_to_h", "wo", "down"
                ):
                    if hasattr(mod, out_name):
                        return getattr(mod, out_name)
        # OPT stores its two MLP projections directly on the decoder layer.
        for out_name in ("fc2", "dense_4h_to_h"):
            if hasattr(layer, out_name):
                return getattr(layer, out_name)
        return None
    return None


def project_out(handle, layer_idx, direction, strength=1.0, component="attn.out"):
    """Project a vector out of one supported linear weight matrix.

    The rank-one form avoids constructing a hidden-size-squared projection
    matrix. ``True`` is returned only when at least one weight value changed.
    """
    if not 0.0 < float(strength) <= 1.0:
        raise ValueError("strength must be in the interval (0, 1].")

    layers = LayerModuleAccess(handle)._layers
    if not layers:
        return False

    if layer_idx < 0 or layer_idx >= len(layers):
        return False

    module = _pick_target_module(layers[layer_idx], component)
    if module is None or not hasattr(module, "weight"):
        return False

    weight = module.weight
    vector = direction.to(device=weight.device, dtype=weight.dtype).reshape(-1)
    norm = vector.norm()
    if not torch.isfinite(norm) or norm.item() < 1e-8:
        return False
    vector = vector / norm

    with torch.no_grad():
        if weight.shape[0] == vector.shape[0]:
            removed = torch.outer(vector, vector @ weight)
        elif weight.shape[1] == vector.shape[0]:
            removed = torch.outer(weight @ vector, vector)
        else:
            return False

        if not torch.isfinite(removed).all() or not torch.count_nonzero(removed).item():
            return False
        weight.sub_(float(strength) * removed)

    return True
