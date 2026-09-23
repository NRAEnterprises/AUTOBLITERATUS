"""Projection: remove a refusal direction from a single layer's weights."""

from __future__ import annotations

import torch


def _pick_target_module(layer, component):
    if component in ("attn.out", "attention", "self_attn"):
        for name in ("self_attn", "attention", "attn", "mixer"):
            if hasattr(layer, name):
                mod = getattr(layer, name)
                for out_name in ("o_proj", "out_proj", "dense", "proj", "c_proj"):
                    if hasattr(mod, out_name):
                        return getattr(mod, out_name)
        return None
    if component in ("mlp.down", "mlp", "ffn"):
        for name in ("mlp", "feed_forward", "ffn", "ff"):
            if hasattr(layer, name):
                mod = getattr(layer, name)
                for out_name in ("down_proj", "fc2", "w2", "c_proj"):
                    if hasattr(mod, out_name):
                        return getattr(mod, out_name)
        return None
    return None


def project_out(handle, layer_idx, direction, strength=1.0, component="attn.out"):
    layers = None
    m = handle.model
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        layers = m.model.layers
    elif hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        layers = m.transformer.h
    elif hasattr(m, "layers"):
        layers = m.layers
    if layers is None or layer_idx >= len(layers):
        return False

    module = _pick_target_module(layers[layer_idx], component)
    if module is None or not hasattr(module, "weight"):
        return False

    weight = module.weight
    d = direction.to(weight.device, weight.dtype)
    d = d / d.norm()

    with torch.no_grad():
        if weight.shape[0] == d.shape[0]:
            proj = torch.outer(d, d)
            weight.sub_(strength * (proj @ weight))
        elif weight.shape[1] == d.shape[0]:
            proj = torch.outer(d, d)
            weight.sub_(strength * (weight @ proj))
        else:
            return False

    return True
