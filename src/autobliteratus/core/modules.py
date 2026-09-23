"""Reach into a loaded model to get attention, MLP, embedding, and MoE blocks."""

from __future__ import annotations


class LayerModuleAccess:
    def __init__(self, handle):
        self.handle = handle
        self.model = handle.model
        self._layers = self._find_layers()

    def _find_layers(self):
        m = self.model
        for attr in ("model", "transformer", "gpt_neox"):
            if hasattr(m, attr):
                inner = getattr(m, attr)
                if hasattr(inner, "layers"):
                    return list(inner.layers)
                if hasattr(inner, "h"):
                    return list(inner.h)
        if hasattr(m, "layers"):
            return list(m.layers)
        return []

    def count(self):
        return len(self._layers)

    def block(self, idx):
        return self._layers[idx]

    def attention(self, idx):
        block = self._layers[idx]
        for name in ("self_attn", "attention", "attn", "mixer"):
            if hasattr(block, name):
                return getattr(block, name)
        return None

    def mlp(self, idx):
        block = self._layers[idx]
        for name in ("mlp", "feed_forward", "ffn", "ff"):
            if hasattr(block, name):
                return getattr(block, name)
        return None

    def moe_experts(self, idx):
        block = self._layers[idx]
        for name in ("block_sparse_moe", "moe", "experts"):
            if hasattr(block, name):
                return getattr(block, name)
        mlp = self.mlp(idx)
        if mlp is not None and hasattr(mlp, "experts"):
            return mlp.experts
        return None

    def embedding(self):
        m = self.model
        for path in ("model.embed_tokens", "transformer.wte", "embed_tokens", "wte"):
            obj = m
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                return obj
        return None

    def lm_head(self):
        for name in ("lm_head", "output", "embed_out"):
            if hasattr(self.model, name):
                return getattr(self.model, name)
        return None
