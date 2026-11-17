"""Reach into a loaded model to get attention, MLP, embedding, and MoE blocks."""

from __future__ import annotations


class LayerModuleAccess:
    def __init__(self, handle):
        self.handle = handle
        self.model = handle.model
        self._layers = self._find_layers()

    def _find_layers(self):
        m = self.model
        # Common Hugging Face layouts across decoder-only, encoder-decoder,
        # and hybrid architectures. Order matters: prefer the text decoder to
        # a vision/audio tower when the model is multimodal.
        paths = (
            "model.language_model.layers",
            "language_model.model.layers",
            "model.decoder.layers",
            "decoder.layers",
            "model.decoder.block",
            "decoder.block",
            "transformer.h",
            "transformer.blocks",
            "transformer.layers",
            "transformer.block",
            "gpt_neox.layers",
            "model.layers",
            "model.blocks",
            "layers",
            "blocks",
        )
        for path in paths:
            obj = m
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                try:
                    blocks = list(obj)
                except TypeError:
                    continue
                if blocks and hasattr(blocks[0], "register_forward_hook"):
                    return blocks

        # Some models expose their decoder only through get_decoder(), with no
        # stable public path. Avoid guessing from state-dict names.
        get_decoder = getattr(m, "get_decoder", None)
        if callable(get_decoder):
            decoder = get_decoder()
            for attr in ("layers", "block", "h", "blocks"):
                blocks = getattr(decoder, attr, None)
                if blocks is not None:
                    try:
                        block_list = list(blocks)
                    except TypeError:
                        continue
                    if block_list:
                        return block_list

        for attr in ("model", "transformer", "gpt_neox", "decoder", "language_model"):
            if hasattr(m, attr):
                inner = getattr(m, attr)
                if hasattr(inner, "layers"):
                    return list(inner.layers)
                if hasattr(inner, "h"):
                    return list(inner.h)
                if hasattr(inner, "blocks"):
                    return list(inner.blocks)
                if hasattr(inner, "block"):
                    return list(inner.block)
        if hasattr(m, "layers"):
            return list(m.layers)
        return []

    def count(self):
        return len(self._layers)

    def block(self, idx):
        return self._layers[idx]

    def attention(self, idx):
        block = self._layers[idx]
        for name in ("self_attn", "attention", "attn", "mixer", "self_attention"):
            if hasattr(block, name):
                return getattr(block, name)
        return None

    def mlp(self, idx):
        block = self._layers[idx]
        for name in ("mlp", "feed_forward", "ffn", "ff", "feedforward"):
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
