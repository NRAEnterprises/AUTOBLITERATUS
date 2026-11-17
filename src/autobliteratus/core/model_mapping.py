"""Conservative structural mapping inventory for supported model types.

This registry is an inventory and validation surface, not permission to mutate
weights. Every category is explicitly labeled as verified, conditional, or
unsupported. Runtime code must validate the actual module and state-dict paths
before using a mapping; an entry never implies that every parameter is covered.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

AREA_CATEGORIES = (
    "layer_stack",
    "attention",
    "feed_forward_or_mixer",
    "embedding",
    "lm_head",
    "normalization",
    "position_or_recurrent_state",
    "kv_cache",
    "routing",
    "input_output_projection",
)


@dataclass(frozen=True)
class AreaMapping:
    status: str
    paths: tuple[str, ...]
    notes: str


@dataclass(frozen=True)
class ModelMapping:
    model_type: str
    family: str
    areas: Mapping[str, AreaMapping]
    runtime_checks: tuple[str, ...]


def _area(status: str, paths: tuple[str, ...] = (), notes: str = "") -> AreaMapping:
    if status not in {"verified", "conditional", "unsupported"}:
        raise ValueError(f"Unknown mapping status: {status}")
    return AreaMapping(status=status, paths=paths, notes=notes)


def _mapping(
    model_type: str,
    family: str,
    *,
    verified: Mapping[str, tuple[str, ...]],
    conditional: Mapping[str, tuple[str, ...]] = (),
    notes: str = "Runtime module and state-dict validation is required before use.",
) -> ModelMapping:
    areas = {
        category: _area("unsupported", notes="No safe mapping asserted for this category.")
        for category in AREA_CATEGORIES
    }
    for category, paths in dict(verified).items():
        areas[category] = _area("verified", paths, notes)
    for category, paths in dict(conditional).items():
        areas[category] = _area("conditional", paths, "Candidate paths require runtime confirmation.")
    return ModelMapping(model_type, family, areas, ("config_model_type", "module_paths", "state_dict_shapes"))


MODEL_MAPPINGS: dict[str, ModelMapping] = {
    "gpt2": _mapping("gpt2", "dense_transformer", verified={
        "layer_stack": ("transformer.h",), "attention": ("transformer.h[].attn.c_proj",),
        "feed_forward_or_mixer": ("transformer.h[].mlp.c_proj",), "embedding": ("transformer.wte", "transformer.wpe"),
        "lm_head": ("lm_head",), "normalization": ("transformer.h[].ln_1", "transformer.h[].ln_2"),
        "position_or_recurrent_state": ("transformer.wpe",), "kv_cache": ("generate(use_cache=...)",),
        "input_output_projection": ("lm_head",),
    }),
    "gpt_neo": _mapping("gpt_neo", "dense_transformer", verified={
        "layer_stack": ("transformer.h",), "attention": ("transformer.h[].attn.attention",),
        "feed_forward_or_mixer": ("transformer.h[].mlp",), "embedding": ("transformer.wte", "transformer.wpe"),
        "lm_head": ("lm_head",), "kv_cache": ("generate(use_cache=...)",),
    }, conditional={"normalization": ("transformer.h[].ln_1", "transformer.h[].ln_2")}),
    "gpt_neox": _mapping("gpt_neox", "dense_transformer", verified={
        "layer_stack": ("gpt_neox.layers",), "attention": ("gpt_neox.layers[].attention.dense",),
        "feed_forward_or_mixer": ("gpt_neox.layers[].mlp.dense_4h_to_h",), "embedding": ("gpt_neox.embed_in",),
        "lm_head": ("embed_out",), "normalization": ("gpt_neox.layers[].input_layernorm", "gpt_neox.final_layer_norm"),
        "kv_cache": ("generate(use_cache=...)",), "input_output_projection": ("embed_out",),
    }),
    "opt": _mapping("opt", "dense_transformer", verified={
        "layer_stack": ("model.decoder.layers",), "attention": ("model.decoder.layers[].self_attn.out_proj",),
        "feed_forward_or_mixer": ("model.decoder.layers[].fc2",), "embedding": ("model.decoder.embed_tokens", "model.decoder.embed_positions"),
        "lm_head": ("lm_head",), "normalization": ("model.decoder.layers[].self_attn_layer_norm", "model.decoder.layers[].final_layer_norm"),
        "kv_cache": ("generate(use_cache=...)",), "input_output_projection": ("lm_head",),
    }),
    "bloom": _mapping("bloom", "dense_transformer", verified={
        "layer_stack": ("transformer.h",), "attention": ("transformer.h[].self_attention.dense",),
        "feed_forward_or_mixer": ("transformer.h[].mlp.dense_4h_to_h",), "embedding": ("transformer.word_embeddings",),
        "lm_head": ("lm_head",), "normalization": ("transformer.h[].input_layernorm", "transformer.ln_f"),
        "kv_cache": ("generate(use_cache=...)",), "input_output_projection": ("lm_head",),
    }),
    "falcon": _mapping("falcon", "dense_transformer", verified={
        "layer_stack": ("transformer.h",), "attention": ("transformer.h[].self_attention",),
        "feed_forward_or_mixer": ("transformer.h[].mlp",), "embedding": ("transformer.word_embeddings",),
        "lm_head": ("lm_head",), "kv_cache": ("generate(use_cache=...)",),
    }, conditional={"normalization": ("transformer.h[].input_layernorm",)}),
    "mpt": _mapping("mpt", "dense_transformer", verified={
        "layer_stack": ("transformer.blocks",), "embedding": ("transformer.wte",),
        "lm_head": ("transformer.lm_f",), "kv_cache": ("generate(use_cache=...)",),
    }, conditional={"attention": ("transformer.blocks[].attn",), "feed_forward_or_mixer": ("transformer.blocks[].ffn",)}),
    "llama": _mapping("llama", "dense_transformer", verified={
        "layer_stack": ("model.layers",), "attention": ("model.layers[].self_attn.o_proj",),
        "feed_forward_or_mixer": ("model.layers[].mlp.down_proj",), "embedding": ("model.embed_tokens",),
        "lm_head": ("lm_head",), "normalization": ("model.layers[].input_layernorm", "model.norm"),
        "position_or_recurrent_state": ("model.layers[].self_attn.rotary_emb",), "kv_cache": ("generate(use_cache=...)",),
        "input_output_projection": ("lm_head",),
    }),
    "mistral": _mapping("mistral", "dense_transformer", verified={
        "layer_stack": ("model.layers",), "attention": ("model.layers[].self_attn.o_proj",),
        "feed_forward_or_mixer": ("model.layers[].mlp.down_proj",), "embedding": ("model.embed_tokens",),
        "lm_head": ("lm_head",), "normalization": ("model.layers[].input_layernorm", "model.norm"),
        "position_or_recurrent_state": ("model.layers[].self_attn.rotary_emb",), "kv_cache": ("generate(use_cache=...)",),
        "input_output_projection": ("lm_head",),
    }),
    "qwen2": _mapping("qwen2", "dense_transformer", verified={
        "layer_stack": ("model.layers",), "attention": ("model.layers[].self_attn.o_proj",),
        "feed_forward_or_mixer": ("model.layers[].mlp.down_proj",), "embedding": ("model.embed_tokens",),
        "lm_head": ("lm_head",), "normalization": ("model.layers[].input_layernorm", "model.norm"),
        "position_or_recurrent_state": ("model.layers[].self_attn.rotary_emb",), "kv_cache": ("generate(use_cache=...)",),
        "input_output_projection": ("lm_head",),
    }),
    "qwen3": _mapping("qwen3", "dense_transformer", verified={
        "layer_stack": ("model.layers",), "attention": ("model.layers[].self_attn.o_proj",),
        "feed_forward_or_mixer": ("model.layers[].mlp.down_proj",), "embedding": ("model.embed_tokens",),
        "lm_head": ("lm_head",), "normalization": ("model.layers[].input_layernorm", "model.norm"),
        "position_or_recurrent_state": ("model.layers[].self_attn.rotary_emb",), "kv_cache": ("generate(use_cache=...)",),
        "input_output_projection": ("lm_head",),
    }),
    "phi": _mapping("phi", "dense_transformer", verified={
        "layer_stack": ("transformer.h",), "embedding": ("transformer.embd.wte",),
        "lm_head": ("lm_head",), "kv_cache": ("generate(use_cache=...)",),
    }, conditional={"attention": ("transformer.h[].mixer",), "feed_forward_or_mixer": ("transformer.h[].mlp",)}),
    "gemma": _mapping("gemma", "dense_transformer", verified={
        "layer_stack": ("model.layers",), "attention": ("model.layers[].self_attn.o_proj",),
        "feed_forward_or_mixer": ("model.layers[].mlp.down_proj",), "embedding": ("model.embed_tokens",),
        "lm_head": ("lm_head",), "normalization": ("model.layers[].input_layernorm", "model.norm"),
        "position_or_recurrent_state": ("model.layers[].self_attn.rotary_emb",), "kv_cache": ("generate(use_cache=...)",),
    }),
    "gemma2": _mapping("gemma2", "dense_transformer", verified={
        "layer_stack": ("model.layers",), "attention": ("model.layers[].self_attn.o_proj",),
        "feed_forward_or_mixer": ("model.layers[].mlp.down_proj",), "embedding": ("model.embed_tokens",),
        "lm_head": ("lm_head",), "normalization": ("model.layers[].input_layernorm", "model.norm"),
        "position_or_recurrent_state": ("model.layers[].self_attn.rotary_emb",), "kv_cache": ("generate(use_cache=...)",),
    }),
    "phi3": _mapping("phi3", "dense_transformer", verified={
        "layer_stack": ("model.layers",), "embedding": ("model.embed_tokens",),
        "lm_head": ("lm_head",), "kv_cache": ("generate(use_cache=...)",),
    }, conditional={"attention": ("model.layers[].self_attn",), "feed_forward_or_mixer": ("model.layers[].mlp",)}),
    "olmo": _mapping("olmo", "dense_transformer", verified={
        "layer_stack": ("model.transformer.blocks",), "embedding": ("model.transformer.wte",),
        "lm_head": ("lm_head",), "kv_cache": ("generate(use_cache=...)",),
    }, conditional={"attention": ("model.transformer.blocks[].attn",), "feed_forward_or_mixer": ("model.transformer.blocks[].ffn",)}),
    "olmo2": _mapping("olmo2", "dense_transformer", verified={
        "layer_stack": ("model.layers",), "attention": ("model.layers[].self_attn.o_proj",),
        "feed_forward_or_mixer": ("model.layers[].mlp.down_proj",), "embedding": ("model.embed_tokens",),
        "lm_head": ("lm_head",), "kv_cache": ("generate(use_cache=...)",),
    }),
    "stablelm": _mapping("stablelm", "dense_transformer", verified={
        "layer_stack": ("model.layers",), "embedding": ("model.embed_tokens",),
        "lm_head": ("lm_head",), "kv_cache": ("generate(use_cache=...)",),
    }, conditional={"attention": ("model.layers[].self_attn",), "feed_forward_or_mixer": ("model.layers[].mlp",)}),
    "starcoder2": _mapping("starcoder2", "dense_transformer", verified={
        "layer_stack": ("model.layers",), "attention": ("model.layers[].self_attn.o_proj",),
        "feed_forward_or_mixer": ("model.layers[].mlp.down_proj",), "embedding": ("model.embed_tokens",),
        "lm_head": ("lm_head",), "kv_cache": ("generate(use_cache=...)",),
    }),
    "mixtral": _mapping("mixtral", "moe_sparse", verified={
        "layer_stack": ("model.layers",), "attention": ("model.layers[].self_attn.o_proj",),
        "feed_forward_or_mixer": ("model.layers[].block_sparse_moe.experts[].w2",), "embedding": ("model.embed_tokens",),
        "lm_head": ("lm_head",), "kv_cache": ("generate(use_cache=...)",), "routing": ("model.layers[].block_sparse_moe.gate",),
    }),
    "deepseek_v2": _mapping("deepseek_v2", "moe_shared_routed", verified={
        "layer_stack": ("model.layers",), "embedding": ("model.embed_tokens",), "lm_head": ("lm_head",),
        "kv_cache": ("generate(use_cache=...)",), "routing": ("model.layers[].mlp.gate",),
    }, conditional={"attention": ("model.layers[].self_attn",), "feed_forward_or_mixer": ("model.layers[].mlp",)}),
    "deepseek_v3": _mapping("deepseek_v3", "moe_shared_routed", verified={
        "layer_stack": ("model.layers",), "embedding": ("model.embed_tokens",), "lm_head": ("lm_head",),
        "kv_cache": ("generate(use_cache=...)",), "routing": ("model.layers[].mlp.gate",),
    }, conditional={"attention": ("model.layers[].self_attn",), "feed_forward_or_mixer": ("model.layers[].mlp",)}),
    "mamba": _mapping("mamba", "ssm", verified={
        "layer_stack": ("backbone.layers",), "embedding": ("backbone.embeddings",),
        "lm_head": ("lm_head",), "position_or_recurrent_state": ("backbone.layers[].mixer",),
    }, notes="State-space recurrence is not a KV cache; cache profiling must use the model's explicit recurrent state API."),
    "rwkv": _mapping("rwkv", "rwkv", verified={
        "layer_stack": ("rwkv.blocks",), "embedding": ("rwkv.embeddings",), "lm_head": ("head",),
        "position_or_recurrent_state": ("rwkv.blocks[].attention.time_mix", "rwkv.blocks[].feed_forward.time_mix"),
    }, notes="RWKV recurrent state is not a transformer KV cache and requires a separate state contract."),
    "retnet": _mapping("retnet", "retnet", verified={
        "layer_stack": ("model.layers",), "embedding": ("model.embed_tokens",), "lm_head": ("lm_head",),
        "position_or_recurrent_state": ("model.layers[].retention",),
    }, notes="Retention state is not assumed equivalent to a transformer KV cache."),
    "jamba": _mapping("jamba", "hybrid_attn_ssm", verified={
        "layer_stack": ("model.layers",), "embedding": ("model.embed_tokens",), "lm_head": ("lm_head",),
        "kv_cache": ("generate(use_cache=...)",), "position_or_recurrent_state": ("model.layers[].mixer",),
    }, conditional={"attention": ("model.layers[].self_attn",), "feed_forward_or_mixer": ("model.layers[].mlp",)}),
}


def get_model_mapping(model_type: str) -> ModelMapping | None:
    """Return the conservative mapping for a configured model type."""
    return MODEL_MAPPINGS.get(str(model_type).lower())


def mapping_is_complete(mapping: ModelMapping) -> bool:
    """Return whether every declared category has an explicit status."""
    return set(mapping.areas) == set(AREA_CATEGORIES) and all(
        area.status in {"verified", "conditional", "unsupported"} for area in mapping.areas.values()
    )
