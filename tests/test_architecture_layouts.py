from __future__ import annotations

import pytest
import torch

from autobliteratus.core.loader import ModelHandle
from autobliteratus.core.modules import LayerModuleAccess
from autobliteratus.core.projection import _pick_target_module, project_out


@pytest.mark.parametrize("model_kind", ["gpt2", "mistral", "qwen2", "opt", "gpt_neox", "bloom"])
def test_decoder_layers_and_projection_targets_are_discovered(model_kind: str) -> None:
    from transformers import (
        BloomConfig,
        BloomForCausalLM,
        GPT2Config,
        GPT2LMHeadModel,
        GPTNeoXConfig,
        GPTNeoXForCausalLM,
        MistralConfig,
        MistralForCausalLM,
        OPTConfig,
        OPTForCausalLM,
        Qwen2Config,
        Qwen2ForCausalLM,
    )

    if model_kind == "gpt2":
        config = GPT2Config(vocab_size=32, n_embd=16, n_layer=2, n_head=4, n_positions=32)
        model = GPT2LMHeadModel(config)
    elif model_kind == "mistral":
        config = MistralConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=32, sliding_window=32,
        )
        model = MistralForCausalLM(config)
    elif model_kind == "qwen2":
        config = Qwen2Config(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=32,
        )
        model = Qwen2ForCausalLM(config)
    elif model_kind == "opt":
        config = OPTConfig(
            vocab_size=32, hidden_size=16, ffn_dim=32, num_hidden_layers=2,
            num_attention_heads=4, max_position_embeddings=32,
        )
        model = OPTForCausalLM(config)
    elif model_kind == "gpt_neox":
        config = GPTNeoXConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=32,
        )
        model = GPTNeoXForCausalLM(config)
    else:
        config = BloomConfig(
            vocab_size=32, hidden_size=16, n_layer=2, n_head=4,
            hidden_dropout=0.0, attention_dropout=0.0,
        )
        model = BloomForCausalLM(config)

    handle = ModelHandle(
        model=model,
        tokenizer=None,
        path="in-memory-test-model",
        architecture=model.config.model_type,
    )
    access = LayerModuleAccess(handle)

    assert access.count() == 2
    for index in range(2):
        block = access.block(index)
        attention_output = _pick_target_module(block, "attn.out")
        mlp_down = _pick_target_module(block, "mlp.down")
        assert attention_output is not None, model_kind
        assert mlp_down is not None, model_kind
        assert attention_output.weight.ndim == 2
        assert mlp_down.weight.ndim == 2

    target = _pick_target_module(access.block(0), "attn.out")
    before = target.weight.detach().clone()
    direction = torch.arange(1, 17, dtype=torch.float32)
    assert project_out(handle, 0, direction, strength=0.5, component="attn.out") is True
    after = target.weight.detach()
    assert not torch.equal(before, after)
