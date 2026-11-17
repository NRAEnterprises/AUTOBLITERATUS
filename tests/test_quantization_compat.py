from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from autobliteratus.quantization_compat import (
    _AliasedStateDict,
    _output_weight_alias,
    _tied_output_weight_key,
)
from benchmarks.quantization_quality_matrix import InvalidMetricError, wikitext_metrics


def _tiny_gpt_neox():
    transformers = pytest.importorskip("transformers")
    config = transformers.GPTNeoXConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    return transformers.GPTNeoXForCausalLM(config)


def test_legacy_output_head_name_is_remapped_lazily() -> None:
    model = _tiny_gpt_neox()
    output = model.get_output_embeddings()
    output_module_name = next(name for name, module in model.named_modules() if module is output)
    expected_key = f"{output_module_name}.weight"
    value = torch.arange(output.weight.numel(), dtype=torch.float32).reshape(output.weight.shape)
    state = {"embed_out.weight": value}

    aliases = _output_weight_alias(model, state, {})
    assert aliases == {expected_key: "embed_out.weight"}
    remapped = _AliasedStateDict(state, aliases)
    assert expected_key in remapped
    assert "embed_out.weight" not in remapped
    assert torch.equal(remapped[expected_key], value)
    assert len(remapped) == 1


def test_ambiguous_output_head_alias_fails_closed() -> None:
    model = _tiny_gpt_neox()
    output = model.get_output_embeddings()
    value = torch.ones_like(output.weight)
    state = {"embed_out.weight": value, "output_layer.weight": value.clone()}

    with pytest.raises(RuntimeError, match="multiple possible output-head aliases"):
        _output_weight_alias(model, state, {})


def test_missing_tied_output_head_key_is_accepted_only_with_embedding_alias() -> None:
    transformers = pytest.importorskip("transformers")
    model = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            tie_word_embeddings=True,
        )
    )
    input_module = model.get_input_embeddings()
    output_module = model.get_output_embeddings()
    input_name = next(name for name, module in model.named_modules() if module is input_module)
    output_name = next(name for name, module in model.named_modules() if module is output_module)
    state = {f"{input_name}.weight": torch.ones_like(input_module.weight)}

    assert _tied_output_weight_key(model, state, {}) == f"{output_name}.weight"
    assert _tied_output_weight_key(model, {}, {}) is None
    assert _tied_output_weight_key(model, state, {output_name: {"weights": "qint8"}}) is None


def test_nonfinite_logits_are_classified_as_invalid_metrics() -> None:
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[1, 2]])}

    class Model:
        def parameters(self):
            return iter((torch.nn.Parameter(torch.zeros(1)),))

        def __call__(self, **kwargs):
            return SimpleNamespace(logits=torch.full((1, 2, 4), float("nan")))

    with pytest.raises(InvalidMetricError, match="non-finite model logits"):
        wikitext_metrics(Model(), Tokenizer(), ["two tokens"], max_length=8)
