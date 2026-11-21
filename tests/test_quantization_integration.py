from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from autobliteratus.core.model_mapping_discovery import _model_identity
from autobliteratus.pipeline import run_pipeline
from autobliteratus.quantization_compat import load_quantized_model
from autobliteratus.tuning.mapping_promotion_gate import evaluate_mapping_promotion


def _promoted_test_capability(model: object) -> tuple[dict[str, object], dict[str, object]]:
    """Build deterministic evidence for this fixture's exact pre-mutation model."""
    identity = _model_identity(SimpleNamespace(handle=SimpleNamespace(model=model)))
    category = "quantization.integration"
    path = "fixture.model"
    experiment_id = hashlib.sha256(
        f"{identity['identity_sha256']}:{category}:{path}".encode()
    ).hexdigest()
    measurement_signature = hashlib.sha256(
        f"{experiment_id}:measured".encode()
    ).hexdigest()
    discovery = {
        "schema_version": 1,
        "status": "measured",
        "model_identity": identity,
        "promotions": [
            {
                "experiment_id": experiment_id,
                "category": category,
                "path": path,
                "status": "verified",
                "measurement_signature": measurement_signature,
            }
        ],
    }
    capability = evaluate_mapping_promotion(
        discovery,
        replay_results={
            experiment_id: {
                "status": "passed",
                "measurement_signature": measurement_signature,
            }
        },
        regression_report={
            "status": "passed",
            "regressions": [],
            "baseline_metrics": {"fixture_metric": 1.0},
            "post_metrics": {"fixture_metric": 1.0},
            "metrics": {"fixture_metric": {"regressed": False}},
        },
        expected_identity=identity,
    )
    assert capability["status"] == "promoted"
    return capability, identity


@pytest.mark.parametrize(
    ("model_kind", "hidden_size", "num_layers", "quantization"),
    [
        ("llama", 16, 1, "quanto-int8"),
        ("llama", 16, 1, "quanto-int4"),
        ("llama", 16, 1, "quanto-int8-weight-only"),
        ("llama", 16, 1, "quanto-int4-weight-only"),
        ("llama", 32, 2, "quanto-int8"),
        ("llama", 32, 2, "quanto-int4"),
        ("mistral", 16, 1, "quanto-int8"),
        ("mistral", 16, 1, "quanto-int4"),
        ("qwen2", 16, 1, "quanto-int8"),
        ("qwen2", 16, 1, "quanto-int4"),
        ("gpt_neox", 16, 1, "quanto-int8-weight-only"),
        ("gpt2", 16, 1, "quanto-int8"),
        ("gpt2", 16, 1, "quanto-int4"),
        ("gpt2", 16, 1, "quanto-int8-weight-only"),
        ("gpt2", 16, 1, "quanto-int4-weight-only"),
        ("gpt2", 32, 2, "quanto-int8"),
        ("gpt2", 32, 2, "quanto-int4"),
    ],
)
def test_load_calibrate_edit_export_reload_and_infer(
    tmp_path: Path,
    model_kind: str,
    hidden_size: int,
    num_layers: int,
    quantization: str,
) -> None:
    """Small random models verify plumbing, not real-world quality or reasoning."""
    pytest.importorskip("optimum.quanto")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        AutoTokenizer,
        GPTNeoXConfig,
        GPTNeoXForCausalLM,
        LlamaConfig,
        LlamaForCausalLM,
        MistralConfig,
        MistralForCausalLM,
        PreTrainedTokenizerFast,
        Qwen2Config,
        Qwen2ForCausalLM,
    )

    torch.manual_seed(17)
    if model_kind == "llama":
        config = LlamaConfig(
            vocab_size=64,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 2,
            num_hidden_layers=num_layers,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
            bos_token_id=2,
            eos_token_id=3,
            pad_token_id=1,
        )
        model = LlamaForCausalLM(config)
    elif model_kind == "mistral":
        config = MistralConfig(
            vocab_size=64,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 2,
            num_hidden_layers=num_layers,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
            sliding_window=64,
            bos_token_id=2,
            eos_token_id=3,
            pad_token_id=1,
        )
        model = MistralForCausalLM(config)
    elif model_kind == "qwen2":
        config = Qwen2Config(
            vocab_size=64,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 2,
            num_hidden_layers=num_layers,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
            bos_token_id=2,
            eos_token_id=3,
            pad_token_id=1,
        )
        model = Qwen2ForCausalLM(config)
    elif model_kind == "gpt_neox":
        config = GPTNeoXConfig(
            vocab_size=64,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 2,
            num_hidden_layers=num_layers,
            num_attention_heads=4,
            max_position_embeddings=64,
            bos_token_id=2,
            eos_token_id=3,
            pad_token_id=1,
            tie_word_embeddings=False,
        )
        model = GPTNeoXForCausalLM(config)
    else:
        from transformers import GPT2Config, GPT2LMHeadModel

        config = GPT2Config(
            vocab_size=64,
            n_embd=hidden_size,
            n_layer=num_layers,
            n_head=4,
            n_positions=64,
            n_ctx=64,
            bos_token_id=2,
            eos_token_id=3,
            pad_token_id=1,
        )
        model = GPT2LMHeadModel(config)

    vocab = {"[UNK]": 0, "[PAD]": 1, "[BOS]": 2, "[EOS]": 3}
    calibration_texts = [
        "A small language model reads representative neutral text.",
        "Calibration samples should resemble inputs used by the application.",
        "The quick brown fox jumps over a quiet stream.",
        "A measured evaluation compares the same input on each model copy.",
    ]
    for text in calibration_texts:
        for token in text.split():
            vocab.setdefault(token, len(vocab))
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )

    source = tmp_path / "source"
    output = tmp_path / "quantized"
    source.mkdir()
    model.save_pretrained(source, safe_serialization=True)
    tokenizer.save_pretrained(source)
    calibration_file = tmp_path / "calibration.txt"
    calibration_file.write_text("\n".join(calibration_texts) + "\n", encoding="utf-8")

    source_weights = source / "model.safetensors"
    if not source_weights.exists():
        source_weights = next(source.glob("*.safetensors"))
    source_digest_before = hashlib.sha256(source_weights.read_bytes()).hexdigest()
    mapping_capability, mapping_identity = _promoted_test_capability(model)

    # A disabled ablation keeps this test focused on quantization preserving
    # an existing model; quantization itself is the tested in-memory edit.
    calibration_momentum = 0.9
    calibration_streamline = True
    if model_kind == "llama" and hidden_size == 16 and quantization == "quanto-int8":
        calibration_momentum = 0.0
    elif model_kind == "llama" and hidden_size == 16 and quantization == "quanto-int4":
        calibration_momentum = 1.0
        calibration_streamline = False
    extra_stages = []
    if model_kind == "llama" and hidden_size == 16 and quantization == "quanto-int8":
        extra_stages = [
            "tuning.weight_fingerprint_auto",
            "tuning.provenance_recording_auto",
            "tuning.reproducibility_auto",
            "tuning.memory_footprint_auto",
            "tuning.model_card_generation_auto",
        ]
    max_shard_size = (
        "1KB"
        if model_kind == "llama" and hidden_size == 16 and quantization == "quanto-int8-weight-only"
        else "5GB"
    )
    result = run_pipeline(
        model_path=str(source),
        output_dir=str(output),
        quantization=quantization,
        calibration_data=str(calibration_file),
            calibration_samples=4,
            calibration_max_length=32,
            calibration_momentum=calibration_momentum,
            calibration_streamline=calibration_streamline,
        device="cpu",
        dtype="float32",
        allow_mutation=True,
        enable=["tuning.model_export", *extra_stages],
        disable=["tuning.refusal_direction_ablation"],
        max_shard_size=max_shard_size,
        stage_options={
            "mapping_capability": mapping_capability,
            "mapping_expected_identity": mapping_identity,
        },
    )

    if model_kind == "gpt2":
        # Quanto 0.2.7 does not expose GPT-2 Conv1D weights through its
        # supported weight map. LayerNorm activation entries are not enough to
        # claim that the requested weight-quantization mode was applied.
        assert not result["success"]
        assert any(
            "no supported" in row["error"]
            or "no quantized weights" in row["error"]
            or "did not quantize any supported modules" in row["error"]
            for row in result["failed"]
        ), result["failed"]
        assert not output.exists()
        assert hashlib.sha256(source_weights.read_bytes()).hexdigest() == source_digest_before
        return

    assert result["success"], result["failed"]
    assert result["model_modified"] is True
    assert result["quantization"]["mode"] == quantization
    assert result["quantization"]["weight_type"] == (
        "qint4" if "int4" in quantization else "qint8"
    )
    assert result["quantization"]["weight_quantized_module_count"] > 0
    if quantization.endswith("weight-only"):
        assert result["quantization"]["activation_type"] is None
        assert result["quantization"]["calibration_samples"] == 0
    else:
        assert result["quantization"]["activation_type"] == "qint8"
        assert result["quantization"]["calibration_samples"] >= 1
        assert result["quantization"]["calibration_momentum"] == calibration_momentum
        assert result["quantization"]["calibration_streamline"] is calibration_streamline
    assert result["export"]["validated"] is True
    assert result["export"]["validation"]["weight_quantized_module_count"] == result["quantization"]["weight_quantized_module_count"]
    assert result["export"]["validation"]["activation_quantized_module_count"] == result["quantization"]["activation_quantized_module_count"]
    assert (output / "quanto_qmap.json").is_file()
    assert list(output.glob("*.safetensors"))
    export_manifest = json.loads((output / "autobliteratus_export.json").read_text(encoding="utf-8"))
    assert export_manifest["max_shard_size"] == max_shard_size
    assert export_manifest["sha256"]
    if max_shard_size == "1KB":
        assert (output / "model.safetensors.index.json").is_file()
        assert len(list(output.glob("*.safetensors"))) > 1
    if extra_stages:
        assert (output / "autobliteratus_provenance.json").is_file()
        assert (output / "autobliteratus_weight_fingerprint.json").is_file()
        export_manifest = json.loads((output / "autobliteratus_export.json").read_text(encoding="utf-8"))
        assert export_manifest["reproducibility"]["runtime"]["bitwise_reproducibility_claimed"] is False
        assert export_manifest["memory"]["optimization_performed"] is False
        assert "Model Artifact Card" in (output / "README.md").read_text(encoding="utf-8")
    assert hashlib.sha256(source_weights.read_bytes()).hexdigest() == source_digest_before

    # Simulate a pre-Transformers-5 GPT-NeoX checkpoint head name. Quanto's
    # stock 0.2.7 loader silently ignores this alias under newer Transformers.
    expected_head = None
    if model_kind == "gpt_neox":
        from safetensors.torch import load_file, save_file

        weight_file = next(output.glob("*.safetensors"))
        saved = load_file(str(weight_file), device="cpu")
        if "lm_head.weight" in saved:
            expected_head = saved.pop("lm_head.weight")
            saved["embed_out.weight"] = expected_head
        else:
            expected_head = saved.get("embed_out.weight")
        assert expected_head is not None
        save_file(saved, str(weight_file))

    # Reload through the compatibility loader, then run forward and generation.
    reloaded = load_quantized_model(output, device="cpu")
    if expected_head is not None:
        actual_head = reloaded.get_output_embeddings().weight.detach().cpu()
        assert torch.equal(actual_head, expected_head)
    reloaded_tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True)
    encoded = reloaded_tokenizer(
        "A small language model reads text.",
        return_tensors="pt",
        truncation=True,
        max_length=32,
    )
    with torch.no_grad():
        logits = reloaded(**encoded, use_cache=False).logits
        generated = reloaded.generate(**encoded, max_new_tokens=2, do_sample=False)
    assert logits.shape[:2] == encoded["input_ids"].shape
    assert torch.isfinite(logits).all()
    assert generated.shape[1] > encoded["input_ids"].shape[1]

    # Record a quantitative smoke metric without asserting quality for random
    # weights: randomly initialized models have no meaningful accuracy target.
    metric = float(logits.float().square().mean().sqrt().item())
    assert math.isfinite(metric)
