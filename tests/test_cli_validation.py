from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from autobliteratus.cli import parse_args
from autobliteratus.core.loader import load_model
from autobliteratus.pipeline import run_pipeline
from autobliteratus.tuning._27_architecture_detection import run as detect_architecture
from autobliteratus.tuning.toggle_registry import TOGGLES


def test_cli_rejects_prototype_toggle_without_model_path() -> None:
    prototype = next(toggle_id for toggle_id, meta in TOGGLES.items() if not meta["available"])
    with pytest.raises(SystemExit, match="prototypes"):
        parse_args(["--enable", prototype])


def test_pipeline_rejects_prototype_before_model_loading(monkeypatch) -> None:
    prototype = next(toggle_id for toggle_id, meta in TOGGLES.items() if not meta["available"])

    def should_not_load(*args, **kwargs):
        raise AssertionError("model load must not occur for an unavailable stage")

    monkeypatch.setattr("autobliteratus.core.adapters.TuningAdapter.from_path", should_not_load)
    with pytest.raises(ValueError, match="prototypes"):
        run_pipeline(model_path="invalid-model", enable=[prototype])


def test_pipeline_rejects_unimplemented_target_framework_before_model_loading(monkeypatch) -> None:
    def should_not_load(*args, **kwargs):
        raise AssertionError("model load must not occur for an unsupported target framework")

    monkeypatch.setattr("autobliteratus.core.adapters.TuningAdapter.from_path", should_not_load)
    with pytest.raises(ValueError, match="Only the Transformers"):
        run_pipeline(model_path="invalid-model", target_frameworks=["pytorch-native"])


def test_pipeline_rejects_export_disable_for_modifying_run_before_model_loading(monkeypatch) -> None:
    def should_not_load(*args, **kwargs):
        raise AssertionError("model load must not occur when a required export is disabled")

    monkeypatch.setattr("autobliteratus.core.adapters.TuningAdapter.from_path", should_not_load)
    with pytest.raises(ValueError, match="cannot be disabled"):
        run_pipeline(model_path="invalid-model", disable=["tuning.model_export"])


def test_weight_only_mode_does_not_require_calibration_data(monkeypatch) -> None:
    def reached_loader(*args, **kwargs):
        raise AssertionError("mutation authorization should fail before model loading")

    monkeypatch.setattr("autobliteratus.core.adapters.TuningAdapter.from_path", reached_loader)
    with pytest.raises(PermissionError, match="Mutation-capable stages"):
        run_pipeline(
            model_path="invalid-model",
            quantization="quanto-int8-weight-only",
            device="cpu",
        )


def test_cli_exposes_calibration_diagnostics() -> None:
    args = parse_args(["--calibration-momentum", "0", "--calibration-no-streamline"])
    assert args.calibration_momentum == 0.0
    assert args.calibration_streamline is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"ablation_strength": float("nan")}, "ablation_strength"),
        ({"ablation_component": "unknown"}, "ablation_component"),
    ],
)
def test_direct_pipeline_rejects_invalid_ablation_options_before_loading(
    monkeypatch, kwargs, message
) -> None:
    def should_not_load(*args, **call_kwargs):
        raise AssertionError("model loading should not be reached")

    monkeypatch.setattr("autobliteratus.core.adapters.TuningAdapter.from_path", should_not_load)
    with pytest.raises(ValueError, match=message):
        run_pipeline(model_path="invalid-model", **kwargs)


def test_encoder_decoder_model_is_rejected_before_weight_loading(tmp_path) -> None:
    from transformers import T5Config

    T5Config(vocab_size=32, d_model=16, d_ff=32, num_layers=1, num_heads=4).save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="Encoder-decoder models are not supported"):
        load_model(str(tmp_path), device="cpu")


def test_architecture_stage_reports_observed_model_structure() -> None:
    model = torch.nn.Module()
    model.config = SimpleNamespace(
        model_type="gpt2",
        architectures=["GPT2LMHeadModel"],
        n_embd=16,
        num_experts=None,
        is_encoder_decoder=False,
        vision_config=None,
        audio_config=None,
        video_config=None,
    )
    model.linear = torch.nn.Linear(16, 16)
    adapter = SimpleNamespace(
        handle=SimpleNamespace(model=model),
        modules=SimpleNamespace(count=lambda: 12),
    )
    context = {}

    report = detect_architecture(adapter, context)

    assert report["model_type"] == "gpt2"
    assert report["family"] == "decoder-only-causal-lm"
    assert report["block_count"] == 12
    assert report["linear_module_count"] == 1
    assert context["architecture_report"] == report
