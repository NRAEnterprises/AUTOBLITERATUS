from __future__ import annotations

import pytest

from autobliteratus.cli import parse_args
from autobliteratus.core.loader import load_model
from autobliteratus.pipeline import run_pipeline
from autobliteratus.run_config import load_run_config, parse_memory_limits


def test_yaml_config_is_safe_and_cli_values_override(tmp_path) -> None:
    config = tmp_path / "profile.yaml"
    config.write_text(
        "model_path: from-config\ndevice: cpu\ndevice_map: auto\n"
        "max_memory:\n  0: 12GiB\n  cpu: 32GiB\n",
        encoding="utf-8",
    )
    args = parse_args(["--config", str(config), "--model-path", "from-cli", "--device-map", "balanced"])
    assert args.model_path == "from-cli"
    assert args.device == "cpu"
    assert args.device_map == "balanced"
    assert args.max_memory == {0: "12GiB", "cpu": "32GiB"}


def test_toml_config_is_supported(tmp_path) -> None:
    config = tmp_path / "profile.toml"
    config.write_text(
        '[autobliteratus]\nmodel_path = "local/model"\nquantization = "none"\n',
        encoding="utf-8",
    )
    assert load_run_config(config)["model_path"] == "local/model"


def test_yaml_stage_options_are_loaded_as_data(tmp_path) -> None:
    config = tmp_path / "stage-options.yaml"
    config.write_text(
        "model_path: local/model\n"
        "enable: [tuning.multi_objective_auto, tuning.auto_security_scanning]\n"
        "stage_options:\n"
        "  multi_objective_objectives: {quality: max, latency: min}\n"
        "  multi_objective_candidates:\n"
        "    - {id: candidate-a, config: {mode: int8}, scores: {quality: 0.9, latency: 2.0}}\n",
        encoding="utf-8",
    )
    values = load_run_config(config)
    assert values["stage_options"]["multi_objective_objectives"] == {"quality": "max", "latency": "min"}
    assert values["stage_options"]["multi_objective_candidates"][0]["id"] == "candidate-a"
    args = parse_args(["--config", str(config)])
    assert args.stage_options["multi_objective_candidates"][0]["id"] == "candidate-a"


def test_evaluation_gate_profile_options_are_loaded_through_cli(tmp_path) -> None:
    config = tmp_path / "evaluation-gate.yaml"
    config.write_text(
        "model_path: local/model\n"
        "enable: [tuning.benchmark_regression_auto]\n"
        "stage_options:\n"
        "  evaluation_texts: [sample one, sample two]\n"
        "  evaluation_max_samples: 2\n"
        "  evaluation_max_length: 128\n"
        "  regression_metric_directions: {mean_nll: min, perplexity: min, forward_tokens_per_second: max}\n"
        "  regression_tolerances: {mean_nll: 0.02, forward_tokens_per_second: 0.10}\n",
        encoding="utf-8",
    )
    args = parse_args(["--config", str(config)])
    assert args.enable == ["tuning.benchmark_regression_auto"]
    assert args.stage_options["evaluation_texts"] == ["sample one", "sample two"]
    assert args.stage_options["regression_metric_directions"]["mean_nll"] == "min"
    assert args.stage_options["regression_tolerances"]["forward_tokens_per_second"] == 0.10


def test_stage_options_must_be_a_mapping(tmp_path) -> None:
    config = tmp_path / "bad-stage-options.yaml"
    config.write_text("stage_options: [not, a, mapping]\n", encoding="utf-8")
    with pytest.raises(TypeError, match="stage_options must be a mapping"):
        load_run_config(config)


def test_config_rejects_unknown_keys_and_unsafe_yaml(tmp_path) -> None:
    import yaml

    unknown = tmp_path / "unknown.yaml"
    unknown.write_text("model_path: test\ntrust_everything: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown run configuration keys"):
        load_run_config(unknown)

    unsafe = tmp_path / "unsafe.yaml"
    unsafe.write_text("!!python/object/apply:os.system ['echo unsafe']\n", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        load_run_config(unsafe)


def test_memory_limit_parser_accepts_mapping_and_cli_pairs() -> None:
    assert parse_memory_limits(["0=20GiB", "cpu=64GB"]) == {0: "20GiB", "cpu": "64GB"}
    with pytest.raises(ValueError, match="DEVICE=SIZE"):
        parse_memory_limits(["malformed"])


def test_plan_only_returns_operation_graph_without_loading_model(tmp_path, monkeypatch) -> None:
    def should_not_load(*args, **kwargs):
        raise AssertionError("plan-only must not instantiate a model loader")

    monkeypatch.setattr("autobliteratus.core.adapters.TuningAdapter.from_path", should_not_load)
    result = run_pipeline(
        model_path="not-downloaded-yet",
        work_dir=str(tmp_path / "work"),
        plan_only=True,
        device_map="auto",
        max_memory={0: "8GiB"},
    )
    assert result["success"]
    assert result["planned"]
    assert result["executed"] == []
    assert result["plan"]["device_map"] == "auto"
    assert result["plan"]["max_memory"] == {0: "8GiB"}


def test_plan_only_reports_requested_unavailable_stage_without_loading(tmp_path, monkeypatch) -> None:
    from autobliteratus.tuning.toggle_registry import TOGGLES

    prototype = next(key for key, value in TOGGLES.items() if not value["available"])
    monkeypatch.setattr(
        "autobliteratus.core.adapters.TuningAdapter.from_path",
        lambda *args, **kwargs: pytest.fail("model load must not happen"),
    )
    result = run_pipeline(
        model_path="not-downloaded-yet",
        work_dir=str(tmp_path / "work"),
        enable=[prototype],
        plan_only=True,
    )
    assert result["success"] is False
    assert prototype in result["unavailable"]


def test_cli_parses_accelerator_sharding_options() -> None:
    args = parse_args(
        ["--device-map", "balanced", "--max-memory", "0=20GiB", "--max-memory", "cpu=64GB", "--offload-folder", "/tmp/ab-offload"]
    )
    assert args.device_map == "balanced"
    assert args.max_memory == {0: "20GiB", "cpu": "64GB"}
    assert args.offload_folder == "/tmp/ab-offload"


def test_loader_supports_accelerate_device_map_on_cpu(tmp_path) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    source = tmp_path / "tiny-gpt2"
    source.mkdir()
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=8,
            n_embd=8,
            n_layer=1,
            n_head=2,
            n_positions=16,
            n_ctx=16,
            bos_token_id=2,
            eos_token_id=3,
            pad_token_id=1,
        )
    )
    model.save_pretrained(source, safe_serialization=True)
    backend = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "[BOS]": 2, "[EOS]": 3, "hello": 4}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    tokenizer.save_pretrained(source)

    handle = load_model(
        str(source),
        device="cpu",
        dtype="float32",
        device_map="auto",
        max_memory={"cpu": "2GiB"},
        offload_folder=str(tmp_path / "offload"),
    )
    assert handle.device in {"cpu", "mixed:cpu"}
    assert handle.model.hf_device_map
    assert (tmp_path / "offload").is_dir()
