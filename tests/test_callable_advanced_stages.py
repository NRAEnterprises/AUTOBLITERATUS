from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from autobliteratus.pipeline import run_pipeline
from autobliteratus.tuning._09_multi_objective_auto_tuning import (
    run as run_multi_objective,
)
from autobliteratus.tuning._10_auto_security_scanning import run as run_security_scan
from autobliteratus.tuning._16_anti_router_collapse_auto_tuning import (
    _entropy_report,
    _read_probe_texts,
)
from autobliteratus.tuning._16_anti_router_collapse_auto_tuning import (
    run as run_router_scan,
)
from autobliteratus.tuning.toggle_registry import TOGGLES


class _Tokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": torch.ones((1, 3), dtype=torch.long)}


class _FixedRouter(torch.nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, hidden):
        return self.logits.to(hidden.device).reshape(1, 1, -1).expand(*hidden.shape[:-1], -1)


class _RouterModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.embedding = torch.nn.Embedding(4, 2)
        self.router = _FixedRouter(logits)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids):
        hidden = self.embedding(input_ids)
        return self.router(hidden)


class _ModelHandle:
    def __init__(self, model, num_experts=2):
        self.model = model
        self.tokenizer = _Tokenizer()
        self.num_experts = num_experts

    def summary(self):
        return "fake-local-test-model"


class _Adapter:
    def __init__(self, model, num_experts=2):
        self.handle = _ModelHandle(model, num_experts=num_experts)


def test_router_pipeline_captures_real_outputs_and_reports_low_utilization():
    model = _RouterModel(torch.tensor([12.0, -12.0]))
    model.train()
    adapter = _Adapter(model)
    context = {"routing_probe_texts": ["representative prompt"], "routing_entropy_threshold": 0.5}

    report = run_router_scan(adapter, context)

    assert report["status"] == "measured"
    assert report["collapsed_layers"] == ["router"]
    assert report["layers"]["router"]["expert_load"][0] > 0.999
    assert report["model_modified"] is False
    assert model.training is True
    assert all(not module._forward_hooks for module in model.modules())
    assert context["router_collapse_report"] is report


def test_router_pipeline_reports_no_text_and_rejects_invalid_expert_shape():
    adapter = _Adapter(_RouterModel(torch.tensor([0.0, 0.0])))
    report = run_router_scan(adapter, {})
    assert report["status"] == "not_evaluated"

    with pytest.raises(ValueError, match="must end in 2 experts"):
        _entropy_report({"layer": torch.ones(2, 3)}, 2, 0.5)


def test_router_probe_reads_json_array_calibration_file(tmp_path):
    calibration = tmp_path / "routing.json"
    calibration.write_text('[{"prompt": "one"}, {"text": "two"}]', encoding="utf-8")
    assert _read_probe_texts({"calibration_data": str(calibration)}) == ["one", "two"]


def test_multi_objective_pipeline_computes_front_and_weighted_choice():
    context = {
        "multi_objective_objectives": {"quality": "max", "latency": "min"},
        "multi_objective_weights": {"quality": 0.2, "latency": 0.8},
        "multi_objective_candidates": [
            {"id": "balanced-quality", "config": {"mode": "a"}, "scores": {"quality": 0.9, "latency": 200}},
            {"id": "fast", "config": {"mode": "b"}, "scores": {"quality": 0.8, "latency": 100}},
            {"id": "dominated", "config": {"mode": "c"}, "scores": {"quality": 0.7, "latency": 250}},
        ],
    }

    report = run_multi_objective(None, context)

    assert report["status"] == "selected"
    assert [row["candidate_id"] for row in report["pareto_frontier"]] == ["balanced-quality", "fast"]
    assert report["selected_candidate"]["candidate_id"] == "fast"
    assert report["model_modified"] is False
    assert context["multi_objective_report"] is report


def test_multi_objective_pipeline_handles_empty_and_rejects_nonfinite_scores():
    assert run_multi_objective(None, {})["status"] == "no_candidates"
    context = {
        "multi_objective_objectives": {"quality": "max"},
        "multi_objective_candidates": [{"config": {}, "scores": {"quality": float("nan")}}],
    }
    with pytest.raises(ValueError, match="finite number"):
        run_multi_objective(None, context)


def test_security_pipeline_finds_nonfinite_and_extreme_weights_without_mutation():
    model = torch.nn.Module()
    model.nan_parameter = torch.nn.Parameter(torch.tensor([1.0, float("nan")]))
    model.large_parameter = torch.nn.Parameter(torch.tensor([1.0, 1001.0]))
    before_nan = model.nan_parameter.detach().clone()
    before_large = model.large_parameter.detach().clone()
    context = {"security_scan_max_abs": 1000.0}

    report = run_security_scan(_Adapter(model), context)

    assert report["status"] == "findings"
    assert report["parameters_scanned"] == 2
    assert {item["kind"] for item in report["findings"]} == {
        "non_finite_parameter_values",
        "extreme_parameter_magnitude",
    }
    assert report["model_modified"] is False
    assert model.nan_parameter[0].item() == before_nan[0].item()
    assert torch.isnan(model.nan_parameter[1]) and torch.isnan(before_nan[1])
    assert torch.equal(model.large_parameter, before_large)
    assert context["security_scan_report"] is report
    assert "does not detect backdoors" in report["limitations"][0]


def test_security_pipeline_clean_model_and_threshold_validation():
    model = torch.nn.Linear(2, 2)
    report = run_security_scan(_Adapter(model), {})
    assert report["status"] == "no_numeric_anomalies_detected"
    with pytest.raises(ValueError, match="finite and positive"):
        run_security_scan(_Adapter(model), {"security_scan_max_abs": float("inf")})


def test_pipeline_executes_selected_stages_and_returns_reports(monkeypatch, tmp_path):
    model = _RouterModel(torch.tensor([10.0, -10.0]))
    adapter = _Adapter(model)
    selected = [
        "tuning.anti_router_collapse",
        "tuning.multi_objective_auto",
        "tuning.auto_security_scanning",
    ]
    disabled = [
        toggle_id
        for toggle_id, metadata in TOGGLES.items()
        if metadata["available"] and toggle_id not in selected
    ]
    monkeypatch.setattr("autobliteratus.core.adapters.TuningAdapter.from_path", lambda *args, **kwargs: adapter)

    result = run_pipeline(
        model_path="fake-local-model",
        work_dir=str(tmp_path / "work"),
        dry_run=True,
        enable=selected,
        disable=disabled,
        stage_options={
            "routing_probe_texts": ["pipeline route probe"],
            "multi_objective_objectives": {"quality": "max", "latency": "min"},
            "multi_objective_candidates": [
                {"id": "a", "config": {"mode": "a"}, "scores": {"quality": 0.9, "latency": 2.0}},
                {"id": "b", "config": {"mode": "b"}, "scores": {"quality": 0.8, "latency": 1.0}},
            ],
        },
    )

    assert result["success"] is True
    assert set(selected).issubset(result["executed"])
    assert result["router_collapse"]["status"] == "measured"
    assert result["multi_objective"]["status"] == "selected"
    assert result["security_scan"]["status"] == "no_numeric_anomalies_detected"
    assert result["model_modified"] is False


def test_pipeline_rejects_unknown_stage_options_before_model_load(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "autobliteratus.core.adapters.TuningAdapter.from_path",
        lambda *args, **kwargs: pytest.fail("validation should happen before model load"),
    )
    with pytest.raises(ValueError, match="Unknown stage_options"):
        run_pipeline(
            model_path="fake-local-model",
            work_dir=str(tmp_path / "work"),
            plan_only=True,
            stage_options={"execute_arbitrary_code": True},
        )


def test_cli_prints_summaries_for_selected_stages(monkeypatch, capsys):
    from autobliteratus import __main__ as entrypoint
    from autobliteratus import cli

    args = SimpleNamespace(model_path="model", work_dir="work")
    for key, value in {
        "output_dir": None,
        "quantization": "none",
        "target_frameworks": ["transformers"],
        "deployment_target": "server",
        "enable": [],
        "disable": [],
        "dry_run": True,
        "ablation_strength": 1.0,
        "ablation_component": "attn.out",
        "trust_remote_code": False,
        "calibration_data": None,
        "calibration_samples": 1,
        "calibration_max_length": 8,
        "calibration_momentum": 0.9,
        "calibration_streamline": True,
        "device": "cpu",
        "dtype": "float32",
        "device_map": None,
        "max_memory": None,
        "offload_folder": None,
        "max_shard_size": "1GB",
        "plan_only": False,
        "stage_options": None,
    }.items():
        setattr(args, key, value)
    monkeypatch.setattr(cli, "parse_args", lambda argv=None: args)
    monkeypatch.setattr(
        "autobliteratus.pipeline.run_pipeline",
        lambda **kwargs: {
            "success": True,
            "executed": [],
            "skipped": [],
            "failed": [],
            "model_summary": "fake",
            "model_modified": False,
            "router_collapse": {"status": "measured", "collapsed_layers": ["router"]},
            "multi_objective": {
                "status": "selected",
                "candidate_count": 2,
                "pareto_frontier": [{}, {}],
                "selected_candidate": {"candidate_id": "b"},
            },
            "security_scan": {"status": "findings", "parameters_scanned": 4, "findings": [{}]},
        },
    )

    assert entrypoint.main([]) == 0
    output = capsys.readouterr().out
    assert "router diagnostics: measured (collapsed layers: 1)" in output
    assert "multi-objective: selected (candidates: 2, frontier: 2, selected: b)" in output
    assert "numeric security scan: findings (parameters scanned: 4, findings: 1)" in output
