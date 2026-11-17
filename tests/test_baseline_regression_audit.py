from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from autobliteratus.pipeline import run_pipeline
from autobliteratus.tuning._07_baseline_auto_tuning_pre_post import (
    BaselineAutoTuningPrePost,
    validate_metrics,
)
from autobliteratus.tuning._07_baseline_auto_tuning_pre_post import (
    run as run_baseline,
)
from autobliteratus.tuning._81_benchmark_regression_auto_tuning import (
    BenchmarkRegressionAutoTuning,
    RegressionDetectedError,
)
from autobliteratus.tuning._81_benchmark_regression_auto_tuning import (
    run as run_regression,
)
from autobliteratus.tuning._89_audit_trail_auto_tuning import finalize_pipeline_audit
from autobliteratus.tuning.evaluation_metrics import load_evaluation_texts
from autobliteratus.tuning.toggle_registry import TOGGLES


class _Tokenizer:
    def __call__(self, text, **kwargs):
        return {
            "input_ids": torch.tensor([[1, 2, 3, 2]], dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
        }


class _TinyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(4, 8)
        self.lm_head = torch.nn.Linear(8, 4)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return SimpleNamespace(logits=self.lm_head(self.embedding(input_ids)))


class _Handle:
    architecture = "test-causal-lm"
    architecture_class = "dense_transformer"
    device = "cpu"
    dtype = "float32"

    def __init__(self, model):
        self.model = model
        self.tokenizer = _Tokenizer()

    def summary(self):
        return "tiny-test-causal-lm"


class _Adapter:
    def __init__(self):
        self.handle = _Handle(_TinyCausalLM())


def _metric_result(metrics, *, sample_count=2, scored_tokens=6):
    return {
        "metrics": metrics,
        "sample_count": sample_count,
        "scored_tokens": scored_tokens,
        "forward_seconds": 0.01,
        "perplexity_cap_nll": 80.0,
    }


def test_baseline_stage_measures_and_stores_finite_paired_metrics():
    adapter = _Adapter()
    adapter.handle.model.train()
    context = {"evaluation_texts": ["evaluation sample one", "evaluation sample two"]}

    report = run_baseline(adapter, context)

    assert report["status"] == "baseline_recorded"
    assert set(report["metrics"]) == {"mean_nll", "perplexity", "forward_tokens_per_second"}
    assert all(math.isfinite(value) for value in report["metrics"].values())
    assert report["sample_count"] == 2
    assert context["baseline_metrics"] == report["metrics"]
    assert adapter.handle.model.training is True
    assert report["model_modified"] is False


def test_baseline_stage_requires_an_explicit_evaluation_battery():
    with pytest.raises(ValueError, match="require explicit evaluation_texts or evaluation_data"):
        run_baseline(_Adapter(), {})


def test_jsonl_evaluation_data_is_parsed_and_sample_limited(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text('{"prompt":"one"}\n{"text":"two"}\n{"other":"ignored"}\n', encoding="utf-8")
    assert load_evaluation_texts({"evaluation_data": str(path), "evaluation_max_samples": 1}) == ["one"]


def test_baseline_metric_validation_and_thresholds_are_fail_closed():
    with pytest.raises(ValueError, match="finite number"):
        validate_metrics({"accuracy": float("nan")}, label="baseline")
    with pytest.raises(ValueError, match="metric names must match"):
        BaselineAutoTuningPrePost().derive_thresholds({"loss": 1.0}, {"accuracy": 0.9})

    thresholds = BaselineAutoTuningPrePost(tolerance_ratio=0.05).derive_thresholds(
        {"loss": 0.0, "speed": 10.0},
        {"loss": 0.01, "speed": 11.0},
    )
    assert thresholds["loss"]["absolute_tolerance"] == 0.0
    assert thresholds["speed"]["absolute_tolerance"] == pytest.approx(0.5)


def test_regression_comparison_respects_min_max_directions_and_relative_tolerance():
    checker = BenchmarkRegressionAutoTuning(regression_threshold=0.02)
    baseline = {"mean_nll": 1.0, "perplexity": 2.7, "forward_tokens_per_second": 100.0}
    good = {"mean_nll": 1.01, "perplexity": 2.74, "forward_tokens_per_second": 98.5}
    assert checker.compare(baseline, good).status == "passed"

    bad = {"mean_nll": 1.1, "perplexity": 2.7, "forward_tokens_per_second": 80.0}
    report = checker.compare(baseline, bad)
    assert report.status == "regression"
    assert report.regressions == ["mean_nll", "forward_tokens_per_second"]
    assert report.per_benchmark["mean_nll"]["allowed_absolute_delta"] == pytest.approx(0.02)
    assert report.per_benchmark["forward_tokens_per_second"]["tolerance_ratio"] == 0.10

    strict_speed = BenchmarkRegressionAutoTuning(regression_tolerances={"forward_tokens_per_second": 0.01})
    strict_report = strict_speed.compare(
        {"mean_nll": 1.0, "perplexity": 2.7, "forward_tokens_per_second": 100.0},
        {"mean_nll": 1.0, "perplexity": 2.7, "forward_tokens_per_second": 98.0},
    )
    assert strict_report.regressions == ["forward_tokens_per_second"]


def test_regression_comparison_rejects_missing_nonfinite_or_directionless_metrics():
    checker = BenchmarkRegressionAutoTuning()
    with pytest.raises(ValueError, match="names must match exactly"):
        checker.compare({"mean_nll": 1.0}, {"other_metric": 1.0})
    with pytest.raises(ValueError, match="finite number"):
        checker.compare({"mean_nll": float("inf")}, {"mean_nll": 1.0})
    with pytest.raises(ValueError, match="directions must name every"):
        BenchmarkRegressionAutoTuning(directions={"unknown": "min"}).compare(
            {"mean_nll": 1.0}, {"mean_nll": 1.0}
        )


def test_regression_stage_stores_failure_report_before_raising(monkeypatch):
    import autobliteratus.tuning._81_benchmark_regression_auto_tuning as module

    monkeypatch.setattr(
        module,
        "evaluate_causal_lm",
        lambda *args, **kwargs: _metric_result(
            {"mean_nll": 1.2, "perplexity": 3.0, "forward_tokens_per_second": 80.0}
        ),
    )
    context = {
        "evaluation_texts": ["sample"],
        "baseline_metrics": {"mean_nll": 1.0, "perplexity": 2.7, "forward_tokens_per_second": 100.0},
    }
    with pytest.raises(RegressionDetectedError, match="mean_nll"):
        run_regression(_Adapter(), context)
    assert context["benchmark_regression_report"]["status"] == "regression"
    assert "mean_nll" in context["benchmark_regression_report"]["regressions"]


def test_audit_finalizer_writes_unique_atomic_prompt_redacted_artifacts(tmp_path):
    adapter = _Adapter()
    context = {
        "work_dir": str(tmp_path),
        "model_path": "/private/model/location",
        "quantization": "none",
        "evaluation_texts": ["secret prompt should not be persisted"],
        "baseline_report": {"status": "baseline_recorded", "metrics": {"mean_nll": 1.0}, "sample_count": 1},
        "benchmark_regression_report": {"status": "passed", "regressions": [], "post_metrics": {"mean_nll": 1.0}},
        "export_report": {"path": "/output/model", "format": "transformers", "validated": True},
    }
    report = SimpleNamespace(
        executed=["tuning.baseline_auto_pre_post", "tuning.audit_trail_auto"],
        skipped=["tuning.model_export"],
        failed=[{"id": "tuning.benchmark_regression_auto", "error_type": "RegressionDetectedError", "error": "secret prompt"}],
    )

    first = finalize_pipeline_audit(adapter, context, report)
    second = finalize_pipeline_audit(adapter, context, report)

    assert first["status"] == "written"
    assert first["jsonl_path"] != second["jsonl_path"]
    payload = json.loads(Path(first["jsonl_path"]).read_text(encoding="utf-8").splitlines()[0])
    markdown = Path(first["markdown_path"]).read_text(encoding="utf-8")
    assert payload["stages"]["failed"] == [
        {"stage": "tuning.benchmark_regression_auto", "error_type": "RegressionDetectedError"}
    ]
    assert payload["artifact"]["validated"] is True
    assert "secret prompt" not in markdown
    assert "secret prompt" not in Path(first["jsonl_path"]).read_text(encoding="utf-8")
    assert context["audit_report"] == second


def test_pipeline_captures_baseline_before_edit_and_audits_regression_failure(monkeypatch, tmp_path):
    import autobliteratus.tuning._07_baseline_auto_tuning_pre_post as baseline_module
    import autobliteratus.tuning._81_benchmark_regression_auto_tuning as regression_module

    adapter = _Adapter()
    baseline_values = _metric_result(
        {"mean_nll": 1.0, "perplexity": 2.7, "forward_tokens_per_second": 100.0}
    )
    post_values = _metric_result(
        {"mean_nll": 1.3, "perplexity": 3.6, "forward_tokens_per_second": 70.0}
    )
    monkeypatch.setattr(baseline_module, "evaluate_causal_lm", lambda *args, **kwargs: baseline_values)
    monkeypatch.setattr(regression_module, "evaluate_causal_lm", lambda *args, **kwargs: post_values)
    selected = [
        "tuning.baseline_auto_pre_post",
        "tuning.benchmark_regression_auto",
        "tuning.audit_trail_auto",
    ]
    disabled = [
        toggle_id
        for toggle_id, meta in TOGGLES.items()
        if meta["available"] and toggle_id not in selected
    ]
    monkeypatch.setattr("autobliteratus.core.adapters.TuningAdapter.from_path", lambda *args, **kwargs: adapter)

    result = run_pipeline(
        model_path="test-model-id",
        work_dir=str(tmp_path / "workspace"),
        dry_run=True,
        enable=selected,
        disable=disabled,
        stage_options={"evaluation_texts": ["prompt that must not be audited"]},
    )

    assert result["success"] is False
    assert result["baseline"]["status"] == "baseline_recorded"
    assert result["benchmark_regression"]["status"] == "regression"
    assert any(item["id"] == "tuning.benchmark_regression_auto" for item in result["failed"])
    assert result["audit"]["status"] == "written"
    audit_text = Path(result["audit"]["jsonl_path"]).read_text(encoding="utf-8")
    assert "prompt that must not be audited" not in audit_text
    audit_data = json.loads(audit_text)
    assert "tuning.baseline_auto_pre_post" in audit_data["stages"]["executed"]
    assert any(item["stage"] == "tuning.benchmark_regression_auto" for item in audit_data["stages"]["failed"])


def test_plan_orders_baseline_before_edits_and_regression_after(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "autobliteratus.core.adapters.TuningAdapter.from_path",
        lambda *args, **kwargs: pytest.fail("plan-only must not load model weights"),
    )
    result = run_pipeline(
        model_path="not-downloaded-model",
        work_dir=str(tmp_path / "work"),
        plan_only=True,
        enable=[
            "tuning.benchmark_regression_auto",
            "tuning.refusal_direction_ablation",
            "tuning.model_export",
        ],
        stage_options={"evaluation_texts": ["evaluation sample"]},
    )
    order = result["plan"]["operation_graph"]
    assert order.index("tuning.baseline_auto_pre_post") < order.index("tuning.refusal_direction_ablation")
    assert order.index("tuning.refusal_direction_ablation") < order.index("tuning.benchmark_regression_auto")


def test_audit_path_cannot_write_inside_source_model(tmp_path, monkeypatch):
    source = tmp_path / "source-model"
    source.mkdir()
    monkeypatch.setattr(
        "autobliteratus.core.adapters.TuningAdapter.from_path",
        lambda *args, **kwargs: pytest.fail("path validation must precede model load"),
    )
    with pytest.raises(ValueError, match="overlaps source"):
        run_pipeline(
            model_path=str(source),
            work_dir=str(source),
            dry_run=True,
            enable=["tuning.audit_trail_auto"],
        )


def test_regression_preflight_requires_evaluation_input_and_baseline_dependency(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "autobliteratus.core.adapters.TuningAdapter.from_path",
        lambda *args, **kwargs: pytest.fail("preflight must run before model loading"),
    )
    with pytest.raises(ValueError, match="Provide exactly one"):
        run_pipeline(
            model_path="model-id",
            work_dir=str(tmp_path / "work"),
            plan_only=True,
            enable=["tuning.benchmark_regression_auto"],
        )
    with pytest.raises(ValueError, match="requires tuning.baseline_auto_pre_post"):
        run_pipeline(
            model_path="model-id",
            work_dir=str(tmp_path / "work"),
            plan_only=True,
            enable=["tuning.benchmark_regression_auto"],
            disable=["tuning.baseline_auto_pre_post"],
            stage_options={"evaluation_texts": ["sample"]},
        )
