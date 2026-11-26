from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

import autobliteratus.observability.pareto as pareto_engine
from autobliteratus.observability.pareto import (
    MeasurementObservation,
    build_quality_latency_pareto,
)

QUALITY = "answer_quality"
LATENCY = "latency_ms"


def _observation(
    run_id: str,
    metric: str,
    value: float | None,
    *,
    candidate: str = "a",
    cohort: str = "workload-1",
    unit: str | None = None,
    stage: str = "evaluate",
    day: int = 20,
    commit: str | None = "commit-1",
    extra_dimensions: dict | None = None,
) -> MeasurementObservation:
    dimensions = {
        "workload_id": cohort,
        "backend": candidate,
    }
    if extra_dimensions:
        dimensions.update(extra_dimensions)
    return MeasurementObservation(
        run_id=run_id,
        commit_sha=commit,
        measured_at=datetime(2026, 11, day, 12, tzinfo=timezone.utc),
        stage=stage,
        metric=metric,
        value=value,
        unit=unit or ("score" if metric == QUALITY else "ms"),
        dimensions=dimensions,
    )


def _pair(
    run_id: str,
    quality: float,
    latency: float,
    *,
    candidate: str = "a",
    cohort: str = "workload-1",
    quality_stage: str = "evaluate",
    latency_stage: str = "evaluate",
    day: int = 20,
    commit: str | None = "commit-1",
) -> list[MeasurementObservation]:
    return [
        _observation(
            run_id, QUALITY, quality, candidate=candidate, cohort=cohort,
            stage=quality_stage, day=day, commit=commit,
        ),
        _observation(
            run_id, LATENCY, latency, candidate=candidate, cohort=cohort,
            stage=latency_stage, day=day, commit=commit,
        ),
    ]


def _report(observations, **overrides):
    arguments = {
        "quality_metric": QUALITY,
        "latency_metric": LATENCY,
        "cohort_dimensions": ["workload_id"],
        "candidate_dimensions": ["backend"],
    }
    arguments.update(overrides)
    return build_quality_latency_pareto(observations, **arguments)


def _by_candidate(report, cohort="workload-1"):
    return {
        row["candidate"]["backend"]: row
        for row in report.rows
        if row["cohort"]["workload_id"] == cohort
    }


def test_three_candidate_frontier_marks_strictly_dominated_candidate():
    observations = []
    observations += _pair("run-a", 0.80, 100, candidate="fast-low")
    observations += _pair("run-b", 0.95, 150, candidate="accurate-slow")
    observations += _pair("run-c", 0.90, 120, candidate="balanced")
    observations += _pair("run-d", 0.75, 180, candidate="dominated")

    report = _report(observations)
    rows = _by_candidate(report)

    assert rows["fast-low"]["pareto_efficient"] is True
    assert rows["accurate-slow"]["pareto_efficient"] is True
    assert rows["balanced"]["pareto_efficient"] is True
    assert rows["dominated"]["pareto_efficient"] is False
    assert rows["dominated"]["dominated_by"] == [
        {"backend": "accurate-slow"},
        {"backend": "balanced"},
        {"backend": "fast-low"},
    ]
    assert report.as_dict()["name"] == "quality_latency_pareto"


def test_lower_is_better_quality_direction_reverses_quality_comparison():
    observations = []
    observations += _pair("run-a", 0.20, 100, candidate="best-loss")
    observations += _pair("run-b", 0.30, 120, candidate="dominated")

    rows = _by_candidate(_report(observations, quality_direction="lower"))

    assert rows["best-loss"]["pareto_efficient"] is True
    assert rows["dominated"]["pareto_efficient"] is False
    assert rows["dominated"]["dominated_by"] == [{"backend": "best-loss"}]


def test_tied_points_are_both_efficient_and_tradeoffs_remain_efficient():
    observations = []
    observations += _pair("run-a", 0.9, 100, candidate="tie-a")
    observations += _pair("run-b", 0.9, 100, candidate="tie-b")
    observations += _pair("run-c", 0.95, 120, candidate="quality-first")
    observations += _pair("run-d", 0.85, 80, candidate="speed-first")

    rows = _by_candidate(_report(observations))

    assert rows["tie-a"]["pareto_efficient"] is True
    assert rows["tie-b"]["pareto_efficient"] is True
    assert rows["tie-a"]["dominated_by"] == []
    assert rows["quality-first"]["pareto_efficient"] is True
    assert rows["speed-first"]["pareto_efficient"] is True


def test_candidates_in_distinct_cohorts_never_dominate_each_other():
    observations = []
    observations += _pair("run-a", 0.5, 50, candidate="candidate-a", cohort="task-a")
    observations += _pair("run-b", 0.9, 100, candidate="candidate-b", cohort="task-b")

    report = _report(observations)

    assert len(report.rows) == 2
    assert all(row["pareto_efficient"] for row in report.rows)


def test_pairing_is_limited_to_run_candidate_and_repetitions_use_two_level_medians():
    observations = []
    # Run 1 has three quality repetitions and one latency repetition.
    observations += [
        _observation("run-1", QUALITY, 0.0),
        _observation("run-1", QUALITY, 0.8),
        _observation("run-1", QUALITY, 1.0),
        _observation("run-1", LATENCY, 100.0),
    ]
    # Run 2 has one quality and three latency repetitions.
    observations += [
        _observation("run-2", QUALITY, 0.6),
        _observation("run-2", LATENCY, 10.0),
        _observation("run-2", LATENCY, 20.0),
        _observation("run-2", LATENCY, 30.0),
    ]
    # A quality-only run must not pair with latency from another run.
    observations.append(_observation("run-3", QUALITY, 1.0))

    report = _report(observations)
    row = report.rows[0]

    assert row["quality_value"] == pytest.approx(0.7)  # median of run medians .8 and .6
    assert row["latency_value"] == pytest.approx(60.0)  # median of run medians 100 and 20
    assert row["paired_runs"] == 2
    assert row["quality_measurements"] == 4
    assert row["latency_measurements"] == 4
    assert report.metadata["coverage"]["unpaired_run_candidate_keys"] == 1


def test_unpaired_quality_and_latency_from_different_runs_are_not_imputed_or_joined():
    report = _report(
        [
            _observation("quality-run", QUALITY, 0.9),
            _observation("latency-run", LATENCY, 100.0),
        ]
    )

    assert report.rows == ()
    assert report.metadata["coverage"]["paired_run_candidate_keys"] == 0
    assert report.metadata["coverage"]["unpaired_run_candidate_keys"] == 2
    assert report.metadata["provenance"]["commits"] == []


def test_minimum_paired_runs_filters_candidate_and_reports_coverage():
    observations = _pair("run-1", 0.8, 100)
    observations += _pair("run-2", 0.85, 110, candidate="two-runs")
    observations += _pair("run-3", 0.86, 105, candidate="two-runs")

    report = _report(observations, min_paired_runs=2)

    assert [row["candidate"]["backend"] for row in report.rows] == ["two-runs"]
    assert report.metadata["coverage"]["candidates_filtered_by_min_paired_runs"] == 1


def test_invalid_dimensions_are_excluded_and_counted():
    observations = [
        _observation("run-a", QUALITY, 0.9, extra_dimensions={"backend": {"nested": "no"}}),
        _observation("run-a", LATENCY, 100),
        _observation("run-b", QUALITY, 0.9, extra_dimensions={"backend": None}),
        _observation("run-b", LATENCY, 100),
        _observation("run-c", QUALITY, 0.9, extra_dimensions={"backend": "api_key=secret"}),
        _observation("run-c", LATENCY, 100),
    ]

    report = _report(observations)

    assert report.rows == ()
    assert report.metadata["coverage"]["quality_invalid_dimensions"] == 3
    assert report.metadata["coverage"]["latency_invalid_dimensions"] == 0
    assert report.metadata["coverage"]["unpaired_run_candidate_keys"] == 3


def test_null_boolean_and_nonfinite_values_are_excluded():
    observations = [
        _observation("run-a", QUALITY, None),
        _observation("run-a", LATENCY, 100),
        _observation("run-b", QUALITY, True),
        _observation("run-b", LATENCY, 100),
        _observation("run-c", QUALITY, float("nan")),
        _observation("run-c", LATENCY, float("inf")),
    ]

    report = _report(observations)

    assert report.rows == ()
    assert report.metadata["coverage"]["quality_invalid_values"] == 3
    assert report.metadata["coverage"]["latency_invalid_values"] == 1


def test_inconsistent_units_within_cohort_are_rejected():
    observations = _pair("run-a", 0.9, 100, candidate="a")
    observations += [
        _observation("run-b", QUALITY, 0.8, candidate="b", unit="fraction"),
        _observation("run-b", LATENCY, 120, candidate="b"),
    ]

    with pytest.raises(ValueError, match="inconsistent units"):
        _report(observations)


def test_ambiguous_metric_stages_require_explicit_stage():
    observations = _pair("run-a", 0.9, 100)
    observations += [
        _observation("run-b", QUALITY, 0.8, stage="score"),
        _observation("run-b", LATENCY, 120),
    ]

    with pytest.raises(ValueError, match="multiple stages"):
        _report(observations)
    report = _report(observations, quality_stage="evaluate")
    assert report.rows[0]["paired_runs"] == 1


def test_date_filters_are_inclusive_and_provenance_tracks_contributing_rows():
    observations = _pair("run-before", 0.2, 200, day=19, commit="before")
    observations += _pair("run-start", 0.8, 100, day=20, commit="start")
    observations += _pair("run-end", 0.9, 90, day=25, commit="end")
    observations += _pair("run-after", 0.1, 300, day=26, commit="after")

    report = _report(observations, start_date=date(2026, 11, 20), end_date=date(2026, 11, 25))

    assert report.rows[0]["paired_runs"] == 2
    assert report.metadata["provenance"] == {
        "date_start": "2026-11-20",
        "date_end": "2026-11-25",
        "commits": ["end", "start"],
    }


def test_bad_dates_metrics_dimension_keys_and_limits_are_rejected():
    with pytest.raises(ValueError, match="on or before"):
        _report([], start_date=date(2026, 11, 2), end_date=date(2026, 11, 1))
    with pytest.raises(ValueError, match="must differ"):
        _report([], latency_metric=QUALITY)
    with pytest.raises(ValueError, match="simple identifier"):
        _report([], cohort_dimensions=["x); DROP TABLE measurements;--"])
    with pytest.raises(ValueError, match="disjoint"):
        _report([], candidate_dimensions=["workload_id"])
    with pytest.raises(ValueError, match="positive integer"):
        _report([], min_paired_runs=0)
    with pytest.raises(ValueError, match="positive integer"):
        _report([], limit=0)


def test_empty_report_and_row_limit_have_stable_coverage_metadata():
    empty = _report([])
    assert empty.rows == ()
    assert empty.metadata["coverage"]["eligible_candidates"] == 0
    assert empty.metadata["provenance"]["date_start"] is None

    observations = []
    for index in range(3):
        observations += _pair(
            f"run-{index}", 0.8 + index / 100, 100 - index,
            candidate=f"candidate-{index}",
        )
    limited = _report(observations, limit=1)
    assert len(limited.rows) == 1
    assert limited.metadata["coverage"]["truncated"] is True
    assert limited.metadata["coverage"]["eligible_candidates"] == 3


def test_dimension_scalar_types_are_preserved_and_do_not_collide():
    observations = []
    observations += [
        _observation("run-a", QUALITY, 0.8, extra_dimensions={"mode": 1}),
        _observation("run-a", LATENCY, 100, extra_dimensions={"mode": 1}),
        _observation("run-b", QUALITY, 0.9, extra_dimensions={"mode": True}),
        _observation("run-b", LATENCY, 90, extra_dimensions={"mode": True}),
    ]
    report = _report(
        observations,
        candidate_dimensions=["backend", "mode"],
    )

    modes = [row["candidate"]["mode"] for row in report.rows]
    assert modes == [1, True]


def test_input_observation_cap_fails_explicitly(monkeypatch):
    monkeypatch.setattr(pareto_engine, "MAX_INPUT_OBSERVATIONS", 3)
    observations = _pair("run-a", 0.8, 100) + _pair("run-b", 0.9, 90)

    with pytest.raises(ValueError, match="input limit"):
        _report(observations)
