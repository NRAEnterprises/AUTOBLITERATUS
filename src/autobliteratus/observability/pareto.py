"""Pure aggregation and Pareto classification for quality-versus-latency reports.

This module deliberately does not open a database or implement the CLI. A
read-only query layer should supply ``MeasurementObservation`` rows and serialize
the returned ``ParetoReport``. The engine applies the v1 pairing and aggregation
contract deterministically and does not infer missing measurements.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from statistics import median
from typing import Any

from autobliteratus.observability.telemetry_store import redact_text

MAX_REPORT_ROWS = 1_000
DEFAULT_REPORT_LIMIT = 200
MAX_INPUT_OBSERVATIONS = 250_000
MAX_DIMENSION_TEXT_LENGTH = 256
_DIMENSION_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_METRIC_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:+/-]{0,63}$")
_UNIT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:+/%-]{0,63}$")


def validate_metric_labels(quality_metric: str, latency_metric: str) -> None:
    """Validate metric labels shared by the pure engine and database adapter."""
    if not isinstance(quality_metric, str) or not _METRIC_RE.fullmatch(quality_metric):
        raise ValueError("quality_metric must be a short safe metric label")
    if not isinstance(latency_metric, str) or not _METRIC_RE.fullmatch(latency_metric):
        raise ValueError("latency_metric must be a short safe metric label")
    if quality_metric == latency_metric:
        raise ValueError("quality_metric and latency_metric must differ")


@dataclass(frozen=True)
class MeasurementObservation:
    """One already-read measurement joined to its run provenance."""

    run_id: str
    commit_sha: str | None
    measured_at: datetime
    stage: str | None
    metric: str
    value: int | float | None
    unit: str
    dimensions: Mapping[str, Any] | str | None


@dataclass(frozen=True)
class ParetoReport:
    """Rows and coverage/provenance metadata for a Pareto comparison."""

    rows: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready report envelope."""
        return {"name": "quality_latency_pareto", "rows": list(self.rows), "metadata": self.metadata}


def _validate_dimension_keys(
    cohort_dimensions: Sequence[str], candidate_dimensions: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if isinstance(cohort_dimensions, str) or isinstance(candidate_dimensions, str):
        raise TypeError("cohort_dimensions and candidate_dimensions must be sequences of keys")
    cohort = tuple(cohort_dimensions)
    candidate = tuple(candidate_dimensions)
    for label, keys in (("cohort", cohort), ("candidate", candidate)):
        if not keys:
            raise ValueError(f"At least one {label} dimension is required")
        if len(set(keys)) != len(keys):
            raise ValueError(f"{label} dimensions must not contain duplicates")
        if any(not isinstance(key, str) or not _DIMENSION_KEY_RE.fullmatch(key) for key in keys):
            raise ValueError(f"{label} dimension keys must be simple identifier strings")
    if set(cohort) & set(candidate):
        raise ValueError("cohort and candidate dimensions must be disjoint")
    return cohort, candidate


def _validate_date(value: date | None, field: str) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"{field} must be a date or None")
    return value


def _measurement_date(observation: MeasurementObservation) -> date:
    measured_at = observation.measured_at
    if not isinstance(measured_at, datetime):
        raise TypeError("measurement timestamps must be datetime values")
    if measured_at.tzinfo is None or measured_at.utcoffset() is None:
        raise ValueError("measurement timestamps must be timezone-aware")
    return measured_at.astimezone(timezone.utc).date()


def _json_dimensions(value: Mapping[str, Any] | str | None) -> Mapping[str, Any] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, Mapping):
        return None
    return value


def _safe_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, str):
        return (
            1 <= len(value) <= MAX_DIMENSION_TEXT_LENGTH
            and not any(ord(char) < 32 or ord(char) == 127 for char in value)
            and redact_text(value) == value
        )
    if isinstance(value, int):
        return -(2**63) <= value <= 2**63 - 1
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _selected_dimensions(
    observation: MeasurementObservation,
    keys: tuple[str, ...],
) -> tuple[dict[str, Any], tuple[str, ...]] | None:
    dimensions = _json_dimensions(observation.dimensions)
    if dimensions is None:
        return None
    selected: dict[str, Any] = {}
    typed_key: list[str] = []
    for key in keys:
        if key not in dimensions:
            return None
        value = dimensions[key]
        if value is None or not _safe_scalar(value):
            return None
        selected[key] = value
        # JSON encoding keeps values of different JSON scalar types distinct
        # (for example, true and 1) while remaining deterministic.
        typed_key.append(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    return selected, tuple(typed_key)


def _valid_value(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _valid_unit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and _UNIT_RE.fullmatch(value) is not None
        and redact_text(value) == value
    )


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def build_quality_latency_pareto(
    observations: Iterable[MeasurementObservation],
    *,
    quality_metric: str,
    latency_metric: str,
    cohort_dimensions: Sequence[str],
    candidate_dimensions: Sequence[str],
    quality_stage: str | None = None,
    latency_stage: str | None = None,
    quality_direction: str = "higher",
    start_date: date | None = None,
    end_date: date | None = None,
    min_paired_runs: int = 1,
    limit: int = DEFAULT_REPORT_LIMIT,
) -> ParetoReport:
    """Aggregate paired metric observations and classify the Pareto frontier.

    Rows are paired only when ``run_id`` and every selected cohort/candidate
    dimension match exactly. Repetitions are summarized by their per-run
    median; the candidate point is then the median of per-run medians, so runs
    are not weighted by their repetition count.
    """
    validate_metric_labels(quality_metric, latency_metric)
    if quality_direction not in {"higher", "lower"}:
        raise ValueError("quality_direction must be 'higher' or 'lower'")
    if isinstance(min_paired_runs, bool) or not isinstance(min_paired_runs, int) or min_paired_runs < 1:
        raise ValueError("min_paired_runs must be a positive integer")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    start_date = _validate_date(start_date, "start_date")
    end_date = _validate_date(end_date, "end_date")
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("start_date must be on or before end_date")
    cohort_keys, candidate_keys = _validate_dimension_keys(cohort_dimensions, candidate_dimensions)
    applied_limit = min(limit, MAX_REPORT_ROWS)

    source_rows: dict[str, list[tuple[MeasurementObservation, date]]] = {
        quality_metric: [], latency_metric: []
    }
    relevant_observations = 0
    for observation in observations:
        if observation.metric not in source_rows:
            continue
        measured_date = _measurement_date(observation)
        if start_date is not None and measured_date < start_date:
            continue
        if end_date is not None and measured_date > end_date:
            continue
        relevant_observations += 1
        if relevant_observations > MAX_INPUT_OBSERVATIONS:
            raise ValueError(
                f"Pareto report exceeds the {MAX_INPUT_OBSERVATIONS}-observation input limit; "
                "narrow the date range or query filters"
            )
        source_rows[observation.metric].append((observation, measured_date))

    # Refuse implicit stage mixing. Null stage is a distinct stage value.
    for metric, explicit_stage in (
        (quality_metric, quality_stage),
        (latency_metric, latency_stage),
    ):
        stages = {observation.stage for observation, _ in source_rows[metric]}
        if explicit_stage is None and len(stages) > 1:
            raise ValueError(f"{metric!r} appears at multiple stages; specify its stage explicitly")

    selected: dict[
        str,
        list[
            tuple[
                MeasurementObservation,
                date,
                dict[str, Any],
                tuple[str, ...],
                dict[str, Any],
                tuple[str, ...],
            ]
        ],
    ] = {
        quality_metric: [],
        latency_metric: [],
    }
    coverage: dict[str, int | bool] = {
        "quality_records_after_filters": 0,
        "latency_records_after_filters": 0,
        "quality_invalid_values": 0,
        "latency_invalid_values": 0,
        "quality_invalid_dimensions": 0,
        "latency_invalid_dimensions": 0,
        "paired_run_candidate_keys": 0,
        "unpaired_run_candidate_keys": 0,
        "eligible_candidates": 0,
        "candidates_filtered_by_min_paired_runs": 0,
    }
    units_by_cohort: dict[tuple[str, ...], dict[str, set[str]]] = defaultdict(
        lambda: {quality_metric: set(), latency_metric: set()}
    )

    for metric, explicit_stage in (
        (quality_metric, quality_stage),
        (latency_metric, latency_stage),
    ):
        metric_label = "quality" if metric == quality_metric else "latency"
        for observation, measured_date in source_rows[metric]:
            if explicit_stage is not None and observation.stage != explicit_stage:
                continue
            coverage[f"{metric_label}_records_after_filters"] += 1
            if not _valid_value(observation.value):
                coverage[f"{metric_label}_invalid_values"] += 1
                continue
            cohort = _selected_dimensions(observation, cohort_keys)
            candidate = _selected_dimensions(observation, candidate_keys)
            if cohort is None or candidate is None:
                coverage[f"{metric_label}_invalid_dimensions"] += 1
                continue
            if not _valid_unit(observation.unit):
                raise ValueError(f"{metric!r} contains an unsafe or invalid unit")
            cohort_values, cohort_signature = cohort
            candidate_values, candidate_signature = candidate
            units_by_cohort[cohort_signature][metric].add(observation.unit)
            selected[metric].append(
                (
                    observation,
                    measured_date,
                    cohort_values,
                    cohort_signature,
                    candidate_values,
                    candidate_signature,
                )
            )

    for cohort_signature, metric_units in units_by_cohort.items():
        for metric, units in metric_units.items():
            if len(units) > 1:
                raise ValueError(
                    f"{metric!r} has inconsistent units within a comparison cohort: "
                    + ", ".join(sorted(units))
                )

    # One slot per (run, cohort, candidate), with raw repeated values retained
    # until per-run medians have been calculated.
    run_slots: dict[tuple[str, tuple[str, ...], tuple[str, ...]], dict[str, Any]] = {}
    candidate_objects: dict[tuple[tuple[str, ...], tuple[str, ...]], tuple[dict[str, Any], dict[str, Any]]] = {}
    for metric in (quality_metric, latency_metric):
        for (
            observation,
            measured_date,
            cohort_values,
            cohort_signature,
            candidate_values,
            candidate_signature,
        ) in selected[metric]:
            run_key = (observation.run_id, cohort_signature, candidate_signature)
            slot = run_slots.setdefault(
                run_key,
                {
                    quality_metric: [],
                    latency_metric: [],
                    "quality_dates": [],
                    "latency_dates": [],
                    "commits": set(),
                },
            )
            slot[metric].append(float(observation.value))
            slot[f"{'quality' if metric == quality_metric else 'latency'}_dates"].append(measured_date)
            if observation.commit_sha is not None:
                slot["commits"].add(observation.commit_sha)
            point_key = (cohort_signature, candidate_signature)
            candidate_objects[point_key] = (cohort_values, candidate_values)

    paired_slots: dict[tuple[tuple[str, ...], tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    unpaired = 0
    for (run_id, cohort_signature, candidate_signature), slot in run_slots.items():
        has_quality = bool(slot[quality_metric])
        has_latency = bool(slot[latency_metric])
        if not (has_quality and has_latency):
            unpaired += 1
            continue
        paired_slots[(cohort_signature, candidate_signature)].append(
            {
                "run_id": run_id,
                "quality_value": float(median(slot[quality_metric])),
                "latency_value": float(median(slot[latency_metric])),
                "quality_measurements": len(slot[quality_metric]),
                "latency_measurements": len(slot[latency_metric]),
                "dates": slot["quality_dates"] + slot["latency_dates"],
                "commits": slot["commits"],
            }
        )
    coverage["paired_run_candidate_keys"] = sum(len(slots) for slots in paired_slots.values())
    coverage["unpaired_run_candidate_keys"] = unpaired

    point_rows: list[dict[str, Any]] = []
    rows_by_cohort: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for point_key, slots in paired_slots.items():
        if len(slots) < min_paired_runs:
            coverage["candidates_filtered_by_min_paired_runs"] += 1
            continue
        cohort_signature, candidate_signature = point_key
        cohort_object, candidate_object = candidate_objects[point_key]
        units = units_by_cohort[cohort_signature]
        quality_unit = next(iter(units[quality_metric]))
        latency_unit = next(iter(units[latency_metric]))
        dates = [day for slot in slots for day in slot["dates"]]
        commits = sorted({commit for slot in slots for commit in slot["commits"]})
        row: dict[str, Any] = {
            "cohort": cohort_object,
            "candidate": candidate_object,
            "quality_metric": quality_metric,
            "quality_unit": quality_unit,
            "quality_direction": quality_direction,
            "quality_value": float(median(slot["quality_value"] for slot in slots)),
            "latency_metric": latency_metric,
            "latency_unit": latency_unit,
            "latency_value": float(median(slot["latency_value"] for slot in slots)),
            "paired_runs": len(slots),
            "quality_measurements": sum(slot["quality_measurements"] for slot in slots),
            "latency_measurements": sum(slot["latency_measurements"] for slot in slots),
            "pareto_efficient": True,
            "dominated_by": [],
            "date_start": min(dates).isoformat(),
            "date_end": max(dates).isoformat(),
            "commits": commits,
        }
        point_rows.append(row)
        rows_by_cohort[cohort_signature].append(row)

    for cohort_rows in rows_by_cohort.values():
        for candidate_row in cohort_rows:
            dominators = [
                other
                for other in cohort_rows
                if other is not candidate_row and _dominates(other, candidate_row, quality_direction)
            ]
            candidate_row["pareto_efficient"] = not dominators
            candidate_row["dominated_by"] = [
                other["candidate"]
                for other in sorted(dominators, key=lambda item: _canonical(item["candidate"]))
            ]

    point_rows.sort(
        key=lambda row: (
            _canonical(row["cohort"]),
            not row["pareto_efficient"],
            _canonical(row["candidate"]),
        )
    )
    coverage["eligible_candidates"] = len(point_rows)
    rows = tuple(point_rows[:applied_limit])
    truncated = len(point_rows) > len(rows)
    coverage["returned_rows"] = len(rows)
    coverage["truncated"] = truncated
    coverage["requested_limit"] = limit
    coverage["applied_limit"] = applied_limit

    all_dates = [
        datetime.fromisoformat(row["date_start"]).date() for row in point_rows
    ] + [datetime.fromisoformat(row["date_end"]).date() for row in point_rows]
    all_commits = sorted({commit for row in point_rows for commit in row["commits"]})
    metadata = {
        "source_tables": ["measurements", "pipeline_runs"],
        "filters": {
            "quality_metric": quality_metric,
            "latency_metric": latency_metric,
            "quality_stage": quality_stage,
            "latency_stage": latency_stage,
            "cohort_dimensions": list(cohort_keys),
            "candidate_dimensions": list(candidate_keys),
            "quality_direction": quality_direction,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "min_paired_runs": min_paired_runs,
            "requested_limit": limit,
            "applied_limit": applied_limit,
        },
        "coverage": coverage,
        "provenance": {
            "date_start": min(all_dates).isoformat() if all_dates else None,
            "date_end": max(all_dates).isoformat() if all_dates else None,
            "commits": all_commits,
        },
    }
    return ParetoReport(rows=rows, metadata=metadata)


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any], direction: str) -> bool:
    if direction == "higher":
        quality_no_worse = left["quality_value"] >= right["quality_value"]
        quality_better = left["quality_value"] > right["quality_value"]
    else:
        quality_no_worse = left["quality_value"] <= right["quality_value"]
        quality_better = left["quality_value"] < right["quality_value"]
    latency_no_worse = left["latency_value"] <= right["latency_value"]
    latency_better = left["latency_value"] < right["latency_value"]
    return (
        quality_no_worse
        and latency_no_worse
        and (quality_better or latency_better)
    )
