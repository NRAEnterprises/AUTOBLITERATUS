"""Fail-closed promotion of sandbox mapping evidence into runtime capability records."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

PROMOTION_SCHEMA_VERSION = 1
PROMOTION_SCHEMA = {
    "name": "autobliteratus.mapping_capability",
    "version": PROMOTION_SCHEMA_VERSION,
    "required": (
        "model_identity",
        "promotion_ids",
        "replayability",
        "regression",
        "schema_signature",
    ),
}
PROMOTION_SCHEMA_SIGNATURE = hashlib.sha256(
    json.dumps(PROMOTION_SCHEMA, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


class MappingPromotionError(ValueError):
    """Raised when evidence cannot authorize runtime mapping use."""


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _finite_metrics(metrics: Any, label: str, errors: list[str]) -> None:
    if not isinstance(metrics, dict) or not metrics:
        errors.append(f"{label} must be a non-empty metric object")
        return
    for name, value in metrics.items():
        if not isinstance(name, str) or not name:
            errors.append(f"{label} contains an invalid metric name")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            errors.append(f"{label}.{name} must be finite")


def _validate_regression(regression: Any, errors: list[str]) -> dict[str, Any]:
    if not isinstance(regression, dict):
        errors.append("regression report is required")
        return {"status": "missing"}
    if regression.get("status") != "passed":
        errors.append("regression report must have status=passed")
    if regression.get("regressions"):
        errors.append("regression report contains regressions")
    baseline = regression.get("baseline_metrics")
    post = regression.get("post_metrics")
    _finite_metrics(baseline, "regression.baseline_metrics", errors)
    _finite_metrics(post, "regression.post_metrics", errors)
    if isinstance(baseline, dict) and isinstance(post, dict) and set(baseline) != set(post):
        errors.append("regression baseline and post metric names must match")
    metrics = regression.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        errors.append("regression.metrics must be a non-empty metric comparison object")
    elif any(not isinstance(row, dict) or row.get("regressed") is not False for row in metrics.values()):
        errors.append("regression.metrics contains an unverified metric row")
    return {
        "status": regression.get("status"),
        "metric_names": sorted(metrics) if isinstance(metrics, dict) else [],
        "regression_digest": _digest(regression),
    }


def evaluate_mapping_promotion(
    discovery_report: dict[str, Any],
    *,
    replay_results: dict[str, Any],
    regression_report: dict[str, Any],
    expected_identity: dict[str, Any],
    quality_resource_reports: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate evidence and return a signed-by-digest capability record.

    This is a deterministic schema signature, not asymmetric cryptographic
    signing. A future deployment trust root can replace the digest without
    changing the capability contract.
    """
    errors: list[str] = []
    if not isinstance(discovery_report, dict) or discovery_report.get("schema_version") != 1:
        errors.append("discovery report schema_version must be 1")
    if not isinstance(expected_identity, dict) or not expected_identity:
        errors.append("expected_identity must be a non-empty object")
    actual_identity = discovery_report.get("model_identity") if isinstance(discovery_report, dict) else None
    if actual_identity != expected_identity:
        errors.append("discovery model identity does not match expected identity")
    battery = discovery_report.get("unknown_factor_battery")
    evidence_report = battery if isinstance(battery, dict) else discovery_report
    if evidence_report.get("status") != "measured":
        errors.append("discovery battery must be complete with status=measured")
    promotions = evidence_report.get("promotions", [])
    if not isinstance(promotions, list) or not promotions:
        errors.append("at least one verified discovery promotion is required")
        promotions = []
    replay_results = replay_results if isinstance(replay_results, dict) else {}
    promotion_ids: list[str] = []
    replayed: list[dict[str, Any]] = []
    for promotion in promotions:
        if not isinstance(promotion, dict) or promotion.get("status") != "verified":
            errors.append("every promotion must be a verified object")
            continue
        identity = expected_identity.get("identity_sha256")
        category = promotion.get("category")
        path = promotion.get("path")
        experiment_id = promotion.get("experiment_id")
        expected_experiment_id = hashlib.sha256(f"{identity}:{category}:{path}".encode()).hexdigest()
        if experiment_id != expected_experiment_id:
            errors.append(f"promotion experiment_id mismatch for {category}:{path}")
        measurement_signature = promotion.get("measurement_signature")
        if not isinstance(measurement_signature, str) or not measurement_signature:
            errors.append(f"promotion measurement signature missing for {category}:{path}")
        replay = replay_results.get(experiment_id)
        if not isinstance(replay, dict) or replay.get("status") != "passed":
            errors.append(f"promotion replay failed or is missing for {experiment_id}")
        elif replay.get("measurement_signature") != measurement_signature:
            errors.append(f"promotion replay signature mismatch for {experiment_id}")
        else:
            replayed.append({"experiment_id": experiment_id, "status": "passed"})
        promotion_ids.append(experiment_id)
    regression = _validate_regression(regression_report, errors)
    evidence_coordination = None
    if quality_resource_reports is not None:
        from autobliteratus.tuning.promotion_evidence_coordinator import (
            coordinate_promotion_evidence,
        )

        evidence_coordination = coordinate_promotion_evidence(quality_resource_reports)
        if evidence_coordination["status"] != "passed":
            errors.extend(
                f"quality/resource evidence: {error}"
                for error in evidence_coordination["errors"]
            )
    payload = {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "schema_signature": PROMOTION_SCHEMA_SIGNATURE,
        "model_identity": expected_identity,
        "promotion_ids": sorted(promotion_ids),
        "replayability": {"status": "passed" if len(replayed) == len(promotion_ids) and promotion_ids else "rejected", "experiments": replayed},
        "regression": regression,
    }
    if evidence_coordination is not None:
        payload["evidence_coordination"] = evidence_coordination
    payload["capability_digest"] = _digest(payload)
    payload["status"] = "promoted" if not errors else "rejected"
    payload["errors"] = errors
    payload["mutation_authorized"] = payload["status"] == "promoted"
    return payload


def validate_mapping_capability(capability: Any, expected_identity: dict[str, Any] | None = None) -> None:
    """Validate the immutable capability envelope at the mutation boundary."""
    if not isinstance(capability, dict) or capability.get("status") != "promoted":
        raise MappingPromotionError("A promoted mapping capability is required before mutation.")
    if capability.get("schema_version") != PROMOTION_SCHEMA_VERSION:
        raise MappingPromotionError("Unsupported mapping capability schema version.")
    if capability.get("schema_signature") != PROMOTION_SCHEMA_SIGNATURE:
        raise MappingPromotionError("Mapping capability schema signature mismatch.")
    if capability.get("mutation_authorized") is not True:
        raise MappingPromotionError("Mapping capability is not mutation-authorized.")
    if expected_identity is not None and capability.get("model_identity") != expected_identity:
        raise MappingPromotionError("Mapping capability model identity mismatch.")
    payload = {
        key: capability.get(key)
        for key in (
            "schema_version",
            "schema_signature",
            "model_identity",
            "promotion_ids",
            "replayability",
            "regression",
        )
    }
    if "evidence_coordination" in capability:
        payload["evidence_coordination"] = capability["evidence_coordination"]
    if _digest(payload) != capability.get("capability_digest"):
        raise MappingPromotionError("Mapping capability digest mismatch.")
