from __future__ import annotations

from autobliteratus.tuning.mapping_promotion_gate import (
    PROMOTION_SCHEMA_SIGNATURE,
    MappingPromotionError,
    evaluate_mapping_promotion,
    validate_mapping_capability,
)


def _inputs():
    identity = {"identity_sha256": "model-identity", "state_fingerprint": "state-1"}
    category = "attention"
    path = "model.layers[].self_attn.o_proj"
    from hashlib import sha256

    experiment_id = sha256(f"{identity['identity_sha256']}:{category}:{path}".encode()).hexdigest()
    measurement_signature = "measurement-1"
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
    replay = {
        experiment_id: {"status": "passed", "measurement_signature": measurement_signature}
    }
    regression = {
        "status": "passed",
        "regressions": [],
        "baseline_metrics": {"mean_nll": 1.0},
        "post_metrics": {"mean_nll": 1.01},
        "metrics": {"mean_nll": {"regressed": False}},
    }
    return identity, discovery, replay, regression, experiment_id


def test_promotion_requires_replay_and_regression_evidence() -> None:
    identity, discovery, replay, regression, experiment_id = _inputs()
    capability = evaluate_mapping_promotion(
        discovery,
        replay_results=replay,
        regression_report=regression,
        expected_identity=identity,
    )
    assert capability["status"] == "promoted"
    assert capability["schema_signature"] == PROMOTION_SCHEMA_SIGNATURE
    assert capability["promotion_ids"] == [experiment_id]
    validate_mapping_capability(capability, identity)


def test_successful_identity_bound_promotion_record_creation() -> None:
    identity, discovery, replay, regression, experiment_id = _inputs()

    capability = evaluate_mapping_promotion(
        discovery,
        replay_results=replay,
        regression_report=regression,
        expected_identity=identity,
    )

    assert capability["status"] == "promoted"
    assert capability["mutation_authorized"] is True
    assert capability["model_identity"] == identity
    assert capability["promotion_ids"] == [experiment_id]
    assert capability["replayability"] == {
        "status": "passed",
        "experiments": [{"experiment_id": experiment_id, "status": "passed"}],
    }
    assert capability["regression"]["status"] == "passed"
    assert capability["capability_digest"]
    validate_mapping_capability(capability, expected_identity=identity)


def test_promotion_embeds_complete_quality_resource_coordination() -> None:
    identity, discovery, replay, regression, _ = _inputs()
    reports = {
        "capability_report": {"status": "measured"},
        "safety_report": {"status": "passed"},
        "latency_report": {"status": "measured"},
        "throughput_report": {"status": "measured"},
        "memory_report": {"status": "measured"},
        "kv_cache_report": {"status": "measured"},
        "regression_report": {"status": "passed"},
        "resource_budget_report": {"status": "within_budget"},
    }

    capability = evaluate_mapping_promotion(
        discovery,
        replay_results=replay,
        regression_report=regression,
        expected_identity=identity,
        quality_resource_reports=reports,
    )

    assert capability["status"] == "promoted"
    assert capability["evidence_coordination"]["status"] == "passed"
    assert capability["evidence_coordination"]["mutation_authorized"] is False
    validate_mapping_capability(capability, expected_identity=identity)


def test_full_discovery_envelope_uses_nested_battery() -> None:
    identity, discovery, replay, regression, _ = _inputs()
    envelope = {
        "schema_version": 1,
        "status": "persisted",
        "model_identity": identity,
        "unknown_factor_battery": discovery,
    }
    capability = evaluate_mapping_promotion(
        envelope,
        replay_results=replay,
        regression_report=regression,
        expected_identity=identity,
    )
    assert capability["status"] == "promoted"


def test_promotion_rejects_replay_signature_tampering() -> None:
    identity, discovery, replay, regression, experiment_id = _inputs()
    replay[experiment_id]["measurement_signature"] = "tampered"
    capability = evaluate_mapping_promotion(
        discovery,
        replay_results=replay,
        regression_report=regression,
        expected_identity=identity,
    )
    assert capability["status"] == "rejected"
    assert any("signature mismatch" in error for error in capability["errors"])


def test_promotion_rejects_regression_and_identity_failures() -> None:
    identity, discovery, replay, regression, _ = _inputs()
    regression["status"] = "regression"
    regression["regressions"] = ["mean_nll"]
    capability = evaluate_mapping_promotion(
        discovery,
        replay_results=replay,
        regression_report=regression,
        expected_identity={**identity, "state_fingerprint": "stale"},
    )
    assert capability["status"] == "rejected"
    assert any("identity" in error for error in capability["errors"])
    assert any("status=passed" in error for error in capability["errors"])


def test_capability_digest_and_schema_are_fail_closed() -> None:
    identity, discovery, replay, regression, _ = _inputs()
    capability = evaluate_mapping_promotion(
        discovery,
        replay_results=replay,
        regression_report=regression,
        expected_identity=identity,
    )
    capability["promotion_ids"] = []
    try:
        validate_mapping_capability(capability, identity)
    except MappingPromotionError as exc:
        assert "digest mismatch" in str(exc)
    else:
        raise AssertionError("tampered capability must be rejected")
