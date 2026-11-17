"""Bounded, read-only experiments for unknown mapping factors."""

from __future__ import annotations

import copy
import hashlib
import time
from typing import Any

from autobliteratus.core.model_mapping_discovery import _state_fingerprint


class DiscoveryExperimentError(ValueError):
    """Raised when the discovery experiment contract is invalid."""


def _experiment_id(identity: str, category: str, path: str) -> str:
    payload = f"{identity}:{category}:{path}".encode()
    return hashlib.sha256(payload).hexdigest()


def _candidate_map(report: dict[str, Any], context: dict[str, Any]) -> dict[str, list[str]]:
    supplied = context.get("unknown_factor_candidates")
    if supplied is not None:
        if not isinstance(supplied, dict):
            raise DiscoveryExperimentError("unknown_factor_candidates must be a category-to-path mapping.")
        candidates = supplied
    else:
        candidates = {
            category: area.get("candidate_paths", [])
            for category, area in report.get("areas", {}).items()
            if area.get("status") == "unknown"
        }
    normalized: dict[str, list[str]] = {}
    for category, paths in candidates.items():
        if not isinstance(category, str) or not category:
            raise DiscoveryExperimentError("Unknown-factor categories must be non-empty strings.")
        if not isinstance(paths, (list, tuple)) or any(not isinstance(path, str) or not path for path in paths):
            raise DiscoveryExperimentError(f"Candidate paths for {category!r} must be non-empty strings.")
        normalized[category] = list(dict.fromkeys(paths))
    return normalized


def run_unknown_factor_battery(
    adapter: Any,
    discovery_report: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run bounded read-only removal probes and return promotion candidates.

    An adapter may expose ``sandbox_removal_probe(category, path, context)``.
    The probe must perform its removal experiment in an isolated copy and
    return measured booleans ``removal_supported`` and ``quality_preserved``.
    Missing probes remain unknown. Any mutation observed on the live model
    fails the experiment and can never promote a mapping.
    """
    context = context or {}
    candidates = _candidate_map(discovery_report, context)
    max_experiments = context.get("unknown_factor_max_experiments", 64)
    max_seconds = context.get("unknown_factor_max_seconds", 30.0)
    if isinstance(max_experiments, bool) or not isinstance(max_experiments, int) or not 1 <= max_experiments <= 512:
        raise DiscoveryExperimentError("unknown_factor_max_experiments must be an integer in [1, 512].")
    if isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float)) or not 0 < float(max_seconds) <= 600:
        raise DiscoveryExperimentError("unknown_factor_max_seconds must be finite and in (0, 600].")

    model = adapter.handle.model
    probe = getattr(adapter, "sandbox_removal_probe", None)
    identity = discovery_report["model_identity"]["identity_sha256"]
    started = time.monotonic()
    experiments: list[dict[str, Any]] = []
    promotions: list[dict[str, Any]] = []
    attempted = 0
    truncated = False

    for category, paths in candidates.items():
        for path in paths:
            if attempted >= max_experiments or time.monotonic() - started > float(max_seconds):
                truncated = True
                break
            attempted += 1
            experiment_id = _experiment_id(identity, category, path)
            before = _state_fingerprint(model)
            observation: dict[str, Any] = {
                "experiment_id": experiment_id,
                "category": category,
                "path": path,
                "status": "unknown",
                "live_state_unchanged": None,
            }
            snapshot = None
            try:
                if not callable(probe):
                    observation["reason"] = "sandbox_removal_probe_unavailable"
                else:
                    snapshot = copy.deepcopy(model.state_dict())
                    raw = probe(category, path, dict(context))
                    after_probe = _state_fingerprint(model)
                    observation["live_state_unchanged"] = before == after_probe
                    model.load_state_dict(snapshot, strict=True)
                    after_restore = _state_fingerprint(model)
                    if after_restore != before:
                        raise RuntimeError("live model could not be restored after discovery probe")
                    if not observation["live_state_unchanged"]:
                        observation.update(
                            {
                                "status": "failed",
                                "reason": "live_model_mutated_during_probe",
                                "before_fingerprint": before,
                                "after_fingerprint": after_probe,
                            }
                        )
                    elif not isinstance(raw, dict):
                        observation["reason"] = "probe_result_not_object"
                    else:
                        observation["removal_supported"] = raw.get("removal_supported") is True
                        observation["quality_preserved"] = raw.get("quality_preserved") is True
                        observation["measurement"] = {
                            key: value
                            for key, value in raw.items()
                            if key not in {"user_label", "label", "promotion"}
                        }
                        if observation["removal_supported"] and observation["quality_preserved"]:
                            observation["status"] = "verified"
                            promotions.append(
                                {
                                    "experiment_id": experiment_id,
                                    "category": category,
                                    "path": path,
                                    "status": "verified",
                                    "evidence": "isolated_removal_probe_and_quality_check",
                                }
                            )
                        else:
                            observation["status"] = "unknown"
                            observation["reason"] = "removal_or_quality_not_verified"
            except (RuntimeError, TypeError, ValueError) as exc:
                if snapshot is not None:
                    model.load_state_dict(snapshot, strict=True)
                observation.update(
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            experiments.append(observation)
        if truncated:
            break

    status = "incomplete" if truncated else "measured"
    return {
        "stage_id": "tuning.model_mapping_unknown_factor_battery",
        "schema_version": 1,
        "status": status,
        "model_identity": discovery_report["model_identity"],
        "candidate_count": sum(len(paths) for paths in candidates.values()),
        "attempted_count": attempted,
        "verified_factor_count": len(promotions),
        "unknown_factor_count": sum(item["status"] == "unknown" for item in experiments),
        "failed_factor_count": sum(item["status"] == "failed" for item in experiments),
        "experiments": experiments,
        "promotions": promotions,
        "runtime_pipeline_touched": False,
        "model_modified": False,
        "mutation_performed": False,
        "optimization_performed": False,
        "warnings": [
            "Promotions are evidence-bound and never use user labels.",
            "Unknown and failed factors remain fail-closed.",
        ],
    }


def merge_dynamic_mapping_updates(database: dict[str, Any], report: dict[str, Any]) -> None:
    """Merge only verified experiment promotions into a database object."""
    dynamic = database.setdefault("dynamic_mappings", {})
    if not isinstance(dynamic, dict):
        raise TypeError("Mapping database dynamic_mappings must be an object.")
    identity = report["model_identity"]["identity_sha256"]
    history = dynamic.setdefault(identity, [])
    if not isinstance(history, list):
        raise TypeError("Dynamic mapping history must be a list.")
    existing = {item.get("experiment_id") for item in history if isinstance(item, dict)}
    for promotion in report.get("promotions", []):
        if promotion["experiment_id"] not in existing:
            history.append(promotion)
            existing.add(promotion["experiment_id"])
