from __future__ import annotations

from types import SimpleNamespace

import pytest

from autobliteratus.tuning._47_tokenization_consistency_auto_tuning import (
    run as run_tokenization,
)
from autobliteratus.tuning._70_version_compatibility_auto_tuning import (
    run as run_compatibility,
)
from autobliteratus.tuning.toggle_registry import TOGGLES


class _Tokenizer:
    def __init__(self, drift: bool = False) -> None:
        self.drift = drift

    def encode(self, text: str) -> list[int]:
        values = [len(part) for part in text.split()]
        return values[:-1] + [values[-1] + 1] if self.drift and values else values


def test_promoted_registry_entries_are_callable() -> None:
    assert TOGGLES["tuning.version_compatibility_auto"]["available"] is True
    assert TOGGLES["tuning.tokenization_consistency_auto"]["available"] is True


def test_version_compatibility_reports_each_framework_without_mutation(tmp_path) -> None:
    calls: list[tuple[str, str]] = []
    artifact = tmp_path / "model"
    artifact.mkdir()

    def load_transformers(path: str) -> None:
        calls.append(("transformers", path))

    def load_broken(path: str) -> None:
        calls.append(("broken", path))
        raise RuntimeError("backend unavailable")

    context = {
        "exported_path": str(artifact),
        "compatibility_loaders": {"transformers": load_transformers, "broken": load_broken},
    }
    report = run_compatibility(SimpleNamespace(), context)
    assert report["status"] == "measured"
    assert report["per_framework"] == {"broken": "fail: RuntimeError", "transformers": "ok"}
    assert report["model_modified"] is False
    assert len(calls) == 2


def test_version_compatibility_fails_closed_on_bad_loader_contract() -> None:
    with pytest.raises(TypeError, match="mapping"):
        run_compatibility(SimpleNamespace(), {"compatibility_loaders": []})


def test_tokenization_consistency_measures_drift_and_preserves_model() -> None:
    context = {
        "tokenizer_before": _Tokenizer(),
        "tokenizer_after": _Tokenizer(drift=True),
        "tokenization_corpus": ["one two", "three"],
    }
    report = run_tokenization(SimpleNamespace(), context)
    assert report["status"] == "measured"
    assert report["drift_count"] == 2
    assert report["samples_checked"] == 2
    assert report["model_modified"] is False
    assert context["tokenization_consistency_report"] is report


def test_tokenization_consistency_requires_explicit_baseline() -> None:
    report = run_tokenization(SimpleNamespace(), {})
    assert report["status"] == "not_evaluated"
