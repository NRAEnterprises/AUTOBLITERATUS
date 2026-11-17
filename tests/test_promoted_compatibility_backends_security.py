from __future__ import annotations

from types import SimpleNamespace

import pytest

from autobliteratus.tuning._47_tokenization_consistency_auto_tuning import (
    run as run_tokenization,
)
from autobliteratus.tuning._70_version_compatibility_auto_tuning import (
    run as run_compatibility,
)


class _Tokenizer:
    def __init__(self, offset: int = 0) -> None:
        self.offset = offset

    def encode(self, text: str) -> list[int]:
        return [len(text) + self.offset]


def test_tokenization_drift_examples_are_redacted() -> None:
    context = {
        "tokenizer_before": _Tokenizer(),
        "tokenizer_after": _Tokenizer(offset=1),
        "tokenization_corpus": ["sensitive prompt"],
    }
    report = run_tokenization(SimpleNamespace(), context)
    assert report["status"] == "measured"
    assert report["drift_count"] == 1
    assert "text" not in report["examples"][0]
    assert len(report["examples"][0]["text_sha256"]) == 64


def test_tokenization_rejects_unbounded_sample_count() -> None:
    context = {
        "tokenizer_before": _Tokenizer(),
        "tokenizer_after": _Tokenizer(),
        "tokenization_corpus": ["x"],
        "tokenization_max_samples": 10_001,
    }
    with pytest.raises(ValueError, match="10000"):
        run_tokenization(SimpleNamespace(), context)


def test_tokenization_rejects_oversized_text() -> None:
    context = {
        "tokenizer_before": _Tokenizer(),
        "tokenizer_after": _Tokenizer(),
        "tokenization_corpus": ["x" * 1_000_001],
    }
    with pytest.raises(ValueError, match="1000000"):
        run_tokenization(SimpleNamespace(), context)


def test_compatibility_requires_existing_export_directory(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        run_compatibility(
            SimpleNamespace(),
            {
                "exported_path": str(tmp_path / "missing"),
                "compatibility_loaders": {"transformers": lambda path: None},
            },
        )
