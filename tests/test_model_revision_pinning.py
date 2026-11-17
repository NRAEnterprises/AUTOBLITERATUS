from __future__ import annotations

import pytest

from autobliteratus.cli import parse_args
from autobliteratus.core.loader import load_model
from autobliteratus.pipeline import run_pipeline


def test_cli_parses_model_revision() -> None:
    args = parse_args(["--model-path", "org/model", "--model-revision", "a" * 40])
    assert args.model_revision == "a" * 40


def test_plan_only_preserves_model_revision() -> None:
    result = run_pipeline(
        model_path="org/model",
        model_revision="b" * 40,
        plan_only=True,
    )
    assert result["plan"]["model_revision"] == "b" * 40


def test_remote_model_load_requires_immutable_revision() -> None:
    with pytest.raises(ValueError, match="immutable.*revision"):
        load_model("org/model", device="cpu")
