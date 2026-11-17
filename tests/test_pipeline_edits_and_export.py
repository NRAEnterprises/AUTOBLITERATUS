from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from autobliteratus.core.projection import project_out
from autobliteratus.tuning._00_refusal_direction_ablation import run as run_ablation
from autobliteratus.tuning._96_model_export import (
    _publish_without_replacing,
    _validate_disjoint_paths,
)
from autobliteratus.tuning.dispatch import build_registry
from autobliteratus.tuning.mutation_firewall import MutationAuthorizationError
from autobliteratus.tuning.toggle_registry import TOGGLES


def test_dispatch_registers_only_explicit_callable_stages() -> None:
    registry = build_registry()
    assert callable(registry["tuning.refusal_direction_ablation"])
    assert callable(registry["tuning.model_export"])
    assert callable(registry["tuning.anti_router_collapse"])
    assert callable(registry["tuning.multi_objective_auto"])
    assert callable(registry["tuning.auto_security_scanning"])
    assert callable(registry["tuning.baseline_auto_pre_post"])
    assert callable(registry["tuning.benchmark_regression_auto"])
    assert callable(registry["tuning.audit_trail_auto"])
    assert "tuning.pipeline_entry" not in registry


def test_class_only_toggle_defaults_off_but_remains_visible() -> None:
    assert "tuning.post_ablation_capability" in TOGGLES
    assert TOGGLES["tuning.post_ablation_capability"]["available"] is False
    assert TOGGLES["tuning.post_ablation_capability"]["default"] is False


def test_projection_changes_weights_without_building_dense_projection() -> None:
    linear = torch.nn.Linear(3, 3, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 1.0], [0.0, 2.0, 5.0]]))
    layer = SimpleNamespace(self_attn=SimpleNamespace(o_proj=linear))
    model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
    handle = SimpleNamespace(model=model)
    before = linear.weight.detach().clone()

    assert project_out(handle, 0, torch.tensor([1.0, 0.0, 0.0]), strength=1.0)
    assert not torch.equal(before, linear.weight)
    # Repeating the projection after the component is removed must be a no-op.
    assert not project_out(handle, 0, torch.tensor([1.0, 0.0, 0.0]), strength=1.0)


def test_ablation_stage_is_blocked_without_mutation_authorization() -> None:
    class FakeAdapter:
        def refusal_directions(self):
            return {"directions": {0: torch.tensor([1.0, 0.0])}}

        def project(self, *args, **kwargs):
            return False

    context = {"ablation_strength": 1.0, "ablation_component": "attn.out"}
    with pytest.raises(MutationAuthorizationError, match="requires allow_mutation"):
        run_ablation(FakeAdapter(), context)
    assert context.get("model_modified", False) is False


def test_output_path_must_be_disjoint_from_local_source(tmp_path: Path) -> None:
    source = tmp_path / "original"
    source.mkdir()
    with pytest.raises(ValueError, match="overlaps source"):
        _validate_disjoint_paths(str(source), source / "modified")
    with pytest.raises(ValueError, match="overlaps source"):
        _validate_disjoint_paths(str(source), tmp_path)


def test_output_path_allows_sibling_directory(tmp_path: Path) -> None:
    source = tmp_path / "original"
    source.mkdir()
    _validate_disjoint_paths(str(source), tmp_path / "modified")


def test_publish_is_atomic_and_never_replaces_an_existing_directory(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "manifest.json").write_text("first", encoding="utf-8")
    target = tmp_path / "published"
    _publish_without_replacing(staging, target)
    assert (target / "manifest.json").read_text(encoding="utf-8") == "first"

    replacement = tmp_path / "replacement"
    replacement.mkdir()
    with pytest.raises(FileExistsError):
        _publish_without_replacing(replacement, target)
    assert (target / "manifest.json").read_text(encoding="utf-8") == "first"
