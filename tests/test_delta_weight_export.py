from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file

from autobliteratus.tuning._95_delta_weight_export_auto_tuning import (
    DeltaWeightExportAutoTuning,
)
from autobliteratus.tuning.delta_weight_export_auto_tuning import (
    export_weight_delta,
    reconstruct_weight_delta,
)

_IDENTITY = {"model": "tiny-fixture", "revision": "test", "identity_sha256": "0" * 64}


def _states() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    base = {
        "model.layers.0.weight": np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32),
        "model.layers.0.bias": np.array([0.5, -0.25], dtype=np.float16),
        "model.layers.0.counter": np.array([7, 8], dtype=np.int64),
        "model.scale": np.array(0.25, dtype=np.float32),
    }
    updated = {name: value.copy() for name, value in base.items()}
    updated["model.layers.0.weight"][0, 1] = -1.5
    updated["model.layers.0.bias"][1] = 0.75
    updated["model.scale"] = np.array(1.25, dtype=np.float32)
    return base, updated


def test_delta_is_written_reloaded_and_exactly_reconstructed(tmp_path: Path) -> None:
    base, updated = _states()
    expected_base = {name: value.copy() for name, value in base.items()}
    output = tmp_path / "delta"

    report = export_weight_delta(base, updated, output, model_identity=_IDENTITY)

    assert report["status"] == "validated"
    assert report["changed_tensor_count"] == 3
    assert report["tensor_count"] == 4
    assert report["reconstruction_verified"] is True
    assert report["manifest_signed"] is False
    assert report["mutation_authorized"] is False
    assert all(np.array_equal(base[name], expected_base[name]) for name in base)

    manifest = json.loads((output / "autobliteratus_delta.json").read_text(encoding="utf-8"))
    assert manifest["artifact_kind"] == "autobliteratus_weight_delta"
    assert manifest["model_identity"] == _IDENTITY
    delta_tensors = load_file(str(output / "delta.safetensors"))
    assert set(delta_tensors) == {
        "model.layers.0.weight",
        "model.layers.0.bias",
        "model.scale",
    }
    assert delta_tensors["model.scale"].shape == ()

    reconstructed = reconstruct_weight_delta(base, output)
    assert set(reconstructed) == set(updated)
    for name, expected in updated.items():
        assert np.array_equal(reconstructed[name], expected)


def test_changed_integer_tensor_is_rejected_without_publishing(tmp_path: Path) -> None:
    base = {"counter": np.array([1, 2], dtype=np.int64)}
    updated = {"counter": np.array([1, 3], dtype=np.int64)}
    output = tmp_path / "delta"

    with pytest.raises(ValueError, match="non-floating tensor"):
        export_weight_delta(base, updated, output, model_identity=_IDENTITY)

    assert not output.exists()
    assert not list(tmp_path.glob(".delta.staging-*"))


def test_noop_and_nonfinite_states_are_rejected(tmp_path: Path) -> None:
    unchanged = {"weight": np.array([1.0, 2.0], dtype=np.float32)}
    with pytest.raises(ValueError, match="No changed tensors"):
        export_weight_delta(unchanged, unchanged, tmp_path / "noop", model_identity=_IDENTITY)

    invalid = {"weight": np.array([1.0, np.inf], dtype=np.float32)}
    with pytest.raises(ValueError, match="Non-finite"):
        export_weight_delta(invalid, unchanged, tmp_path / "nonfinite", model_identity=_IDENTITY)


def test_key_shape_and_dtype_changes_are_rejected(tmp_path: Path) -> None:
    base = {"weight": np.array([1.0, 2.0], dtype=np.float32)}
    with pytest.raises(ValueError, match="keys differ"):
        export_weight_delta(
            base, {"other": base["weight"] + 1}, tmp_path / "keys", model_identity=_IDENTITY
        )
    with pytest.raises(ValueError, match="shape changed"):
        export_weight_delta(
            base,
            {"weight": np.ones((1, 2), dtype=np.float32)},
            tmp_path / "shape",
            model_identity=_IDENTITY,
        )
    with pytest.raises(ValueError, match="dtype changed"):
        export_weight_delta(
            base,
            {"weight": np.ones(2, dtype=np.float64)},
            tmp_path / "dtype",
            model_identity=_IDENTITY,
        )

    assert not (tmp_path / "keys").exists()
    assert not (tmp_path / "shape").exists()
    assert not (tmp_path / "dtype").exists()


def test_reload_rejects_a_different_baseline_and_corrupt_artifact(tmp_path: Path) -> None:
    base, updated = _states()
    output = tmp_path / "delta"
    export_weight_delta(base, updated, output, model_identity=_IDENTITY)

    other_base = {name: value.copy() for name, value in base.items()}
    other_base["model.layers.0.weight"][0, 0] += 1
    with pytest.raises(ValueError, match="different baseline"):
        reconstruct_weight_delta(other_base, output)

    weights = output / "delta.safetensors"
    weights.write_bytes(weights.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        reconstruct_weight_delta(base, output)


def test_existing_output_is_never_replaced_and_wrapper_stays_disabled(tmp_path: Path) -> None:
    base, updated = _states()
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        export_weight_delta(base, updated, output, model_identity=_IDENTITY)

    wrapper = DeltaWeightExportAutoTuning(work_dir=str(tmp_path / "disabled"))
    assert wrapper.run(base, updated) == Path()
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / "disabled").exists()


def test_model_identity_must_contain_a_sha256_binding(tmp_path: Path) -> None:
    base, updated = _states()
    with pytest.raises(ValueError, match="identity_sha256"):
        export_weight_delta(base, updated, tmp_path / "delta", model_identity={"model": "fixture"})
