from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from autobliteratus.tuning import _96_model_export as model_export
from autobliteratus.tuning._66_weight_fingerprint_auto_tuning import (
    run as run_fingerprint,
)
from autobliteratus.tuning._88_provenance_recording_auto_tuning import (
    fingerprint_model,
)
from autobliteratus.tuning._88_provenance_recording_auto_tuning import (
    run as run_provenance,
)


class _Adapter:
    def __init__(self, model) -> None:
        self.handle = SimpleNamespace(
            model=model,
            config=SimpleNamespace(model_type="fixture", hidden_size=2),
            architecture="fixture",
        )


def test_fingerprint_binds_parameter_names_shapes_and_values() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    first_digest, first_count = fingerprint_model(model)

    with torch.no_grad():
        model.weight[0, 0] += 1.0
    second_digest, second_count = fingerprint_model(model)

    assert first_count == second_count == 1
    assert first_digest != second_digest


def test_fingerprint_rejects_empty_state_dict() -> None:
    class EmptyModel:
        def state_dict(self):
            return {}

    with pytest.raises(RuntimeError, match="empty state dictionary"):
        fingerprint_model(EmptyModel())


def test_provenance_binds_revision_config_and_calibration(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration.txt"
    calibration.write_text("representative row\n", encoding="utf-8")
    context = {
        "model_path": "fixture-model",
        "model_revision": "commit-sha-123",
        "calibration_data": str(calibration),
        "calibration_samples": 1,
        "calibration_max_length": 16,
        "calibration_momentum": 0.9,
        "calibration_streamline": True,
        "quantization": "none",
        "target_frameworks": ["transformers"],
        "deployment_target": "server",
        "device": "cpu",
        "dtype": "float32",
        "_resolved_toggles": {"tuning.model_export": True, "tuning.provenance_recording_auto": True},
    }

    report = run_provenance(_Adapter(torch.nn.Linear(2, 2, bias=False)), context)

    assert report["source_model_revision"] == "commit-sha-123"
    assert report["model_config_sha256"]
    assert report["model_state_sha256"]
    assert report["calibration_sha256"]
    assert report["enabled_stages"] == [
        "tuning.model_export",
        "tuning.provenance_recording_auto",
    ]
    assert context["provenance_report"] is report


def test_weight_fingerprint_stage_exposes_hash_and_tensor_count() -> None:
    context = {}
    report = run_fingerprint(_Adapter(torch.nn.Linear(2, 2, bias=False)), context)

    assert len(report["sha256"]) == 64
    assert report["tensor_count"] == 1
    assert report["stage"] == "final-pre-export"
    assert context["weight_fingerprint"] is report


@pytest.mark.parametrize(
    ("files", "message"),
    [
        (["tokenizer.json"], "config.json is missing"),
        (["config.json", "tokenizer.json"], "no non-empty safetensors"),
    ],
)
def test_export_validation_rejects_incomplete_artifacts(tmp_path: Path, files, message) -> None:
    for name in files:
        path = tmp_path / name
        path.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        model_export._verify_export(
            model=torch.nn.Linear(2, 2),
            tokenizer=SimpleNamespace(),
            output_dir=tmp_path,
            trust_remote_code=False,
        )


def test_export_validation_requires_tokenizer_artifact(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"not-a-real-safetensor")

    with pytest.raises(RuntimeError, match="tokenizer files are missing"):
        model_export._verify_export(
            model=torch.nn.Linear(2, 2),
            tokenizer=SimpleNamespace(),
            output_dir=tmp_path,
            trust_remote_code=False,
        )


def test_export_removes_staging_directory_after_validation_failure(monkeypatch, tmp_path: Path) -> None:
    staging_paths: list[Path] = []

    def fake_save(model, output_dir, max_shard_size):
        staging_paths.append(output_dir)
        (output_dir / "config.json").write_text("{}", encoding="utf-8")

    class _Tokenizer:
        def save_pretrained(self, output_dir):
            (output_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(model_export, "_save_model_for_export", fake_save)
    adapter = _Adapter(torch.nn.Linear(2, 2, bias=False))
    adapter.handle.tokenizer = _Tokenizer()

    with pytest.raises(RuntimeError, match="no non-empty safetensors"):
        model_export.export_model(
            adapter,
            {
                "model_path": "fixture-model",
                "output_dir": str(tmp_path / "published"),
                "trust_remote_code": False,
            },
        )

    assert staging_paths
    assert not staging_paths[0].exists()
    assert not (tmp_path / "published").exists()
