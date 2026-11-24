from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from autobliteratus.tuning import delta_weight_export_auto_tuning as delta_export
from autobliteratus.tuning.delta_weight_export_auto_tuning import (
    export_weight_delta,
    reconstruct_weight_delta,
)

torch = pytest.importorskip(
    "torch", reason="PyTorch is required for tensor round-trip integration coverage"
)


_IDENTITY = {
    "model": "tiny-tied-pytorch-fixture",
    "revision": "integration-test",
    "identity_sha256": "1" * 64,
}


class TinyTiedModel(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.embedding = torch.nn.Embedding(2, 2, dtype=dtype)
        self.projection = torch.nn.Linear(2, 2, bias=False, dtype=dtype)
        self.projection.weight = self.embedding.weight
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.tensor([[1.0, -2.0], [3.0, 4.0]], dtype=dtype)
            )
        self.register_buffer("scalar", torch.tensor(0.25, dtype=dtype))
        self.register_buffer("counter", torch.tensor([7], dtype=torch.int64))

    def forward(self, token_ids):
        return self.projection(self.embedding(token_ids))


@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
def test_pytorch_tiny_model_delta_round_trip_and_state_reload(
    tmp_path: Path, dtype_name: str
) -> None:
    dtype = getattr(torch, dtype_name)
    baseline = TinyTiedModel(dtype=dtype)
    updated = copy.deepcopy(baseline)

    with torch.no_grad():
        updated.embedding.weight.add_(
            torch.tensor([[0.5, -0.25], [0.125, 1.0]], dtype=dtype)
        )
        updated.scalar.fill_(1.25)

    baseline_state = {
        name: value.detach().clone() for name, value in baseline.state_dict().items()
    }
    expected_state = {
        name: value.detach().clone() for name, value in updated.state_dict().items()
    }
    output = tmp_path / f"delta-{dtype_name}"

    report = export_weight_delta(
        baseline,
        updated,
        output,
        model_identity=_IDENTITY,
    )

    assert report["status"] == "validated"
    assert report["reconstruction_verified"] is True
    assert report["manifest_signed"] is False
    assert report["mutation_authorized"] is False
    assert report["changed_tensor_count"] == 3  # tied weight entries plus scalar buffer
    assert all(
        torch.equal(baseline.state_dict()[name], value)
        for name, value in baseline_state.items()
    )

    manifest = json.loads(
        (output / "autobliteratus_delta.json").read_text(encoding="utf-8")
    )
    assert manifest["tensors"]["embedding.weight"]["dtype"] == dtype_name
    assert manifest["tensors"]["projection.weight"]["dtype"] == dtype_name
    assert manifest["tensors"]["scalar"]["shape"] == []
    assert manifest["tensors"]["counter"]["changed"] is False

    reconstructed = reconstruct_weight_delta(baseline, output)
    reloaded = copy.deepcopy(baseline)
    tensor_state = {
        name: torch.as_tensor(array, dtype=baseline.state_dict()[name].dtype)
        for name, array in reconstructed.items()
    }
    reloaded.load_state_dict(tensor_state, strict=True)

    assert reloaded.embedding.weight is reloaded.projection.weight
    for name, expected in expected_state.items():
        assert torch.equal(reloaded.state_dict()[name], expected)

    # A second reload from the published artifact must also reproduce all state.
    second_reconstruction = reconstruct_weight_delta(baseline, output)
    for name, expected in expected_state.items():
        actual = torch.as_tensor(
            second_reconstruction[name], dtype=reloaded.state_dict()[name].dtype
        )
        assert torch.equal(actual, expected)


def test_pytorch_delta_post_publish_failure_cleans_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = TinyTiedModel(dtype=torch.float32)
    updated = copy.deepcopy(baseline)
    with torch.no_grad():
        updated.embedding.weight.add_(torch.ones_like(updated.embedding.weight) * 0.5)

    output = tmp_path / "delta-failure"
    original_reconstruct = delta_export.reconstruct_weight_delta
    reconstruct_calls = 0

    def fail_on_published_reload(base_state, artifact_dir):
        nonlocal reconstruct_calls
        reconstruct_calls += 1
        if reconstruct_calls == 2:
            raise RuntimeError("simulated post-publish verification failure")
        return original_reconstruct(base_state, artifact_dir)

    monkeypatch.setattr(
        delta_export, "reconstruct_weight_delta", fail_on_published_reload
    )
    with pytest.raises(RuntimeError, match="post-publish verification"):
        export_weight_delta(
            baseline,
            updated,
            output,
            model_identity=_IDENTITY,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".delta-failure.staging-*"))
