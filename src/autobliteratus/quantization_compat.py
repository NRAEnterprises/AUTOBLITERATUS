"""Compatibility helpers for reloading Quanto checkpoints across Transformers versions.

Quanto 0.2.7 materializes a model from ``AutoModelForCausalLM.from_config`` and
loads its checkpoint with ``load_state_dict``. Transformers may rename
checkpoint keys while ``from_pretrained`` loads older architecture formats.
This module reconciles the common output-head alias safely before Quanto's
requantization step, and fails closed if the model still has incompatible keys.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any


class _AliasedStateDict(Mapping[str, Any]):
    """A lazy mapping that exposes one legacy key under the model's expected name."""

    def __init__(self, source: Mapping[str, Any], aliases: dict[str, str]) -> None:
        self._source = source
        self._aliases = aliases

    def __getitem__(self, key: str) -> Any:
        return self._source[self._aliases.get(key, key)]

    def __iter__(self) -> Iterator[str]:
        for key in self._source:
            if key not in self._aliases.values():
                yield key
        for key in self._aliases:
            yield key

    def __len__(self) -> int:
        return len(self._source)

    def __contains__(self, key: object) -> bool:
        if key in self._aliases.values():
            return False
        return key in self._aliases or key in self._source


def _output_weight_alias(model: Any, state_dict: Mapping[str, Any], qmap: dict[str, Any]) -> dict[str, str]:
    """Map a unique serialized output-head weight to the instantiated model key."""
    output = model.get_output_embeddings()
    if output is None or not hasattr(output, "weight") or bool(getattr(model.config, "tie_word_embeddings", False)):
        return {}

    module_name = next(
        (name for name, module in model.named_modules() if module is output),
        None,
    )
    if not module_name:
        return {}
    output_config = qmap.get(module_name, {})
    if isinstance(output_config, dict) and str(output_config.get("weights", "none")).lower() not in {
        "none", "null", "false", ""
    }:
        return {}

    expected_key = f"{module_name}.weight"
    if expected_key in state_dict:
        return {}

    expected_shape = tuple(output.weight.shape)
    # Known HF checkpoint-era aliases avoid scanning/materializing unrelated
    # large tensors. The fallback only examines candidates with output-head
    # names and requires a unique exact shape match.
    leaf = module_name.rsplit(".", 1)[-1]
    candidate_names = {"embed_out", "lm_head", "output_layer", "output"}
    candidate_names.add(leaf)
    candidates: list[str] = []
    for key in state_dict:
        if not key.endswith(".weight") and key != "weight":
            continue
        prefix = key[:-len(".weight")] if key.endswith(".weight") else ""
        if prefix.rsplit(".", 1)[-1] not in candidate_names:
            continue
        if key in model.state_dict():
            continue
        try:
            if tuple(state_dict[key].shape) == expected_shape:
                candidates.append(key)
        except (KeyError, OSError, RuntimeError):
            continue

    if len(candidates) == 1:
        return {expected_key: candidates[0]}
    if len(candidates) > 1:
        raise RuntimeError(
            "Quanto checkpoint contains multiple possible output-head aliases; "
            f"refusing ambiguous remap to {expected_key!r}: {candidates[:5]}"
        )
    return {}


def _tied_output_weight_key(model: Any, state_dict: Mapping[str, Any], qmap: dict[str, Any]) -> str | None:
    """Return a safely omissible tied head key when the embedding alias is saved."""
    if not bool(getattr(model.config, "tie_word_embeddings", False)):
        return None
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if (
        input_embeddings is None
        or output_embeddings is None
        or not hasattr(input_embeddings, "weight")
        or not hasattr(output_embeddings, "weight")
        or tuple(input_embeddings.weight.shape) != tuple(output_embeddings.weight.shape)
    ):
        return None

    module_names = {id(module): name for name, module in model.named_modules()}
    input_name = module_names.get(id(input_embeddings))
    output_name = module_names.get(id(output_embeddings))
    if not input_name or not output_name:
        return None
    output_config = qmap.get(output_name, {})
    if isinstance(output_config, dict) and str(output_config.get("weights", "none")).lower() not in {
        "none", "null", "false", ""
    }:
        return None

    input_key = f"{input_name}.weight"
    output_key = f"{output_name}.weight"
    if output_key in state_dict or input_key not in state_dict:
        return None
    try:
        if tuple(state_dict[input_key].shape) == tuple(output_embeddings.weight.shape):
            return output_key
    except (KeyError, OSError, RuntimeError):
        return None
    return None


def load_quantized_model(
    model_path: str | os.PathLike[str],
    *,
    device: str | None = None,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    revision: str | None = None,
    **kwargs: Any,
) -> Any:
    """Load a Quanto causal-LM artifact with legacy output-head key repair.

    The loader mirrors Quanto's supported Transformers path, but checks the
    state-dict compatibility result rather than silently accepting missing
    output-head weights. It works with both single-file and sharded safetensors.
    """
    try:
        import torch
        from accelerate import init_empty_weights
        from huggingface_hub import snapshot_download
        from optimum.quanto import requantize
        from optimum.quanto.models.shared_dict import ShardedStateDict
        from optimum.quanto.models.transformers_models import QuantizedModelForCausalLM
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.modeling_utils import (
            get_checkpoint_shard_files,
            load_state_dict,
        )
        from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME
    except ImportError as exc:
        raise RuntimeError(
            "Quanto artifact loading requires Transformers, Accelerate, and Optimum Quanto."
        ) from exc

    path = Path(model_path).expanduser()
    if path.is_dir():
        working_dir = path
    else:
        if not revision:
            raise ValueError("Remote quantized model loads require an immutable revision (commit SHA).")
        working_dir = Path(
            snapshot_download(
                str(model_path),
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        )

    qmap_path = working_dir / QuantizedModelForCausalLM._qmap_name()
    if not qmap_path.is_file():
        raise ValueError(f"No Quanto quantization map found in {working_dir}.")
    qmap = json.loads(qmap_path.read_text(encoding="utf-8"))
    if not isinstance(qmap, dict) or not qmap:
        raise ValueError("Quanto quantization map is empty or invalid.")

    config = AutoConfig.from_pretrained(
        working_dir,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
        revision="local",  # nosec B615: working_dir is an already downloaded local artifact.
    )
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)

    index_path = working_dir / SAFE_WEIGHTS_INDEX_NAME
    if index_path.is_file():
        _checkpoint_files, metadata = get_checkpoint_shard_files(str(working_dir), str(index_path))
        state_dict: Mapping[str, Any] = ShardedStateDict(
            str(working_dir), metadata["weight_map"]
        )
    else:
        checkpoint_path = working_dir / SAFE_WEIGHTS_NAME
        if not checkpoint_path.is_file():
            raise ValueError(f"No safetensors checkpoint found in {working_dir}.")
        state_dict = load_state_dict(str(checkpoint_path))

    aliases = _output_weight_alias(model, state_dict, qmap)
    if aliases:
        state_dict = _AliasedStateDict(state_dict, aliases)

    load_result: dict[str, Any] = {}
    original_load_state_dict = model.load_state_dict

    def capture_load_result(*args: Any, **load_kwargs: Any):
        result = original_load_state_dict(*args, **load_kwargs)
        load_result["result"] = result
        return result

    model.load_state_dict = capture_load_result
    target_device = torch.device(device) if device and device != "auto" else None
    requantize(model, state_dict=state_dict, quantization_map=qmap, device=target_device)
    model.load_state_dict = original_load_state_dict

    result = load_result.get("result")
    if result is None:
        raise RuntimeError("Quanto did not return a state-dict compatibility result.")
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    tied_output_key = _tied_output_weight_key(model, state_dict, qmap)
    if tied_output_key is not None:
        missing = [key for key in missing if key != tied_output_key]
    if missing or unexpected:
        raise RuntimeError(
            "Quanto checkpoint does not match the instantiated architecture after compatibility remapping; "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    if getattr(model.config, "tie_word_embeddings", True):
        model.tie_weights()
    model.eval()
    return QuantizedModelForCausalLM(model)
