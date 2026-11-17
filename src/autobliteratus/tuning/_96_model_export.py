"""Export and validate the fully transformed model into a new directory."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOGGLE = {
    "id": "tuning.model_export",
    "default": True,
    "desc": "Validate and atomically save a separate full Transformers model copy.",
}


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_disjoint_paths(model_path: str, output_dir: Path) -> None:
    source = Path(model_path).expanduser()
    if not source.exists():
        # Hugging Face repository IDs are not local filesystem paths.
        return
    source = source.resolve()
    target = output_dir.resolve()
    if source.is_dir():
        if _is_relative_to(target, source) or _is_relative_to(source, target):
            raise ValueError(
                f"Output directory {target} overlaps source model directory {source}; "
                "choose a separate sibling or unrelated path."
            )
    elif source == target:
        raise ValueError("Output directory must not be the source model path.")


def _publish_without_replacing(source: Path, target: Path) -> None:
    """Atomically publish a directory without replacing an existing path."""
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Output directory already exists: {target}")

    # Linux renameat2(RENAME_NOREPLACE) closes the check/rename race. Fall
    # back to a second existence check on platforms that lack the syscall.
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is not None:
            renameat2.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            renameat2.restype = ctypes.c_int
            result = renameat2(
                -100,
                os.fsencode(source),
                -100,
                os.fsencode(target),
                1,  # RENAME_NOREPLACE
            )
            if result == 0:
                return
            error_number = ctypes.get_errno()
            if error_number == errno.EEXIST:
                raise FileExistsError(error_number, os.strerror(error_number), str(target))
            if error_number not in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP):
                raise OSError(error_number, os.strerror(error_number), str(target))
    except AttributeError:
        pass

    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Output directory already exists: {target}")
    os.rename(source, target)


def _save_model_for_export(model, output_dir: Path, max_shard_size: str) -> None:
    """Save standard Transformers models and Quanto wrappers with real shard control."""
    wrapped = getattr(model, "_wrapped", None)
    if wrapped is None or not type(model).__module__.startswith("optimum.quanto.models"):
        model.save_pretrained(
            output_dir,
            safe_serialization=True,
            max_shard_size=max_shard_size,
        )
        return

    # Quanto 0.2.7's ModelHubMixin serializer accepts arbitrary keyword args
    # but its private implementation drops max_shard_size. Save the underlying
    # PreTrainedModel directly, then preserve the matching quantization map.
    try:
        from optimum.quanto import quantization_map
        from optimum.quanto.nn import QModuleMixin
    except ImportError as exc:
        raise RuntimeError("Exporting a Quanto model requires optimum-quanto.") from exc
    if getattr(wrapped.config, "tie_word_embeddings", True) and (
        isinstance(wrapped.get_input_embeddings(), QModuleMixin)
        or isinstance(wrapped.get_output_embeddings(), QModuleMixin)
    ):
        wrapped.config.tie_word_embeddings = False
    wrapped.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    qmap_path = output_dir / "quanto_qmap.json"
    qmap_path.write_text(json.dumps(quantization_map(wrapped), indent=2) + "\n", encoding="utf-8")


def _tensor_matches_saved_slice(saved_slice, live_tensor) -> bool:
    """Compare saved values to the live tensor without loading a whole shard."""
    import torch

    shape = tuple(live_tensor.shape)
    if not shape:
        return torch.equal(saved_slice[()], live_tensor.detach().to(device="cpu"))

    # Bound each temporary CPU copy to roughly 8 Mi elements even for very
    # wide embeddings/projections; this avoids a second model-sized allocation.
    trailing_elements = max(1, math.prod(shape[1:]))
    rows_per_chunk = max(1, min(shape[0], (8 * 1024 * 1024) // trailing_elements))
    for start in range(0, shape[0], rows_per_chunk):
        end = min(shape[0], start + rows_per_chunk)
        selector = (slice(start, end),) + (slice(None),) * (len(shape) - 1)
        saved_chunk = saved_slice[selector]
        live_chunk = live_tensor.detach()[start:end].to(device="cpu")
        if not torch.equal(saved_chunk, live_chunk):
            return False
    return True


def _verify_export(
    model,
    tokenizer,
    output_dir: Path,
    trust_remote_code: bool,
    expected_quantization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_file = output_dir / "config.json"
    if not config_file.is_file() or config_file.stat().st_size == 0:
        raise RuntimeError("Export validation failed: config.json is missing or empty.")

    weight_files = sorted(output_dir.glob("*.safetensors"))
    if not weight_files or any(not path.is_file() or path.stat().st_size == 0 for path in weight_files):
        raise RuntimeError("Export validation failed: no non-empty safetensors weight files were produced.")

    tokenizer_files = [
        output_dir / name
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "vocab.txt",
            "tokenizer.model",
            "spiece.model",
            "sentencepiece.bpe.model",
            "merges.txt",
        )
    ]
    if not any(path.is_file() and path.stat().st_size > 0 for path in tokenizer_files):
        raise RuntimeError("Export validation failed: tokenizer files are missing.")

    # Validate local config/tokenizer readability without reloading a second
    # copy of the model weights into RAM.
    from transformers import AutoConfig, AutoTokenizer

    AutoConfig.from_pretrained(  # nosec B615: local_files_only prevents Hub downloads.
        output_dir,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
        revision="local",
    )
    AutoTokenizer.from_pretrained(  # nosec B615: local_files_only prevents Hub downloads.
        output_dir,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
        revision="local",
    )

    from safetensors import safe_open

    quanto_map_file = output_dir / "quanto_qmap.json"
    if quanto_map_file.is_file():
        try:
            quanto_map = json.loads(quanto_map_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("Export validation failed: Quanto quantization map is unreadable.") from exc
        if not isinstance(quanto_map, dict) or not quanto_map:
            raise RuntimeError("Export validation failed: Quanto quantization map is empty or invalid.")

        weight_entries = {
            name: details.get("weights")
            for name, details in quanto_map.items()
            if isinstance(details, dict)
            and str(details.get("weights")).lower() not in {"none", "null", "false", ""}
        }
        activation_entries = {
            name: details.get("activations")
            for name, details in quanto_map.items()
            if isinstance(details, dict)
            and str(details.get("activations")).lower() not in {"none", "null", "false", ""}
        }
        if expected_quantization is not None:
            expected_type = expected_quantization.get("weight_type")
            if not weight_entries:
                raise RuntimeError("Export validation failed: saved Quanto map has no quantized model weights.")
            if any(str(value).lower() != str(expected_type).lower() for value in weight_entries.values()):
                raise RuntimeError("Export validation failed: saved Quanto weight types differ from the requested mode.")
            if len(weight_entries) != expected_quantization.get("weight_quantized_module_count"):
                raise RuntimeError("Export validation failed: saved Quanto weight-module coverage changed during export.")
            if len(activation_entries) != expected_quantization.get("activation_quantized_module_count"):
                raise RuntimeError("Export validation failed: saved Quanto activation-module coverage changed during export.")

        saved_tensor_count = 0
        for file_path in weight_files:
            with safe_open(str(file_path), framework="pt", device="cpu") as safe_file:
                keys = list(safe_file.keys())
                if not keys:
                    raise RuntimeError(f"Export validation failed: {file_path.name} has no tensors.")
                saved_tensor_count += len(keys)
        if saved_tensor_count == 0:
            raise RuntimeError("Export validation failed: Quanto checkpoint contains no tensors.")

        return {
            "validated": True,
            "config": config_file.name,
            "weight_files": [path.name for path in weight_files],
            "saved_tensor_count": saved_tensor_count,
            "tokenizer_validated": True,
            "quantization_backend": "optimum-quanto",
            "quantization_map": quanto_map_file.name,
            "quantization_map_entries": len(quanto_map),
            "weight_quantized_module_count": len(weight_entries),
            "activation_quantized_module_count": len(activation_entries),
            "full_model_reload": "covered by the load-edit-save-reload integration test",
        }

    state = model.state_dict()
    matched_tensor_count = 0
    saved_tensor_keys: set[str] = set()
    for file_path in weight_files:
        with safe_open(str(file_path), framework="pt", device="cpu") as safe_file:
            keys = list(safe_file.keys())
            if not keys:
                raise RuntimeError(f"Export validation failed: {file_path.name} has no tensors.")
            for key in keys:
                saved_tensor_keys.add(key)
                stored_shape = tuple(safe_file.get_slice(key).get_shape())
                live_tensor = state.get(key)
                if live_tensor is None:
                    raise RuntimeError(
                        f"Export validation failed: saved tensor {key!r} is absent from the in-memory model."
                    )
                if tuple(live_tensor.shape) != stored_shape:
                    raise RuntimeError(
                        f"Export validation failed: shape mismatch for tensor {key!r}."
                    )
                if not _tensor_matches_saved_slice(safe_file.get_slice(key), live_tensor):
                    raise RuntimeError(
                        f"Export validation failed: saved values differ from in-memory tensor {key!r}."
                    )
                matched_tensor_count += 1
    if matched_tensor_count == 0:
        raise RuntimeError("Export validation failed: saved tensors do not match the in-memory model.")

    tied_key_patterns = getattr(model, "_tied_weights_keys", None) or []
    missing_keys = set(state) - saved_tensor_keys
    omitted_non_tied = [
        key
        for key in missing_keys
        if not any(re.search(pattern, key) for pattern in tied_key_patterns)
    ]
    if omitted_non_tied:
        preview = ", ".join(sorted(omitted_non_tied)[:5])
        raise RuntimeError(
            "Export validation failed: checkpoint is missing model tensors "
            f"that are not declared tied aliases ({preview})."
        )

    return {
        "validated": True,
        "config": config_file.name,
        "weight_files": [path.name for path in weight_files],
        "matched_tensor_count": matched_tensor_count,
        "tokenizer_validated": True,
    }


def export_model(adapter: Any, context: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(context["output_dir"]).expanduser()
    _validate_disjoint_paths(str(context["model_path"]), output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Choose a new path; existing outputs are never overwritten."
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if not os.access(output_dir.parent, os.W_OK | os.X_OK):
        raise PermissionError(f"Export parent directory is not writable: {output_dir.parent}")
    try:
        estimated_bytes = sum(
            int(parameter.numel()) * int(parameter.element_size())
            for parameter in adapter.handle.model.parameters()
            if not getattr(parameter, "is_meta", False)
        )
    except (AttributeError, RuntimeError, TypeError):
        estimated_bytes = 0
    free_bytes = shutil.disk_usage(output_dir.parent).free
    required_bytes = int(estimated_bytes * 1.1)
    if required_bytes and free_bytes < required_bytes:
        raise OSError(
            f"Insufficient free disk space for export: estimated at least {required_bytes:,} bytes, "
            f"but only {free_bytes:,} bytes are available in {output_dir.parent}."
        )
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=str(output_dir.parent))
    )
    try:
        _save_model_for_export(
            adapter.handle.model,
            temporary_dir,
            str(context.get("max_shard_size", "5GB")),
        )
        adapter.handle.tokenizer.save_pretrained(temporary_dir)
        validation = _verify_export(
            adapter.handle.model,
            adapter.handle.tokenizer,
            temporary_dir,
            trust_remote_code=bool(context.get("trust_remote_code", True)),
            expected_quantization=context.get("quantization_report"),
        )

        provenance = context.get("provenance_report")
        if provenance is not None:
            (temporary_dir / "autobliteratus_provenance.json").write_text(
                json.dumps(provenance, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if context.get("weight_fingerprint") is not None:
            (temporary_dir / "autobliteratus_weight_fingerprint.json").write_text(
                json.dumps(context["weight_fingerprint"], indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if context.get("model_card") is not None:
            from autobliteratus.tuning._87_model_card_generation_auto_tuning import (
                render_model_card,
            )

            card_report = {
                "source_model": str(context.get("model_path")),
                "architecture": context.get("architecture_report"),
                "ablation": context.get("ablation_report"),
                "quantization": context.get("quantization_report"),
                "reproducibility": context.get("reproducibility_report"),
                "memory": context.get("memory_report"),
                "deployment_target": context.get("deployment_target", "server"),
                "target_frameworks": list(context.get("target_frameworks", ["transformers"])),
                "export": {
                    "format": "transformers",
                    "validated": True,
                    "source_output_disjoint": True,
                },
            }
            (temporary_dir / "README.md").write_text(render_model_card(card_report), encoding="utf-8")

        manifest = {
            "format": "transformers",
            "target_frameworks": list(context.get("target_frameworks", ["transformers"])),
            "max_shard_size": str(context.get("max_shard_size", "5GB")),
            "deployment_target": context.get("deployment_target", "server"),
            "execution": {
                "device": context.get("device", "auto"),
                "dtype": context.get("dtype", "auto"),
                "device_map": context.get("device_map"),
                "max_memory": context.get("max_memory"),
                "resolved_hf_device_map": getattr(adapter.handle.model, "hf_device_map", None),
            },
            "architecture": context.get("architecture_report"),
            "provenance": provenance,
            "reproducibility": context.get("reproducibility_report"),
            "memory": context.get("memory_report"),
            "weight_fingerprint": context.get("weight_fingerprint"),
            "source_model": str(context["model_path"]),
            "source_output_disjoint": True,
            "model_modified": bool(context.get("model_modified", False)),
            "ablation": context.get("ablation_report"),
            "quantization": context.get("quantization_report"),
            "validation": validation,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        checksums: dict[str, str] = {}
        for path in sorted(temporary_dir.rglob("*")):
            if not path.is_file():
                continue
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            checksums[str(path.relative_to(temporary_dir))] = digest.hexdigest()
        try:
            import accelerate
            import torch
            import transformers

            runtime = {
                "python": __import__("platform").python_version(),
                "platform": __import__("platform").platform(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "accelerate": accelerate.__version__,
            }
        except ImportError:
            runtime = {"python": __import__("platform").python_version()}
        manifest["runtime"] = runtime
        manifest["sha256"] = checksums
        (temporary_dir / "autobliteratus_export.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # Temporary and final directories share a parent/filesystem, so this
        # publishes the complete validated result in one rename operation.
        _publish_without_replacing(temporary_dir, output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    result = {
        "path": str(output_dir.resolve()),
        "format": "transformers",
        "source_output_disjoint": True,
        "model_modified": bool(context.get("model_modified", False)),
        "validated": True,
        "validation": validation,
    }
    context["exported_path"] = result["path"]
    context["export_report"] = result
    return result


def run(adapter: Any, context: dict[str, Any]) -> dict[str, Any]:
    return export_model(adapter, context)
