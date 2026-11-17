"""Safe loading and normalization for repeatable CLI run plans."""

from __future__ import annotations

from pathlib import Path
from typing import Any

CONFIG_KEYS = {
    "model_path",
    "model_revision",
    "work_dir",
    "output_dir",
    "ablation_strength",
    "ablation_component",
    "quantization",
    "device",
    "dtype",
    "device_map",
    "max_memory",
    "offload_folder",
    "max_shard_size",
    "calibration_data",
    "calibration_samples",
    "calibration_max_length",
    "calibration_momentum",
    "calibration_streamline",
    "target_frameworks",
    "deployment_target",
    "enable",
    "disable",
    "trust_remote_code",
    "dry_run",
    "plan_only",
    "stage_options",
}


def parse_memory_limits(value: Any) -> dict[str | int, str | int] | None:
    """Accept a YAML/TOML mapping or CLI ``DEVICE=SIZE`` entries."""
    if value is None or value == []:
        return None
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, list):
        pairs: list[tuple[str, str]] = []
        for entry in value:
            if not isinstance(entry, str) or "=" not in entry:
                raise ValueError("Each --max-memory value must use DEVICE=SIZE, e.g. 0=20GiB.")
            key, limit = entry.split("=", 1)
            pairs.append((key.strip(), limit.strip()))
        items = pairs
    else:
        raise TypeError("max_memory must be a mapping or a list of DEVICE=SIZE values.")

    result: dict[str | int, str | int] = {}
    for raw_key, raw_limit in items:
        key_text = str(raw_key).strip()
        if not key_text:
            raise ValueError("max_memory device keys cannot be empty.")
        key: str | int = int(key_text) if key_text.isdigit() else key_text
        if isinstance(raw_limit, bool) or not isinstance(raw_limit, (str, int)):
            raise TypeError(f"Invalid memory limit for device {key_text!r}: {raw_limit!r}.")
        if isinstance(raw_limit, str) and not raw_limit.strip():
            raise ValueError(f"Memory limit for device {key_text!r} cannot be empty.")
        if key in result:
            raise ValueError(f"Duplicate max_memory device entry: {key_text!r}.")
        result[key] = raw_limit.strip() if isinstance(raw_limit, str) else raw_limit
    return result


def load_run_config(path: str | Path) -> dict[str, Any]:
    """Load a strict YAML or TOML mapping; unsafe YAML constructors are disabled."""
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Run configuration file does not exist: {config_path}")
    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML run plans require PyYAML; install the project dependencies.") from exc
        payload = yaml.safe_load(text)
    elif suffix == ".toml":
        try:
            import tomllib
        except ImportError:  # Python 3.10
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ImportError as exc:
                raise RuntimeError("TOML run plans on Python 3.10 require the tomli dependency.") from exc
        payload = tomllib.loads(text)
    else:
        raise ValueError("Run config format must be YAML (.yaml/.yml) or TOML (.toml).")

    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise TypeError("Run configuration must contain a top-level mapping.")
    if "autobliteratus" in payload:
        if set(payload) != {"autobliteratus"} or not isinstance(payload["autobliteratus"], dict):
            raise ValueError("The optional 'autobliteratus' section must be the only top-level section and a mapping.")
        payload = payload["autobliteratus"]
    unknown = sorted(set(payload) - CONFIG_KEYS)
    if unknown:
        raise ValueError(f"Unknown run configuration keys: {unknown}")
    result = dict(payload)
    if "stage_options" in result and not isinstance(result["stage_options"], dict):
        raise TypeError("stage_options must be a mapping of stage-specific option names to values.")
    if "max_memory" in result:
        result["max_memory"] = parse_memory_limits(result["max_memory"])
    return result
