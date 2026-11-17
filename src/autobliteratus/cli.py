"""CLI parser for AUTOBLITERATUS."""

from __future__ import annotations

import argparse
import math
import sys

from autobliteratus.run_config import load_run_config, parse_memory_limits

DEVICE_MAPS = {None, "auto", "balanced", "balanced_low_0", "sequential"}
QUANTIZATION_MODES = {
    "none",
    "quanto-int4",
    "quanto-int8",
    "quanto-int4-weight-only",
    "quanto-int8-weight-only",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autobliteratus",
        description="Load, edit, validate, and export a separate model copy.",
    )
    p.add_argument("--model-path", required=False, help="Path or Hugging Face ID of the model to tune.")
    p.add_argument(
        "--model-revision",
        default=None,
        help="Immutable Hugging Face commit SHA or revision for remote model loads; required for remote IDs.",
    )
    p.add_argument("--work-dir", default="./abliterate_workspace", help="Scratch directory.")
    p.add_argument("--config", default=None, help="YAML or TOML run plan; explicit CLI values override the plan.")
    p.add_argument(
        "--output-dir",
        "--output-path",
        dest="output_dir",
        default=None,
        help="New directory for the modified full model (default: <work-dir>/modified_model).",
    )
    p.add_argument("--ablation-strength", type=float, default=1.0, help="Refusal-direction projection strength in (0, 1].")
    p.add_argument(
        "--ablation-component",
        choices=["attn.out", "mlp.down"],
        default="attn.out",
        help="Layer weights to project (default: attention output projections).",
    )
    p.add_argument("--quantization", default="none", choices=sorted(QUANTIZATION_MODES), help="Quanto PTQ mode.")
    p.add_argument(
        "--device",
        default="auto",
        help="Model execution device: auto, cpu, cuda:N, mps, xpu:N, hpu, or npu (backend must exist in PyTorch).",
    )
    p.add_argument(
        "--device-map",
        choices=["auto", "balanced", "balanced_low_0", "sequential"],
        default=None,
        help="Accelerate placement/sharding strategy; requires Accelerate and a supported Transformers model.",
    )
    p.add_argument(
        "--max-memory",
        action="append",
        default=None,
        metavar="DEVICE=SIZE",
        help="Per-device budget for Accelerate (repeat, e.g. 0=20GiB --max-memory cpu=32GiB).",
    )
    p.add_argument(
        "--offload-folder",
        default=None,
        help="Scratch directory for Accelerate CPU/disk offload; kept separate from the model output.",
    )
    p.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Maximum safetensors shard size during export (default: 5GB).",
    )
    p.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    p.add_argument("--calibration-data", default=None, help="Representative text file (.txt, .jsonl, or .json).")
    p.add_argument("--calibration-samples", type=int, default=128)
    p.add_argument("--calibration-max-length", type=int, default=256)
    p.add_argument("--calibration-momentum", type=float, default=0.9)
    p.add_argument(
        "--calibration-no-streamline",
        dest="calibration_streamline",
        action="store_false",
        default=True,
        help="Disable Quanto's calibration streamline optimization for diagnostic comparison.",
    )
    p.add_argument(
        "--calibration-streamline",
        dest="calibration_streamline",
        action="store_true",
        help="Enable Quanto's calibration streamline optimization.",
    )
    p.add_argument("--target-frameworks", nargs="*", default=["transformers"])
    p.add_argument("--deployment-target", default="server", choices=["server", "desktop", "edge", "mobile"])
    p.add_argument("--enable", nargs="*", default=[], metavar="TOGGLE")
    p.add_argument("--disable", nargs="*", default=[], metavar="TOGGLE")
    p.add_argument("--list-toggles", action="store_true")
    dry_run_group = p.add_mutually_exclusive_group()
    dry_run_group.add_argument("--dry-run", dest="dry_run", action="store_true")
    dry_run_group.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    p.set_defaults(dry_run=False)
    plan_group = p.add_mutually_exclusive_group()
    plan_group.add_argument("--plan-only", dest="plan_only", action="store_true", help="Validate and print the operation graph without loading model weights.")
    plan_group.add_argument("--execute-plan", dest="plan_only", action="store_false", help="Override a config plan and execute normally.")
    p.set_defaults(plan_only=False)
    trust_group = p.add_mutually_exclusive_group()
    trust_group.add_argument(
        "--trust-remote-code",
        dest="trust_remote_code",
        action="store_true",
        help="Allow custom Python code from the model repository to execute.",
    )
    trust_group.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    p.set_defaults(trust_remote_code=False)
    return p


def validate_toggles(toggle_ids: list[str]) -> list[str]:
    from autobliteratus.tuning.toggle_registry import TOGGLES

    unknown = [toggle for toggle in toggle_ids if toggle not in TOGGLES]
    if unknown:
        raise SystemExit(f"Unknown toggle ids: {unknown}")
    unavailable = [toggle for toggle in toggle_ids if not TOGGLES[toggle].get("available", False)]
    if unavailable:
        raise SystemExit(
            "Requested tuning stages are prototypes and are not operational in this release: "
            f"{unavailable}. Use --list-toggles to inspect implemented stages."
        )
    return toggle_ids


def _apply_config(args: argparse.Namespace, parser: argparse.ArgumentParser, arguments: list[str]) -> None:
    if not args.config:
        return
    values = load_run_config(args.config)
    explicitly_supplied = set()
    for token in arguments:
        option = token.split("=", 1)[0]
        action = parser._option_string_actions.get(option)
        if action is not None:
            explicitly_supplied.add(action.dest)
    for key, value in values.items():
        if key not in explicitly_supplied:
            setattr(args, key, value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)

    if args.list_toggles:
        from autobliteratus.tuning.toggle_registry import TOGGLES

        for toggle_id, meta in sorted(TOGGLES.items()):
            default = "on" if meta.get("default", True) else "off"
            unavailable = " [prototype: unavailable]" if not meta.get("available", True) else ""
            desc = meta.get("desc", "")
            print(f"{toggle_id:60s} [{default}]{unavailable}  {desc}")
        raise SystemExit(0)

    try:
        _apply_config(args, parser, arguments)
        if args.max_memory is not None:
            args.max_memory = parse_memory_limits(args.max_memory)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    for name in ("enable", "disable", "target_frameworks"):
        value = getattr(args, name)
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            parser.error(f"{name} must be a string list.")
        setattr(args, name, value)

    if args.device_map not in DEVICE_MAPS:
        parser.error(f"--device-map must be one of {sorted(v for v in DEVICE_MAPS if v is not None)}.")
    if args.quantization not in QUANTIZATION_MODES:
        parser.error(f"Unsupported quantization mode: {args.quantization!r}.")
    if not math.isfinite(args.ablation_strength) or not 0.0 < args.ablation_strength <= 1.0:
        parser.error("--ablation-strength must be in the interval (0, 1].")
    if args.plan_only and args.dry_run:
        parser.error("--plan-only and --dry-run are distinct modes and cannot be combined.")

    validate_toggles(args.enable)
    validate_toggles(args.disable)
    overlap = sorted(set(args.enable) & set(args.disable))
    if overlap:
        parser.error(f"A toggle cannot be both enabled and disabled: {overlap}")
    return args
