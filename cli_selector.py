# abliterate/tuning/101_cli_selector.py
#
# Tuning Type 101 of 103:
# CLI selector.
#
# Runtime toggle: not applicable. This is the parser.
#
# Effect:
#   Parses the command line. Reads --enable and --disable flags. Validates
#   every toggle against the registry. Produces the runtime selection that
#   the orchestrator consumes.
#
# This file owns:
#   - argument parsing
#   - toggle validation
#   - selection assembly

from __future__ import annotations

import argparse

from abliterate.tuning.toggle_registry import TOGGLES, enabled_toggles


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="abliterate")
    p.add_argument("--model-path", required=True)
    p.add_argument("--work-dir", default="./abliterate_workspace")
    p.add_argument("--quantization", default="none",
                   choices=["none", "4bit", "8bit", "fp8", "nvfp4", "mxfp4"])
    p.add_argument("--target-frameworks", nargs="*", default=["transformers"])
    p.add_argument("--deployment-target", default="server",
                   choices=["server", "desktop", "edge", "mobile"])
    p.add_argument("--enable", nargs="*", default=[],
                   help="Toggle ids to force on")
    p.add_argument("--disable", nargs="*", default=[],
                   help="Toggle ids to force off")
    p.add_argument("--dry-run", action="store_true")
    return p


def validate(toggle_ids: list) -> list:
    unknown = [t for t in toggle_ids if t not in TOGGLES]
    if unknown:
        raise ValueError(f"Unknown toggles: {unknown}")
    return toggle_ids


def resolve(args) -> dict:
    enable = validate(args.enable)
    disable = validate(args.disable)
    overrides = {tid: True for tid in enable}
    overrides.update({tid: False for tid in disable})
    return {
        "model_path": args.model_path,
        "work_dir": args.work_dir,
        "quantization": args.quantization,
        "target_frameworks": args.target_frameworks,
        "deployment_target": args.deployment_target,
        "toggles": enabled_toggles(overrides),
        "dry_run": args.dry_run,
    }


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return resolve(args)


TOGGLE = {"id": "tuning.cli_selector", "default": True, "always_on": True}
