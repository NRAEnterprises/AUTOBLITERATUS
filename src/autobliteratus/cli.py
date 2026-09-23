"""CLI parser for AUTOBLITERATUS."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autobliteratus",
        description="Post-ablation capability tuning expansion pack.",
    )
    p.add_argument("--model-path", required=False, help="Path or HF id of the model to ablate and tune.")
    p.add_argument("--work-dir", default="./abliterate_workspace", help="Scratch directory.")
    p.add_argument(
        "--quantization",
        default="none",
        choices=["none", "4bit", "8bit", "fp8", "nvfp4", "mxfp4"],
    )
    p.add_argument("--target-frameworks", nargs="*", default=["transformers"])
    p.add_argument(
        "--deployment-target",
        default="server",
        choices=["server", "desktop", "edge", "mobile"],
    )
    p.add_argument("--enable", nargs="*", default=[], metavar="TOGGLE")
    p.add_argument("--disable", nargs="*", default=[], metavar="TOGGLE")
    p.add_argument("--list-toggles", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def validate_toggles(toggle_ids: list[str]) -> list[str]:
    from autobliteratus.tuning.toggle_registry import TOGGLES
    unknown = [t for t in toggle_ids if t not in TOGGLES]
    if unknown:
        raise SystemExit(f"Unknown toggle ids: {unknown}")
    return toggle_ids


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_toggles:
        from autobliteratus.tuning.toggle_registry import TOGGLES
        for tid, meta in sorted(TOGGLES.items()):
            default = "on" if meta.get("default", True) else "off"
            desc = meta.get("desc", "")
            print(f"{tid:60s} [{default}]  {desc}")
        raise SystemExit(0)

    if args.model_path:
        validate_toggles(args.enable)
        validate_toggles(args.disable)
    return args
