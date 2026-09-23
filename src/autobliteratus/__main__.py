"""Module entry point. Enables ``python -m autobliteratus``."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from autobliteratus.cli import parse_args
    from autobliteratus.pipeline import run_pipeline

    args = parse_args(argv)
    if not args.model_path:
        print("--model-path is required for pipeline execution")
        return 1

    result = run_pipeline(
        model_path=args.model_path,
        work_dir=args.work_dir,
        quantization=args.quantization,
        target_frameworks=args.target_frameworks,
        deployment_target=args.deployment_target,
        enable=args.enable,
        disable=args.disable,
        dry_run=args.dry_run,
    )
    print(f"executed: {len(result['executed'])}")
    print(f"skipped:  {len(result['skipped'])}")
    print(f"failed:   {len(result['failed'])}")
    print(f"model:    {result.get('model_summary', '')}")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
