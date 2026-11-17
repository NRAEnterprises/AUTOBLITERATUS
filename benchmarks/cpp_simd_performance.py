"""Benchmark the isolated C++ scalar and SIMD kernels without file-I/O overhead."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "benchmarks" / "cpp_simd_kernel.cpp"
LENGTHS = (1, 4, 16, 33, 256, 1024, 16384)
LOADS = {"light": 5_000, "medium": 1_000, "heavy": 100}


def _compile(compiler: str, output: Path, force_scalar: bool) -> None:
    command = [compiler, "-std=c++17", "-O3", "-DNDEBUG", str(SOURCE), "-o", str(output)]
    if force_scalar:
        command.insert(-2, "-DCPP_SIMD_FORCE_SCALAR")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"compile failed: {result.stderr}")


def _run(binary: Path, mode: str, length: int, iterations: int) -> dict[str, object]:
    result = subprocess.run(
        [str(binary), "--benchmark", mode, str(length), str(iterations)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    fields = dict(item.split("=", 1) for item in result.stdout.strip().split())
    return {
        "mode": mode,
        "length": length,
        "iterations": iterations,
        "elapsed_ns": int(fields["elapsed_ns"]),
        "checksum": float(fields["checksum"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("cpp-simd-performance.json"))
    args = parser.parse_args()
    compiler = shutil.which("g++")
    if compiler is None:
        raise SystemExit("g++ is required")
    with tempfile.TemporaryDirectory(prefix="autobliteratus-cpp-simd-") as directory:
        directory_path = Path(directory)
        simd_binary = directory_path / "simd"
        scalar_binary = directory_path / "scalar"
        _compile(compiler, simd_binary, force_scalar=False)
        _compile(compiler, scalar_binary, force_scalar=True)
        rows = []
        for load_name, iterations in LOADS.items():
            for length in LENGTHS:
                scalar = _run(scalar_binary, "scalar", length, iterations)
                simd = _run(simd_binary, "simd", length, iterations)
                if abs(float(scalar["checksum"]) - float(simd["checksum"])) > 1e-4:
                    raise RuntimeError(f"checksum mismatch for length={length}, load={load_name}")
                rows.append(
                    {
                        "load": load_name,
                        "scalar": scalar,
                        "simd": simd,
                        "speedup": scalar["elapsed_ns"] / max(simd["elapsed_ns"], 1),
                    }
                )
    payload = {
        "schema_version": 1,
        "host_architecture": platform.machine(),
        "compiler": compiler,
        "kernel": "clamp_scale_float32",
        "rows": rows,
        "qualification": "benchmark_only; not_runtime_enabled",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
