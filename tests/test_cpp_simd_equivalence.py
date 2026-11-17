from __future__ import annotations

import math
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from benchmarks.cpp_simd_reference import (
    INPUT_HEADER,
    INPUT_MAGIC,
    apply_clamp_scale,
    read_output,
    write_input,
)

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "benchmarks" / "cpp_simd_kernel.cpp"


@pytest.fixture(scope="module")
def cpp_kernel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is unavailable")
    output = tmp_path_factory.mktemp("cpp-simd") / "cpp_simd_kernel"
    result = subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", str(SOURCE), "-o", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return output


def _run(cpp_kernel: Path, tmp_path: Path, values: list[float], scale: float = 1.25, lower: float = -3.0, upper: float = 3.0):
    input_path = tmp_path / "input.bin"
    output_path = tmp_path / "output.bin"
    write_input(input_path, values, scale, lower, upper)
    result = subprocess.run([str(cpp_kernel), str(input_path), str(output_path)], capture_output=True, text=True, check=False)
    return result, output_path


@pytest.mark.parametrize("count", [1, 3, 4, 7, 15, 16, 17, 31, 32, 33, 127])
def test_simd_matches_scalar_reference_for_widths_and_tails(cpp_kernel: Path, tmp_path: Path, count: int) -> None:
    values = [((index % 9) - 4.0) / 3.0 for index in range(count)]
    result, output_path = _run(cpp_kernel, tmp_path, values)

    assert result.returncode == 0, result.stderr
    actual = read_output(output_path)
    expected = apply_clamp_scale(values, 1.25, -3.0, 3.0)
    assert len(actual) == len(expected)
    assert actual == pytest.approx(expected, rel=0.0, abs=1e-6)


def test_simd_matches_scalar_reference_on_boundary_and_alignment_values(cpp_kernel: Path, tmp_path: Path) -> None:
    values = [-3.0, -2.4, -0.0, 0.0, 2.4, 3.0, 3.0001, -3.0001, 1.1754942e-38, -1.1754942e-38]
    result, output_path = _run(cpp_kernel, tmp_path, values, scale=2.0, lower=-3.0, upper=3.0)

    assert result.returncode == 0, result.stderr
    assert read_output(output_path) == pytest.approx(apply_clamp_scale(values, 2.0, -3.0, 3.0), rel=0.0, abs=1e-6)


def test_simd_rejects_float32_overflow(cpp_kernel: Path, tmp_path: Path) -> None:
    input_path = tmp_path / "overflow.bin"
    output_path = tmp_path / "overflow-out.bin"
    input_path.write_bytes(INPUT_HEADER.pack(INPUT_MAGIC, 1, 2.0, -3.0, 3.0) + struct.pack("<f", 3.4028235e38))
    result = subprocess.run([str(cpp_kernel), str(input_path), str(output_path)], capture_output=True, text=True, check=False)

    assert result.returncode != 0
    assert "cpp_simd_error:float32_overflow" in result.stderr


def test_simd_rejects_malformed_input(cpp_kernel: Path, tmp_path: Path) -> None:
    input_path = tmp_path / "bad.bin"
    output_path = tmp_path / "out.bin"
    input_path.write_bytes(INPUT_HEADER.pack(b"BADMAGIC", 1, 1.0, -1.0, 1.0) + struct.pack("<f", 1.0))

    result = subprocess.run([str(cpp_kernel), str(input_path), str(output_path)], capture_output=True, text=True, check=False)

    assert result.returncode != 0
    assert "cpp_simd_error:invalid_input_header" in result.stderr
    assert not output_path.exists()


def test_reference_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        apply_clamp_scale([math.nan], 1.0, -1.0, 1.0)
