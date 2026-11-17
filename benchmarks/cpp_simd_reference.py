"""Scalar reference and binary contract for the C++/SIMD prototype."""

from __future__ import annotations

import math
import struct
from pathlib import Path

INPUT_MAGIC = b"ABSMD01\0"
OUTPUT_MAGIC = b"ABSOUT1\0"
INPUT_HEADER = struct.Struct("<8sIfff")
OUTPUT_HEADER = struct.Struct("<8sI")
MAX_ELEMENTS = 1_000_000


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def apply_clamp_scale(values: list[float], scale: float, lower: float, upper: float) -> list[float]:
    """Reference operation: finite float32 y = clamp(float32(x*scale), lower, upper)."""
    if not 0 < len(values) <= MAX_ELEMENTS:
        raise ValueError("values must contain between 1 and 1000000 elements")
    scale = _float32(scale)
    lower = _float32(lower)
    upper = _float32(upper)
    if not all(math.isfinite(value) for value in (scale, lower, upper)) or lower > upper:
        raise ValueError("scale and bounds must be finite with lower <= upper")
    result = []
    for value in values:
        value = _float32(value)
        if not math.isfinite(value):
            raise ValueError("values must be finite")
        product = _float32(value * scale)
        if not math.isfinite(product):
            raise OverflowError("float32 multiplication overflow")
        result.append(_float32(min(upper, max(lower, product))))
    return result


def write_input(path: str | Path, values: list[float], scale: float, lower: float, upper: float) -> None:
    result = apply_clamp_scale(values, scale, lower, upper)
    del result
    path = Path(path)
    with path.open("wb") as stream:
        stream.write(INPUT_HEADER.pack(INPUT_MAGIC, len(values), _float32(scale), _float32(lower), _float32(upper)))
        stream.write(struct.pack(f"<{len(values)}f", *(_float32(value) for value in values)))


def read_output(path: str | Path) -> list[float]:
    path = Path(path)
    data = path.read_bytes()
    if len(data) < OUTPUT_HEADER.size:
        raise ValueError("output is truncated")
    magic, count = OUTPUT_HEADER.unpack_from(data)
    if magic != OUTPUT_MAGIC or not 0 < count <= MAX_ELEMENTS:
        raise ValueError("output header is invalid")
    expected = OUTPUT_HEADER.size + count * 4
    if len(data) != expected:
        raise ValueError("output length does not match its header")
    return list(struct.unpack_from(f"<{count}f", data, OUTPUT_HEADER.size))
