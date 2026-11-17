# C++/SIMD Backend Prototype Contract — 2026-11-17

## Scope

This prototype is a pure, report-only signal kernel. It computes finite float32 values using:

```text
y = clamp(float32(x * scale), lower, upper)
```

The C++ executable has no model access, no filesystem access beyond the two explicit input/output paths, no network access, and no permission to mutate runtime artifacts. It is not registered as a runtime tuning stage.

## Buffer contract

Input is a little-endian binary buffer with header:

```text
magic[8] = ABSMD01\\0
count: uint32
scale: float32
lower: float32
upper: float32
values: count * float32
```

Output is:

```text
magic[8] = ABSOUT1\\0
count: uint32
values: count * float32
```

The implementation rejects invalid magic, zero or excessive element counts, truncated or trailing data, nonfinite parameters, invalid bounds, nonfinite inputs, and float32 overflow.

## SIMD and reference behavior

The implementation uses ARM NEON on AArch64 and SSE2 where available, with a scalar tail. The Python reference quantizes every operation to float32 and applies the same clamp order. Equivalence is checked across short, SIMD-width, non-width, and large vector lengths; negative, zero, positive, boundary, and alignment-offset values; and overflow-sensitive inputs.

Passing these tests establishes only equivalence for this kernel and compiler/architecture build. It does not qualify C++/SIMD for model mutation, quantization, CUDA, x86-64, or another compiler/ISA. The prototype remains disabled until sanitizer, architecture-matrix, and resource tests are completed.
