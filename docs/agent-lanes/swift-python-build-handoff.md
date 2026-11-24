# Swift + Python Apple-local runtime: build handoff

**Status: blocked in the inspected runner; no Swift build or execution was attempted.** This is a handoff, not implementation or Apple-target evidence.

## Verified blocker and project context

The current runner reports `Linux 178955f2f03c 6.18.38+ #1 SMP PREEMPT_DYNAMIC Thu Oct 1 05:04:21 UTC 2026 x86_64 x86_64 x86_64 GNU/Linux` (`uname -s`: `Linux`; `uname -m`: `x86_64`). `swift`, `swiftc`, `xcrun`, and `xcodebuild` are all absent from `PATH`; each version/tool probe returned `command not found`. No `/Applications/Xcode*.app`, `/Library/Developer/CommandLineTools`, or Apple developer directory was found. The inspected repository config has no runner labels or macOS/Swift setup; no Apple-Silicon/macOS target is attached or evidenced here. A Linux Swift install or cross-compilation would not supply the Apple SDK, target runtime, or Apple hardware evidence this lane requires.

`configs/polyglot-runtime-pairings.yaml` (schema v1, `swift_python_apple_local_runtime`) says Python owns policy/pipeline; Swift is an opt-in Apple sidecar using MLX Swift or Foundation Models; interchange is `versioned_json_sidecar`; activation is disabled by default. It lists gates for a locked Apple toolchain, CPU-reference equivalence, tool permissions, privacy/dataflow, model/device metadata, and latency/memory/quality. `docs/pipeline-observability-database-plan.md` §17–21 requires a named Apple-Silicon lane, distinct provider identities, per-run SoC/OS/tool/model/precision/workload/latency/throughput/quality/permissions metadata, and no raw prompts/user data by default. Its implementation sequence (§81) explicitly defers Swift/C# experiments until target hardware or deployment requirements are named. Therefore do not mark this lane implemented or enabled from the current runner.

## Minimal first target

On a provisioned **macOS Apple-Silicon (`arm64`)** runner, build a new, isolated, Foundation-only Swift command-line sidecar with a deterministic bounded `text_stats` operation (UTF-8 byte count and Unicode-scalar count) plus a `health` operation. Python remains the caller, validator, and reference implementation. This is only a protocol/build/contract smoke target—not inference, MLX/Foundation Models qualification, or a performance claim. Keep it disabled by default and outside shared runtime stages. After its contract tests pass and the actual model/device target is named, evaluate MLX Swift and Foundation Models as separately identified provider backends with locked package/toolchain versions.

## Proposed v1 sidecar contract

Use newline-delimited JSON over stdin/stdout: exactly one request object per line and exactly one response object per request line. Stdout is protocol-only; diagnostics go to stderr. Python launches the process with bounded input/output and timeout, validates every response, and owns policy and any persisted evidence. No shell interpolation, network access, arbitrary filesystem access, prompt logging, or model-weight writes.

Request envelope:

```json
{"protocol_version":1,"request_id":"stable-caller-id","op":"health","payload":{}}
```

For `text_stats`, `payload` is `{"text":"..."}`. Successful response:

```json
{"protocol_version":1,"request_id":"stable-caller-id","status":"ok","result":{"backend":"swift-foundation-smoke","utf8_bytes":3,"unicode_scalars":3}}
```

Failure response:

```json
{"protocol_version":1,"request_id":"stable-caller-id","status":"error","error":{"code":"unsupported_version","message":"Unsupported protocol version"}}
```

Keep v1 strict: reject malformed JSON, unknown operations, unsupported versions, oversized lines/text, and invalid payload types with stable machine-readable error codes; do not silently coerce or truncate. Preserve request IDs. `health` should report executable/protocol version and runtime OS/architecture; report device/model/toolchain/provider fields only when actually available, otherwise `null`/unavailable (never fabricate). The smoke provider must not claim an inference model. Add a Python reference for the exact `text_stats` semantics and test parity, including empty text, non-ASCII and supplementary Unicode scalars, and byte/count boundaries.

## Test and measurement checklist

- Lock and record macOS version/build, Xcode/Command Line Tools, Swift version, package resolution, machine model/SoC, and architecture; retain `Package.resolved` and environment manifest.
- Compile and run Swift package tests on the actual Apple target; separately run Python contract tests against the built executable. Do not count Linux cross-compilation or static inspection as Apple execution.
- Contract negatives: malformed/oversized JSON, unsupported version/op, wrong types, missing fields, duplicate/empty IDs, Unicode edge cases, deterministic output, stdout purity, stderr diagnostics, exit/crash, timeout, cancellation, restart, and bounded output.
- Compare Swift results to the Python CPU reference exactly. Verify default-off activation and that an unconfigured Python run never launches the sidecar.
- Privacy/dataflow: confirm no prompt/input is persisted or emitted to logs; audit subprocess environment and filesystem/network/tool permissions. Exercise denied/unauthorized tool paths before adding any tools.
- For a later inference backend, bind provider/model/revision, precision, toolchain, OS, SoC, workload hash, seed, and permission set to each result. Run paired CPU-reference quality tests and repeated, warmed/cold latency, throughput, peak-memory, and quality measurements; report unavailable/failed/skipped distinctly. Do not generalize Apple-only observations to ARM/Linux, CUDA, or other accelerators.
- Keep each backend qualified separately. Do not enable globally; require the pairing’s full promotion gates and a named pipeline opt-in.

## Exact next commands for a provisioned agent

First verify that the attached runner is genuinely suitable; stop if any target/tool probe fails:

```sh
cd /home/ubuntu/AUTOBLITERATUS
uname -a
uname -s
uname -m
sw_vers
xcode-select -p
swift --version
swiftc --version
xcrun --find swift
xcrun --sdk macosx --show-sdk-version
system_profiler SPHardwareDataType
```

Then create a new experiment directory (do not reuse or edit shared source/config files), initialize the package, and run the real Swift build/tests:

```sh
cd /home/ubuntu/AUTOBLITERATUS
mkdir -p experiments/swift-python-apple-local-runtime-smoke-2026-11-22
cd experiments/swift-python-apple-local-runtime-smoke-2026-11-22
swift package init --type executable
swift build -v
swift test -v
```

Replace the generated package contents with the v1 sidecar and contract tests above before treating those package commands as meaningful evidence; the untouched template build alone is not a sidecar test. Then build and exercise the executable explicitly, preserving protocol stdout separately from diagnostics:

```sh
swift build -c release
.build/release/swift-python-apple-local-runtime-smoke <<'JSONL'
{"protocol_version":1,"request_id":"health-1","op":"health","payload":{}}
{"protocol_version":1,"request_id":"stats-1","op":"text_stats","payload":{"text":"Aé😀"}}
JSONL
```

Run the repository Python suite and the new Python↔sidecar contract test with the project’s supported environment/dependencies, for example:

```sh
cd /home/ubuntu/AUTOBLITERATUS
python -m pytest -q
python -m pytest -q tests/test_swift_python_apple_sidecar.py
```

The second test path must be added in the experiment/integration work before invocation. Record exact commands, exit codes, captured outputs/hashes, and host/toolchain manifest. Do not claim compile, execution, Apple-target support, inference, or performance until the corresponding command actually ran on the provisioned Mac and its artifacts were inspected.
