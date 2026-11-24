# C# + Python build handoff

**Status: blocked; design only.** No C# skeleton was created and no compile or test was attempted. The C# pairing remains disabled by default.

## Verified current state

- `command -v dotnet` returned no path; `dotnet --info` could not run (`dotnet unavailable in PATH`). Thus this runner has no discoverable .NET SDK/runtime, and no .NET build, test, or execution evidence is available.
- Repository runner: Linux x86_64 (`uname -a`: kernel `6.18.38+`, x86_64). This identifies the current host only; it is not a configured deployment target.
- Repository scan found no `.csproj`, `.sln`/`.slnx`, `global.json`, Dockerfile/compose file, deployment descriptor, or CI workflow under `.github`, `.gitlab`, `.circleci`, or `.buildkite`. No .NET deployment target is configured or named.
- `configs/polyglot-runtime-pairings.yaml:108–115` defines `csharp_python_enterprise_tooling` as a planned, `planned_environment_blocked` lane. It calls for a typed .NET tool host, a versioned MCP or JSON boundary, Python-owned policy/pipeline stages, default-disabled activation, and no direct model-weight access.

## Why this is conditional, not another router

The existing proposed `go_typescript_routing_protocol` pairing (`configs/polyglot-runtime-pairings.yaml:33–40`) already assigns bounded routing to Go and schema-facing integration to TypeScript. The C# design itself says to avoid duplicating that router without a concrete .NET target (`:111–115`). A generic C# gateway would add a third protocol/routing surface without a demonstrated need. Proceed only for a named .NET-hosted enterprise connector or deployment; keep C# as a narrow adapter/tool host and leave routing, model policy, test selection, and pipeline mutation with their existing owners. Compare that specific requirement against Go + TypeScript before implementation.

## Minimal candidate target (not an existing deployment decision)

After the owner names the actual host/consumer, prototype one disabled-by-default .NET console sidecar for that OS/architecture/runtime. If the consumer is an MCP client, use the official MCP C# SDK over stdio and pin its package version; otherwise use one small, explicitly versioned JSON-RPC contract. Do not implement both transports “just in case.” A minimal contract should have a protocol/contract version, request ID, allowlisted tool name, bounded typed arguments, and structured success/error result. Start with a read-only health or telemetry-summary tool only if it serves the named target. Python remains the authority for model identity, policy, and any pipeline action. No model-weight access, arbitrary shell execution, unrestricted filesystem/network access, raw prompts, or secrets.

Target must be made concrete before coding: deployment host and method, supported OS/architecture, installed/shared versus self-contained runtime, network/auth boundary, and .NET SDK/runtime support policy. Then pin the SDK in `global.json`, target framework, MCP/JSON dependencies, and lock files to versions supported by that target. A .NET 10 target is only a candidate for a newly selected deployment, not a verified project requirement.

## Required tests before any enablement

- **Contract/protocol:** golden request/response fixtures, version negotiation/rejection, schema compatibility, unknown tool rejection, malformed/truncated/oversized/deep JSON, stable machine-readable error codes, and Python reference/result parity.
- **Audit and redaction:** for success, denial, malformed input, timeout, cancellation, and internal failure, assert exactly attributable audit events with run/correlation/request IDs, principal *class* (never credentials), allowlisted tool, parameter-schema hash, decision, status/error category, and duration. Seed API keys, bearer tokens, passwords, connection strings, and prompt-like text in arguments/errors; assert no value leaks into logs or audit data. No raw prompts or full stdout/stderr by default; verify redacted output and safe failure paths.
- **Security:** authentication and per-tool authorization boundary; deny unauthenticated/unauthorized and unknown tools; rate/concurrency and payload limits; replay/idempotency behavior; timeout and cancellation propagation; least-privilege filesystem/network access; process isolation; confirm no model-weight access and no policy/pipeline mutation bypass.
- **Measurement/reproducibility:** bind results to commit, SDK/runtime and dependency versions, protocol/schema hash, host/OS/architecture, and run ID. Record correctness/output hashes, throughput, repeated p50/p95 latency, CPU/RSS, and error/timeout counts against the Python baseline under the same workload. Separate cold/warm cases where applicable; do not infer production performance from a smoke run.

## Commands and prerequisites for an equipped agent

Prerequisites: an approved named .NET deployment target; its supported .NET SDK and matching execution runtime (the SDK normally supplies a runtime for local tests); network/package access or a provisioned package cache; Python contract fixtures/reference; and an authorized authentication/test configuration. Do not install or claim a deployment target based on this handoff alone.

On that target, first capture exact availability and versions:

```sh
command -v dotnet
dotnet --info
dotnet --list-sdks
dotnet --list-runtimes
```

Then create the implementation in a dedicated lane directory (not in shared Python runtime paths), using the target-supported framework and SDK patch selected above. Example commands, with `net10.0` only if the named target approves it:

```sh
mkdir -p src/polyglot/csharp-python-toolhost
cd src/polyglot/csharp-python-toolhost
dotnet new sln -n CSharpPythonToolHost
dotnet new console -n CSharpPythonToolHost -f net10.0
dotnet new xunit -n CSharpPythonToolHost.Tests -f net10.0
dotnet sln add CSharpPythonToolHost/CSharpPythonToolHost.csproj
dotnet sln add CSharpPythonToolHost.Tests/CSharpPythonToolHost.Tests.csproj
# Add the selected, pinned MCP SDK only if MCP is the chosen contract.
dotnet restore
dotnet build --no-restore
dotnet test --no-build --logger 'console;verbosity=normal'
dotnet run --project CSharpPythonToolHost -- --smoke
```

Before treating those results as reproducible, add `global.json` with the exact installed SDK patch, enable package lock files and restore in locked mode, commit the contract/golden fixtures, and run the security, audit/redaction, and measurement suites above. Then test on the actual deployment image/runtime as well as the SDK runner. Record commands, exit codes, versions, and artifacts; keep activation disabled until all pairing promotion gates pass.