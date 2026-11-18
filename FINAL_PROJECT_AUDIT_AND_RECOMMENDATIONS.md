# Final Project Audit, Build Progression, and Expansion Recommendations

Date: 2026-11-16

## Executive status

The active core is functionally stable: the complete suite passes with **255 passed and 17 skipped**, the package builds, the CLI lists all 100 registry records, and the active graph has 22 callable stages. The release posture remains **experimental engineering build only**, not general-production release. Mutating execution is now fail-closed behind explicit authorization, transactional rollback, and a promoted mapping capability.

No full runtime breakage was found in the tested active paths. The highest-risk issues are coverage and qualification gaps, not a current failing default path.

## Build-target progression

| Target | Current state | Evidence / remaining work |
|---|---|---|
| `make test` | Pass | 156 passed, 17 skipped. |
| `make test-core` | Pass | Core CLI/config/pipeline/registry/orchestrator/audit coverage. |
| `make test-integration` | Pass or environment-skipped | Quantization integration is guarded by optional dependencies and hardware availability. |
| `make test-quantization` | Pass or environment-skipped | Compatibility tests pass; backend-specific qualification remains. |
| `make lint` | Pass | Focused active-code gate is green. |
| `make lint-all` | **Fail: 177 diagnostics** | 162 are auto-fixable; mostly import ordering, deprecated typing aliases, unused imports, broad exception handling, and legacy prototype style debt. |
| `make build` | Pass | Wheel and source distribution build successfully. |
| `make benchmark-help` | Declared target | Must be run in CI as a no-download CLI wiring smoke test; promote to a required gate. |
| `make clean` | Pass by definition | Add verification that build/cache products are actually absent. |

### Full breakage points to address first

1. **Repository-wide lint gate is now clean for the canonical tree.** `lint-all` covers `src`, `tests`, and `benchmarks`; legacy root-level duplicate scripts have been removed rather than linted as a second source tree.
2. **Coverage can give a false sense of safety.** Existing tests did not import memory telemetry or weight fingerprinting, so an initial focused coverage invocation collected no data. Direct contract probes were required. Add tests that import every promoted stage.
3. **Prototype modules contain class methods but no callable stage contract.** They must remain unavailable; importing or instantiating them is not implementation evidence.
4. **Hardware paths are largely unqualified.** CUDA, ROCm, MPS, XPU, HPU, NPU, multi-GPU, and distributed inference paths are not evidenced on this CPU/ARM64 environment.
5. **Exception isolation is intentionally broad in orchestration.** This is useful for independent reporting but needs redacted structured failure handling and severity classification before production.

## Current pipeline tools and how they work

- **CLI / run configuration:** Parses explicit source/output paths, YAML/TOML plans, device and sharding options, profile precedence, and plan-only mode.
- **Static toggle registry:** Parses module AST, discovers `TOGGLE` metadata, and marks a stage callable only when a top-level `run` exists. It avoids importing unrelated optional dependencies.
- **Dispatch:** Imports only callable stage modules and maps toggle IDs to `run(adapter, context)` functions.
- **Topological orchestrator:** Resolves dependency order, skips unresolved dependencies, isolates stage failures, and reports executed/skipped/failed lists.
- **Model loader / adapter:** Loads supported causal Transformer models, tracks device placement, and exposes the live model through the adapter.
- **Refusal direction ablation:** Projects estimated refusal directions out of supported layer weights.
- **Architecture detection:** Inspects live config, decoder blocks, and eligible linear modules.
- **Calibrated quantization:** Uses representative data and Optimum Quanto when available; unsupported backend/module cases fail closed.
- **Baseline/regression gate:** Captures explicit pre-edit causal-LM metrics and compares post-edit NLL/perplexity/forward throughput with declared directions and tolerances.
- **Security scan:** Detects non-finite values and configurable extreme numeric magnitudes. It is not malware, backdoor, or trojan detection.
- **MoE router diagnostics:** Measures expert utilization on representative prompts; no rebalancing is claimed.
- **Multi-objective/Pareto selector:** Validates caller-supplied candidate scores and deterministically selects a Pareto compromise; it does not generate candidates.
- **Memory telemetry:** Captures host RSS and reported accelerator allocations before export; it does not optimize memory.
- **Weight fingerprinting / provenance:** Hashes the final model state and records tensor-count/provenance metadata.
- **Reproducibility metadata:** Records versions, runtime/device settings, and deterministic-algorithm configuration; it does not set seeds or promise bitwise determinism.
- **Audit trail / model card:** Writes redacted atomic JSONL/Markdown artifacts and summary model-card metadata.
- **Validated exporter:** Stages a separate full model export, validates config/tokenizer/tensor coverage, checksums artifacts, and publishes atomically.

## Unactivated switch portfolio

The 78 unavailable switches are backed by a static inventory and visible in the CLI, but not yet by measurable runtime tools. The sandbox discovery and runtime mapping-promotion gates are operational; they do not promote unavailable tuning stages.

For each switch, the required measurable promotion evidence is: a module-level `run(adapter, context)`, stable report schema, deterministic tiny fixture, negative test, pipeline handoff test, artifact/checksum contract where applicable, and documented coverage threshold.

Priority first promotions should be narrow compatibility and consistency checks that reuse existing infrastructure: version compatibility, tokenization consistency, checkpoint/marker semantics after contracts are specified, and custom benchmark ingestion after dataset isolation is defined. Defer broad “auto tuning” claims until the underlying evaluation signal is trustworthy.

## Recommended strengthening expansions

### Reliability and release engineering

1. Add CI matrices for Python 3.10–3.12, x86-64 CPU, ARM64 CPU, and at least one CUDA runner.
2. Make `make lint-all`, `make benchmark-help`, package installation, and wheel smoke import required CI gates.
3. Add `pytest-cov` to the declared dev extras and a `coverage` Makefile target.
4. Add build reproducibility checks, source-distribution install tests, and wheel content manifests.
5. Add dependency lock/constraints files and vulnerability/license scanning.
6. Add type checking for active code and strict API compatibility checks.
7. Add mutation testing for registry, dependency validation, export atomicity, and regression-gate behavior.

### Pipeline controls

8. Add cancellation tokens, stage timeouts, checkpoint/resume, and transaction rollback metadata.
9. Add stage severity (`blocking`, `warning`, `diagnostic`) and fail-policy configuration.
10. Add a run ledger with stage start/end timestamps, input/output artifact IDs, checksums, and retry history.
11. Add structured redaction tests to prove prompts, secrets, and raw exception text never leak into artifacts.
12. Add idempotence tests for every artifact-producing stage.

### Evaluation quality

13. Version explicit capability batteries for reasoning, code, math, multilingual, long-context, tool calling, structured output, and safety.
14. Add calibration metrics, confidence intervals, bootstrap uncertainty, and minimum sample-size checks.
15. Add generation-latency and first-token latency metrics; current forward throughput is not generation latency.
16. Add baseline drift detection across model revisions and datasets.
17. Add a benchmark plugin contract that receives isolated datasets and returns a typed metric bundle.
18. Add a “quality budget” policy for acceptable regression by metric class.

### Architecture coverage

19. Add explicit adapter contracts for Llama, Mistral, Qwen2, GPT-2, OPT, GPT-NeoX, BLOOM, MoE, and hybrid layouts.
20. Add model fixture factories with tiny deterministic checkpoints rather than relying only on live model downloads.
21. Add multimodal and encoder-decoder rejection tests with actionable messages.
22. Add state-dict shape/layout invariants before and after each mutating stage.
23. Add CKA/representation drift reports where capability changes are suspected.

### Memory, fingerprint, and artifact hardening

24. Raise memory telemetry coverage above 85% with mocked CUDA/MPS/allocation-error branches.
25. Bind fingerprints to parameter name, shape, dtype, tensor count, source identity, stage order, and run ID.
27. Use atomic writes for fingerprint files and a signed or checksum-bound manifest. The current mapping capability uses a deterministic schema signature and digest; asymmetric signing remains a release hardening item.
27. Add changed-weight negative tests and missing-fingerprint behavior tests.
28. Add bounded-memory profiling on representative large checkpoints.
29. Add artifact garbage-collection and failed-run cleanup tests.

### Security and governance

30. Separate numeric anomaly scanning from malware/backdoor/trojan claims in user-facing terminology.
31. Add prompt-injection, data-poisoning, jailbreak, and backdoor evaluation only with versioned datasets and safe redaction.
32. Add SBOM generation, dependency provenance, model license checks, and policy-as-code compliance.
33. Add secrets scanning and forbidden-path enforcement in CI.
34. Add reproducible audit exports with schema versioning and migration tests.

### Ecosystem and usability

35. Add a stable plugin API with versioned stage interfaces and explicit trust boundaries.
36. Add REST/gRPC job submission only after cancellation/resume semantics are defined.
37. Add OpenTelemetry traces and Prometheus-compatible metrics for stage timing and resource use.
38. Add a dashboard for run status, regressions, artifacts, and hardware results.
39. Add documented examples for local CPU, ARM64, CUDA, sharded export, plan-only, and failure recovery.
40. Add a release compatibility matrix for Transformers, PyTorch, Accelerate, PEFT, Safetensors, and Quanto.

## Release recommendation

Do not market the project as production-ready until the lint-all gate is resolved or intentionally separated, direct tests cover every promoted stage, at least one accelerator matrix is qualified, and Phase A contracts are applied to the next prototype batch. Continue internal use as an experimental engineering build with the unavailable-switch guardrails enabled.
