# Phase C Error-Emission Audit — 2026-11-17

## Scope

This audit covers the active diagnostic, quality-gate, resource, export, provenance, and compatibility stages in the Phase C working tree. It compares input-validation errors, candidate-level failures, stage reports, and orchestration failures.

## Findings

| Stage family | Validation behavior | Candidate/runtime behavior | Report shape | Main gap |
|---|---|---|---|---|
| KV-cache profiler | Typed `TypeError`/`ValueError` for bounds and schema | OOM, timeout, cancellation become candidate rows | Full structured report with quality gate | Newly normalized; hardware-specific cache semantics remain unqualified |
| Batch throughput | Typed bounds errors | OOM, timeout, cancellation become candidate rows | Full structured report | Timeout is detected after the synchronous call; it cannot interrupt a blocking kernel |
| Latency | Typed bounds errors; invalid elapsed time raises | No candidate-level OOM/timeout row | Full structured report | Needs the same failure-row contract as throughput/KV |
| Memory telemetry | No user-input validation | Backend allocation errors become `null` readings | Full structured report | Missing readings are warnings, not typed measurement rows |
| Resource budget | Typed budget-schema errors | Missing observations become `missing_metrics`; limits become `violations` | Full structured gate report | Requires explicit upstream reports and is report-only |
| Baseline/regression | Typed metric/tolerance errors | Missing baseline raises; metric mismatch raises | Partial status-only reports | Needs common stage envelope and structured failure categories |
| Compatibility/tokenization | Typed input errors | Missing prerequisites return `not_evaluated` | Partial reports, often without schema/version/artifacts/warnings | Needs common envelope and redacted failure records |
| Fingerprint/provenance | Empty state raises | File/read failures propagate | Partial or raw reports | Needs artifact/hash failure categories and atomic persistence tests |
| Export/quantization | Typed path/config/backend errors | Validation and backend failures raise | Export result is structured only on success | Needs structured failure envelope at orchestration boundary; raw details must be redacted |
| Audit trail | Stage failures are captured by the orchestrator | Error type recorded | Structured audit failure rows | Error message redaction and schema parity need broader tests |

## Common emission contract to apply next

Every promoted stage should return or be translated into a report containing:

```text
stage_id
schema_version
status
model_modified
optimization_performed
metrics or stage-specific measurements
violations
missing_metrics
artifacts
warnings
failures
```

Each `failures[]` row should be machine-readable:

```text
{
  "code": "oom|timeout|cancelled|invalid_input|unsupported_architecture|missing_artifact|backend_unavailable|quality_regression|internal_error",
  "severity": "blocking|warning|diagnostic",
  "candidate_index": integer or null,
  "error_type": exception class name or null,
  "message": redacted stable message,
}
```

Raw prompts, corpus text, secrets, arbitrary callback representations, and full exception text must not enter durable reports. Validation exceptions may remain synchronous for direct API misuse, but pipeline execution should translate expected candidate/stage failures into this envelope.

## Priority order

1. Finish KV-cache candidate rows and quality/resource gates.
2. Bring latency and baseline/regression reports to the same envelope.
3. Add structured failure translation at the orchestrator boundary.
4. Add redaction tests for export, provenance, compatibility, and audit artifacts.
5. Only then consider mutating cache, batch, or deployment selectors.

This audit does not claim that the active modules are production-qualified. It identifies the exact normalization work still required.
