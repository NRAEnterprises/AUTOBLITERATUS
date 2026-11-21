# Resource-Budget Profiler Report Schema

**Status:** Draft normative specification

**Schema version:** `1`

**Stage:** `tuning.resource_budget_gate`

**Purpose:** Define the machine-readable, report-only contract for comparing measured runtime observations against explicit resource budgets before any optimization or mutation is authorized.

## 1. Safety and scope

The resource-budget profiler is diagnostic only. It consumes measurements produced by the memory, latency, and throughput stages; it does not change model parameters, choose an optimization, publish an artifact, or authorize mutation.

A report with `status: within_budget` means only that the available measurements were finite and did not exceed the configured limits for the measured observations. It is not a deployment certification, hardware qualification, quality pass, or mutation capability. Missing, unavailable, timed-out, cancelled, or non-finite measurements must never be interpreted as passing evidence.

The report is evaluated before any future mutating optimizer. A future promotion coordinator may require this report, but the report itself must not grant mutation authority.

## 2. Top-level report schema

A conforming v1 report is a JSON object with these required fields:

| Field | Type | Required | Constraints |
|---|---|---:|---|
| `stage_id` | string | yes | Exactly `tuning.resource_budget_gate` |
| `schema_version` | integer | yes | Exactly `1` for this specification |
| `status` | string | yes | One of `not_configured`, `not_evaluated`, `violations`, `incomplete`, `within_budget` |
| `budget` | object | yes | Validated budget mapping; empty only when status is `not_configured` |
| `observations` | array | yes | Measured observation rows used for comparison; may be empty |
| `candidate_outcomes` | array | yes | Structured non-measured throughput candidates; may be empty |
| `violations` | array | yes | Machine-readable budget violations; empty when no limit was exceeded |
| `missing_metrics` | array | yes | Missing reports or unavailable/non-finite measurements |
| `optimization_performed` | boolean | yes | Must be `false` |
| `model_modified` | boolean | yes | Must be `false` |
| `artifacts` | array | yes | v1 must be an empty array |
| `warnings` | array of strings | yes | Runner and interpretation caveats |

Unknown top-level fields may be retained by consumers for forward compatibility, but v1 producers must not use them to alter the meaning of the required fields.

### 2.1 Status semantics

| Status | Meaning | Safe interpretation |
|---|---|---|
| `not_configured` | No resource budget was supplied | No budget conclusion was made |
| `not_evaluated` | A budget exists, but no measured observation could be evaluated | Must not pass a promotion gate |
| `violations` | At least one finite observation exceeded a configured limit | Candidate set is outside budget |
| `incomplete` | At least one required measurement was unavailable/non-finite, or one or more requested candidates ended with a non-measured outcome | Must not pass a promotion gate |
| `within_budget` | All configured limits were satisfied for all evaluated observations and no measurement was missing | Report-only budget pass for the measured scope |

The status precedence is normative:

1. Empty budget -> `not_configured`
2. No observations -> `not_evaluated`
3. Any violation -> `violations`
4. Otherwise, any missing metric or candidate outcome -> `incomplete`
5. Otherwise -> `within_budget`

## 3. Budget object

The v1 `budget` object accepts only the following keys. Values must be finite, non-negative numbers. Boolean values, NaN, infinity, negative values, and unknown keys are invalid input and must raise a structured validation error before a report is emitted.

| Budget key | Observed metric | Operator | Unit |
|---|---|---|---|
| `max_host_rss_bytes` | `host_rss_bytes` | `<=` | bytes |
| `max_accelerator_allocated_bytes` | `accelerator_allocated_bytes` | `<=` | bytes |
| `max_latency_ms` | `latency_ms` | `<=` | milliseconds |
| `min_tokens_per_second` | `tokens_per_second` | `>=` | tokens per second |
| `max_batch_size` | `batch_size` | `<=` | items |
| `max_sequence_length` | `sequence_length` | `<=` | tokens |

Example:

```json
{
  "max_host_rss_bytes": 4294967296,
  "max_accelerator_allocated_bytes": 3221225472,
  "max_latency_ms": 250.0,
  "min_tokens_per_second": 40.0,
  "max_batch_size": 4,
  "max_sequence_length": 2048
}
```

## 4. Observation schema

Each element of `observations` represents one measured throughput candidate, or one fallback latency measurement when no throughput candidates are available.

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `candidate_index` | integer | yes | Index of the source throughput candidate; `0` for latency fallback |
| `batch_size` | integer or null | yes | Batch size used by the source measurement |
| `sequence_length` | integer or null | yes | Input token length used by the source measurement |
| `tokens_per_second` | finite number or null | yes | Measured generation throughput |
| `latency_ms` | finite number or null | yes | Latency value associated with the observation |
| `host_rss_bytes` | finite integer or null | yes | Host resident-set measurement |
| `accelerator_allocated_bytes` | finite integer or null | yes | Maximum reported allocation across observed accelerator devices |

The v1 gate currently creates observations only from source throughput rows with `status: measured`. If no such rows exist, it may create one fallback observation from a latency report with `status: measured`.

A null observation value is not zero. It means the measurement is unavailable and must produce a corresponding `missing_metrics` entry if that metric is budgeted.

Example:

```json
{
  "candidate_index": 1,
  "batch_size": 4,
  "sequence_length": 512,
  "tokens_per_second": 87.25,
  "latency_ms": 183.4,
  "host_rss_bytes": 2147483648,
  "accelerator_allocated_bytes": 1610612736
}
```

## 5. Candidate-outcomes schema

`candidate_outcomes` preserves every throughput candidate that was present in
the source report but did not have `status: measured`. This prevents OOM,
timeout, cancellation, and other failed candidates from disappearing during
aggregation.

Each outcome has the following fields:

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `candidate_index` | integer | yes | Index of the source candidate |
| `status` | string | yes | Source outcome such as `oom`, `timed_out`, `cancelled`, or `failed` |
| `batch_size` | integer or null | yes | Requested batch size, when supplied |
| `sequence_length` | integer or null | yes | Requested sequence length, when supplied |
| `elapsed_seconds` | finite number | no | Elapsed time when the source reported a finite value |
| `failure_metadata` | object | no | Safe structured failure details from the source runner |

`failure_metadata` uses these runner fields when applicable:

| Field | Type | Meaning |
|---|---|---|
| `kind` | string | `timeout`, `out_of_memory`, or `cancelled` |
| `backend` | string | Resolved device/backend identity, such as `cpu` or `cuda:0` |
| `reason` | string | Non-sensitive cancellation reason, currently `requested` |
| `duration_seconds` | finite number | Measured duration for a timeout |
| `limit_seconds` | finite number | Configured timeout limit |
| `cleanup` | object | OOM cleanup result |

The OOM `cleanup` object contains:

```json
{
  "attempted": true,
  "succeeded": true,
  "action": "cuda.empty_cache"
}
```

Cleanup failures are represented with `succeeded: false`; raw exception text is
not included in the report. The resource-budget gate preserves this metadata
without treating the failed candidate as a measurement.

The gate does not coerce failed outcomes into zero throughput, zero latency,
or synthetic memory values. If at least one candidate outcome is preserved,
the report status is `incomplete` unless an actual measured violation takes
precedence as `violations`. A report with mixed measured and failed candidates
therefore cannot be reported as `within_budget`.

Example:

```json
{
  "candidate_index": 2,
  "status": "oom",
  "batch_size": 8,
  "sequence_length": 2048
}
```

## 6. Violation schema

Each element of `violations` identifies one observed metric that exceeded its configured limit.

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `observation_index` | integer | yes | Index into `observations` |
| `candidate_index` | integer or null | yes | Source candidate index |
| `metric` | string | yes | Observed metric name |
| `budget` | string | yes | Budget key that was exceeded |
| `observed` | finite number | yes | Measured value |
| `limit` | finite number | yes | Configured threshold |
| `operator` | string | yes | `<=` or `>=` |
| `unit` | string | yes | Unit from the budget contract |
| `batch_size` | integer or null | yes | Context for the violating measurement |
| `sequence_length` | integer or null | yes | Context for the violating measurement |

Example:

```json
{
  "observation_index": 1,
  "candidate_index": 1,
  "metric": "latency_ms",
  "budget": "max_latency_ms",
  "observed": 318.2,
  "limit": 250.0,
  "operator": "<=",
  "unit": "milliseconds",
  "batch_size": 4,
  "sequence_length": 512
}
```

A violation is emitted only for a finite observed value that crosses its limit. An unavailable or non-finite value is a missing metric, not a violation.

## 7. Missing-metrics schema

Each element of `missing_metrics` identifies evidence that was required but unavailable.

There are two v1 forms:

### Missing source report

```json
{
  "report": "memory_report",
  "reason": "report_unavailable"
}
```

Allowed report identifiers are:

- `memory_report`
- `throughput_report_or_latency_report`

### Missing value within an observation

```json
{
  "observation_index": 0,
  "metric": "host_rss_bytes",
  "budget": "max_host_rss_bytes",
  "reason": "measurement_unavailable"
}
```

Consumers must treat any non-empty `missing_metrics` array as insufficient evidence for a promotion decision, even when the measured subset is within budget.

## 8. Input report contracts

The profiler consumes the following existing report fields:

- `memory_report.host_rss_bytes`
- `memory_report.accelerator_allocated_bytes`, a mapping of device name to allocation bytes
- `latency_report.status`
- `latency_report.median_latency_ms`
- `latency_report.median_tokens_per_second`
- `latency_report.sequence_length`
- `throughput_report.candidates[]`
- `throughput_report.candidates[].status`
- `throughput_report.candidates[].batch_size`
- `throughput_report.candidates[].sequence_length`
- `throughput_report.candidates[].tokens_per_second`

Only throughput candidates with `status: measured` participate in v1 observations. Source rows with `status: oom`, `timed_out`, or `cancelled` are not successful measurements and must not be converted into zero-valued metrics.

The current implementation reports the absence of all successful candidates as `not_evaluated`. A future schema revision should preserve the individual non-success outcomes explicitly rather than relying only on upstream reports.

## 9. Report-only and mutation invariants

Every conforming report must satisfy:

```text
optimization_performed == false
model_modified == false
artifacts == []
```

The stage may write the report into runtime context under:

```text
resource_budget_report
```

It must not write model parameters, alter the adapter, select a candidate, publish an artifact, or arm the mutation firewall.

## 10. Recommended future extension: schema version 2

The following fields are intentionally not part of v1 because they require changes to the current implementation and fixtures:

- `run_id` and `measurement_id`
- Exact model identity and immutable model revision
- Device, architecture, driver, framework, and kernel metadata
- Workload and prompt-set fingerprint without storing raw prompts
- Warmup count, repetition count, timer source, and synchronization policy
- Per-device allocation rows instead of only the maximum allocation
- OOM recovery result and cleanup status
- Per-candidate timeout and cancellation metadata
- Prefill versus decode latency and throughput
- KV-cache mode and estimated/observed cache bytes
- Quality/regression evidence digest
- Resource-budget configuration digest
- Deterministic report digest

A v2 report must remain backward-distinguishable through `schema_version`; consumers must reject unsupported versions rather than guessing field semantics.

## 11. Example complete v1 report

```json
{
  "stage_id": "tuning.resource_budget_gate",
  "schema_version": 1,
  "status": "violations",
  "budget": {
    "max_host_rss_bytes": 4294967296,
    "max_latency_ms": 250.0,
    "min_tokens_per_second": 40.0,
    "max_batch_size": 4,
    "max_sequence_length": 2048
  },
  "observations": [
    {
      "candidate_index": 0,
      "batch_size": 1,
      "sequence_length": 512,
      "tokens_per_second": 61.4,
      "latency_ms": 164.0,
      "host_rss_bytes": 2147483648,
      "accelerator_allocated_bytes": null
    },
    {
      "candidate_index": 1,
      "batch_size": 4,
      "sequence_length": 512,
      "tokens_per_second": 38.2,
      "latency_ms": 291.0,
      "host_rss_bytes": 3221225472,
      "accelerator_allocated_bytes": null
    }
  ],
  "candidate_outcomes": [],
  "violations": [
    {
      "observation_index": 1,
      "candidate_index": 1,
      "metric": "latency_ms",
      "budget": "max_latency_ms",
      "observed": 291.0,
      "limit": 250.0,
      "operator": "<=",
      "unit": "milliseconds",
      "batch_size": 4,
      "sequence_length": 512
    },
    {
      "observation_index": 1,
      "candidate_index": 1,
      "metric": "tokens_per_second",
      "budget": "min_tokens_per_second",
      "observed": 38.2,
      "limit": 40.0,
      "operator": ">=",
      "unit": "tokens_per_second",
      "batch_size": 4,
      "sequence_length": 512
    }
  ],
  "missing_metrics": [],
  "optimization_performed": false,
  "model_modified": false,
  "artifacts": [],
  "warnings": [
    "Resource budgets evaluate measured runner-specific reports; they do not optimize or certify deployment behavior."
  ]
}
```

## 12. Required validation tests

The current implementation preserves structured failure metadata from the
throughput and KV-cache runners. OOM rows include backend and cleanup status;
timeout rows include measured duration and the configured limit; cancellation
rows include backend and a non-sensitive requested reason. OOM, timeout, and
cancellation-only candidate sets remain `not_evaluated`, while mixed sweeps
remain `incomplete`. A non-finite value in an otherwise measured observation
also produces `incomplete`. The full suite currently reports **265 passed and
17 skipped** after this increment.

The v1 test suite should verify:

1. Every supported budget key and comparison operator.
2. Unknown budget keys, booleans, negative values, NaN, and infinity are rejected.
3. `within_budget` is emitted only when all required measurements are finite and within limits.
4. Every violation includes metric, observed value, limit, operator, unit, batch size, and sequence length.
5. Missing reports produce `not_evaluated`.
6. Missing values in measured observations produce `incomplete`.
7. `oom`, `timed_out`, and `cancelled` source rows are never converted into successful measurements.
8. Empty budgets produce `not_configured`.
9. The stage always remains report-only and sets `model_modified` and `optimization_performed` to `false`.
10. The report is stored in `context["resource_budget_report"]` by object identity.
11. Repeated identical input reports produce identical normalized report content.
12. The profiler cannot arm or satisfy the mutation firewall by itself.
