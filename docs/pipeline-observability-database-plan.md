# AUTOBLITERATUS pipeline observability and database plan

**Status:** Prototype foundation added and a small-ledger CSV/Python vs SQLite vs DuckDB parity benchmark completed; pipeline-wide collection and comparative performance claims are not qualified.

## Architecture decision

Use **Python + SQL with embedded DuckDB as the first local analytical store**, and preserve the portable source evidence as versioned JSONL/Parquet exports. The project already produces test summaries and structured resource reports, while the current CSV ledger is small and aggregate-only. The first schema and importer are intentionally optional: core installations gain no mandatory database dependency, and no existing runtime stage is changed.

DuckDB is a fit for a local analytics workload because it can query Parquet directly, push filters into scans, and read only required columns. Its embedded read-write mode is centered on one writer process, with concurrent writer threads inside that process; multiple processes may read in read-only mode. Therefore test workers must spool events independently and a single parent/collector should batch-ingest them. Do not have parallel pytest workers open the same DuckDB file for writes. [1] [2]

This is a workload-specific choice, not a claim that DuckDB is faster or better than every database. SQLite with WAL and FTS5 is a strong lightweight local baseline for transactional records and lexical search; WAL still has one writer at a time. PostgreSQL adds server concurrency and built-in ranked full-text search, while pgvector can be evaluated if semantic retrieval is justified. ClickHouse/ClickStack is a relevant later baseline for high-cardinality observability at scale and agent-accessible log/trace/metric queries. The benchmark must choose engines based on the tested workload and include operational cost, not only query time. [3] [4] [5] [6]

The current implementation is the v1 SQL schema packaged at `src/autobliteratus/observability/schema-v1.sql`, an opt-in Python database module in `src/autobliteratus/observability/telemetry_store.py`, and a CSV import/summary script at `scripts/telemetry_db.py`. The importer can initialize the store, hash and deduplicate legacy CSV rows, preserve aggregate test counts, and redact common credential patterns in free-text notes. An explicitly enabled pytest plugin now records per-test outcome, phase durations, suite duration, and hashed test/failure identifiers. It does not store test names, raw failure text, or raw command arguments. This remains a **foundation only**: retries/attempts, collection/interruption fidelity, application/stage logs, profiler and benchmark adapters, and authoritative LLM-provider usage are not yet integrated.

## Planned paired-language lanes

### Swift + Python: Apple-local execution

Use Swift only for Apple-specific local inference/device services and native, permissioned integrations. Python remains the policy, evaluation, and pipeline owner. Start with one versioned JSON sidecar for a named Apple-Silicon lane. Evaluate MLX Swift for custom model execution and Apple Foundation Models tool calling for native on-device tasks; keep the two providers as separately identified backends. Apple documents MLX as an Apple-platform framework with Swift bindings, while Foundation Models tools let a model call app-defined code. Neither is evidence of general ARM, CUDA, or cross-vendor coverage. [7] [8]

Record SoC/device identity, OS and Swift/MLX/model versions, precision, memory pressure, prompt/workload hash, latency, throughput, quality result, and tool permissions for each run. Promotion requires a Python reference comparison, privacy/dataflow review, denial-path tests for unauthorized tools, and an Apple hardware matrix. No raw prompts or user data should be persisted by default.

### C# + Python: .NET/MCP integration

Use C# only where an actual .NET deployment or enterprise connector is needed. C# can host a narrowly scoped MCP or versioned JSON interface; Python continues to own model policy, test selection, and pipeline mutation boundaries. Microsoft documents Semantic Kernel agent/function-calling support for C# and Python, and the MCP project publishes an official C# SDK. This lane overlaps with the already planned Go/TypeScript routing concept, so it should not be built as a second generic router. Pick one named .NET target and compare it with the existing routing direction before creating a service. [9] [10]

Log every tool invocation with run/correlation IDs, authenticated principal class (not credentials), allowlisted tool name, parameter-schema hash, decision, response status, duration, and redacted error type. Never log secret argument values or grant the tool host model-weight access. Require authentication, malformed-input, rate-limit, timeout/cancellation, replay, and audit-coverage tests.

### Python + SQL/DuckDB: pipeline evidence and query layer

Use the database to make runs comparable across commits, devices, stages, toolchains, models, and policies. The schema supports run summaries, individual test results, structured pipeline events, normalized measurements, artifact references, LLM token usage, and query benchmarks. Preserve unavailable/failed/skipped outcomes distinctly; missing data must never become zero or a pass.

Keep binary logs and large model artifacts outside the database. Store content-addressed references, SHA-256 digests, sizes, and manifests. Persist structured, redacted event fields in the warehouse. Raw prompts, API keys, tokens, environment dumps, and full stdout/stderr are not stored by default. Keep exact sensitive payloads out of the database; where debugging truly requires a log attachment, write a separately access-controlled, redacted file and store only its reference and hash.

## Ingestion and query workflow

1. **Create a run envelope.** Every test or pipeline run gets a stable run ID and records UTC date/time, commit, branch, command/config digest, OS, CPU/ISA, Python/runtime versions, device/driver, seed, model ID/revision, and runner identity. Never infer missing metadata; retain it as unavailable.
2. **Capture pytest results.** The opt-in plugin records one row per observed test node with pass/fail/skip/xfail, accumulated phase duration, and hashed identifiers. Extend it to handle retry attempts, collection errors, interrupted sessions, and any safely redacted log references. Keep raw stdout/stderr outside the database by default.
3. **Emit stage events.** Standardize stage start, completion, skip, timeout, cancellation, error, and artifact-publication events. Include stage/toggle ID, correlation ID, severity, measured duration, machine-readable error code, and safe attributes. Reuse resource-budget reports rather than inventing duplicate metric names.
4. **Record numeric measurements.** Store named values with explicit units and dimensions: quality scores, memory, throughput, latency, calibration settings, skipped/unavailable flags, and artifact hashes. Record repetitions and synchronization/cache conditions needed to reproduce a measurement.
5. **Record LLM tokenomics.** For each authorized model call, store provider/model, purpose, prompt-template hash, input/output/cached-input tokens, measured or clearly labeled estimated cost, latency, and result/quality reference. Do not store prompt text. Use provider usage responses when available; mark estimated values and pricing snapshot/version separately.
6. **Make queries controlled and useful.** Start with versioned SQL templates for common questions: regressions by commit/device/model, dependency-related skips, repeat failures, quality-versus-memory/latency tradeoffs, and cost per accepted result. Use bounded rows and safe summaries. Add a model-generated SQL feature only later, behind read-only credentials, an allowlisted schema, query timeouts, row limits, and SQL validation.
7. **Export portable evidence.** Support deterministic Parquet partitions by run date/schema version and retain source hashes. DuckDB may query those files directly. Backups and restore tests must prove that the database can be rebuilt from portable evidence.

## Query usefulness and token-cost design

“Deeper” queries should join evidence that is currently separated across result ledgers, profiler reports, quality outcomes, and runtime fingerprints. Examples include: which exact toolchain/hardware/model combinations have recurring failures; whether a latency gain coincides with a quality regression; which skipped tests are caused by a missing dependency; and which pipeline produces the best quality-per-token or quality-per-memory result under an explicit budget.

For agent-facing investigation, use a two-step flow: translate a question into a parameterized, read-only query; then return a capped, ranked evidence set with provenance. Keep routine questions on deterministic SQL templates so they need no LLM call. When a model is used, count the translation and summarization tokens and compare against a clearly defined baseline that sends the same question plus unfiltered logs/results as context. Report tokens per correct answer and answer exactness together; lower token count alone is not a success. Use lexical retrieval first. Add vector embeddings only if a measured workload demonstrates a retrieval gap that filters/FTS cannot meet.

## Benchmark plan and success criteria

Compare the same real, sanitized project evidence and the same query set across:

- the existing CSV/Python baseline for simple summary questions;
- DuckDB + Parquet as the candidate local analytical path;
- SQLite WAL + FTS5 as a small local transactional and lexical-search baseline;
- PostgreSQL full-text search, optionally with pgvector, when shared multi-writer or semantic retrieval is under test;
- ClickHouse/ClickStack only when event volume and shared observability operations justify a server-scale comparison.

Do not fabricate rows to make the project appear large. First benchmark the real history and report when its size is too small to support scaling conclusions. If load testing is needed, replay captured real events deterministically and label that as a replay workload, never as new evidence. For each engine, verify result parity against golden expected outputs before comparing speed. Run cold- and warm-cache cases, repeat each query, and record engine/dependency versions, machine fingerprint, data size, index/build time, ingestion rate, database size, CPU/RSS, p50/p95 latency, concurrency, and exact output hashes.

Initial candidate promotion gates (targets, to be preregistered before comparing engines):

- zero lost or silently duplicated source rows; repeated imports are idempotent;
- 100% exact parity on deterministic count, grouping, and regression queries;
- zero secret leakage in seeded redaction tests and no raw prompts by default;
- at least 20% lower p95 latency than the best simple local baseline on at least three representative multi-dimensional queries, with the same hardware/cache condition and no correctness loss;
- for LLM-assisted questions, at least 30% fewer input-plus-output tokens per correct answer than the unfiltered-context baseline, with answer exactness no worse than baseline;
- report result size and database storage cost alongside performance; do not claim success if the test dataset is too small to be representative.

These are proposed targets, not observed results. Adjust them only before benchmark execution, record why, and preserve failed runs. No honest test can prove superiority over every known database. The goal is measurable improvement on AUTOBLITERATUS workloads, with clear limits and reproducible comparisons.

### Observed smoke comparison

The final 2026-11-22 rerun of `scripts/benchmark_telemetry_engines.py` compared three queries across CSV/Python, SQLite 3.45.1, and DuckDB 2.0.1 using only the five actual rows in `testing/results.csv` (3,167 bytes; no synthetic performance rows). It used 1,000 timed executions and 25 warmups per query/engine. Result hashes matched exactly, and parity was checked during every measured call. Repeated query median ranges were 0.0298–0.0301 ms for CSV/Python, 0.0085–0.0112 ms for SQLite, and 0.7039–0.8741 ms for DuckDB. Database setup/import measured 4.174 ms for SQLite and 79.130 ms for DuckDB; CSV/Python reparses the file for each query. SQLite used its default `DELETE` journal and did not exercise FTS5. First measured latency follows an untimed parity query, so it is not cold-start timing. These observations do not meet the proposed representative-workload gates or qualify DuckDB for promotion. DuckDB uses the project schema while SQLite uses only the two normalized tables required for the queries; file size and setup are not same-schema comparisons. No LLM calls or token measurements were involved.

The human-readable [benchmark report](telemetry-engine-benchmark-2026-11-22.md) documents methods and caveats; the [machine-readable measurements and output hashes](../benchmarks/results/telemetry-ledger-engine-benchmark-2026-11-22.json) retain environment details and per-query distributions. Treat the results as a parity-checked baseline, not an engine ranking.

### Observed system assessment

The [system assessment report](telemetry-system-assessment-2026-11-22.md) and [machine-readable checks](../benchmarks/results/telemetry-system-assessment-2026-11-22.json) add repeated import, schema, closed-file copy/restore, redaction-canary, and read-only concurrency checks. The five source rows imported once and then 20 times with zero duplicate inserts. Three rounds of 250 reads per reader at 1/2/4/8 readers all returned exact results without errors; median aggregate throughput was approximately 1,318/2,439/1,650/1,924 QPS respectively. These figures reuse the same five-row database and do not measure representative-volume scaling. Writer concurrency was not attempted under the intended single-writer design.

The baseline seeded redaction probes found misses for an authorization bearer header, `token=...`, and a `ghp_...`-like string. A separate isolated importer fixture confirmed that an arbitrary command field was stored unchanged; no real telemetry rows were added for that probe. A local draft patch now redacts those forms plus common prefixed environment-variable, assignment, CLI, JSON, and provider-token shapes in imported platform, command, and note fields; 15 focused telemetry tests pass. Treat this as a narrow fix, not a complete privacy qualification: broaden adversarial input coverage and review hashing, retention, and untrusted structured fields before expanding collection. The closed-file copy/restore check passed, but online or automated backup was not tested. No token-cost benchmark or provider-usage integration has been implemented.

## Implementation sequence

1. **Foundation (partially implemented here):** check in the schema, opt-in DuckDB extra, database initializer, idempotent importer for `testing/results.csv`, summary query, redaction tests, and schema/user documentation. Keep the generated `.duckdb` file out of version control.
2. **Automated capture:** opt-in per-test pytest capture is implemented and integration-tested. Next add a structured JSONL stage-event writer. Have parallel workers emit separate JSONL spools and let one collector import them. Test retries, crash/interruption handling, and duplicate delivery.
3. **Existing report integration:** ingest resource-budget reports, benchmark CSV/JSON, quality outcomes, skip reasons, and artifact manifests without changing model behavior or mutation authority.
4. **Query pack and UI:** add parameterized SQL reports plus an optional CLI/observatory view, each showing query definition, data coverage, caveats, and provenance. Benchmark database filtering against LLM context tokens.
5. **Engine bake-off:** run the representative workloads against the baselines above. Publish raw measurements and environment metadata; promote DuckDB only if gates pass. If ingestion becomes truly multi-process/multi-host or needs operational retention/permissions, reevaluate PostgreSQL or ClickHouse rather than forcing DuckDB into that role.
6. **Swift and C# experiments:** implement only after target hardware or .NET deployment requirements are named. Each lane emits the same run/event/measurement schema and receives a separate qualified status.

## Files added in this increment

- `src/autobliteratus/observability/schema-v1.sql` — versioned table and view definitions included in package builds.
- `src/autobliteratus/observability/telemetry_store.py` — optional initializer, CSV importer, summary query, and basic credential-pattern redaction.
- `src/autobliteratus/observability/pytest_plugin.py` — opt-in per-test capture with hashed identifiers and a single transactional batch write per pytest run.
- `scripts/telemetry_db.py` — local database init/import/summary CLI.
- `tests/test_telemetry_store.py` and `tests/test_telemetry_plugin.py` — schema/import checks and an opt-in subprocess capture/privacy integration test.
- `scripts/benchmark_telemetry_engines.py` — reproducible CSV/Python, SQLite, and DuckDB parity/performance comparison on the real ledger.
- `docs/telemetry-engine-benchmark-2026-11-22.md` and `benchmarks/results/telemetry-ledger-engine-benchmark-2026-11-22.json` — final parity-checked smoke comparison and detailed results.
- `scripts/assess_telemetry_system.py`, `docs/telemetry-system-assessment-2026-11-22.md`, and `benchmarks/results/telemetry-system-assessment-2026-11-22.json` — repeated-import, schema, read-concurrency, copy/restore, and redaction-probe assessment.
- `pyproject.toml` — optional `observability` extra; DuckDB is not a core runtime requirement.

The local draft addresses the confirmed redaction misses and raw command-field leakage. The next step is broader security review and persisted-value coverage for additional credential encodings and telemetry fields. Do not mark all telemetry “implemented” until retry/interrupted/collection cases, structured pipeline events, measurement adapters, broad privacy tests, representative real-workload benchmarks, and online/automated backup and restore tests are complete.

## References

[1]: https://duckdb.org/docs/2.0/guides/file_formats/query_parquet "DuckDB querying Parquet files"
[2]: https://duckdb.org/docs/2.0/connect/concurrency "DuckDB concurrency"
[3]: https://www.sqlite.org/wal.html "SQLite write-ahead logging"
[4]: https://sqlite.org/fts5.html "SQLite FTS5"
[5]: https://www.postgresql.org/docs/current/textsearch-intro.html "PostgreSQL full-text search"
[6]: https://clickhouse.com/docs/clickstack/overview "ClickHouse ClickStack observability"
[7]: https://opensource.apple.com/projects/mlx "Apple MLX project"
[8]: https://developer.apple.com/documentation/foundationmodels/expanding-generation-with-tool-calling "Apple Foundation Models tool calling"
[9]: https://learn.microsoft.com/en-us/semantic-kernel/overview/ "Microsoft Semantic Kernel overview"
[10]: https://csharp.sdk.modelcontextprotocol.io/ "Official MCP C# SDK"
