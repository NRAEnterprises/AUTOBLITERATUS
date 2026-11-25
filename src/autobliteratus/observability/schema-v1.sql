CREATE TABLE IF NOT EXISTS telemetry_schema (
    schema_version INTEGER PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    description VARCHAR NOT NULL
);

INSERT INTO telemetry_schema (schema_version, description)
VALUES (1, 'AUTOBLITERATUS pipeline telemetry schema')
ON CONFLICT (schema_version) DO NOTHING;

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id VARCHAR PRIMARY KEY,
    run_date DATE,
    commit_sha VARCHAR,
    platform VARCHAR,
    command VARCHAR,
    status VARCHAR NOT NULL,
    source_name VARCHAR,
    source_sha256 VARCHAR,
    notes_redacted VARCHAR,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS test_summaries (
    run_id VARCHAR PRIMARY KEY REFERENCES pipeline_runs(run_id),
    passed BIGINT NOT NULL CHECK (passed >= 0),
    failed BIGINT NOT NULL CHECK (failed >= 0),
    skipped BIGINT NOT NULL CHECK (skipped >= 0),
    total BIGINT NOT NULL CHECK (total = passed + failed + skipped)
);

CREATE TABLE IF NOT EXISTS test_results (
    run_id VARCHAR NOT NULL REFERENCES pipeline_runs(run_id),
    test_id VARCHAR NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 1 CHECK (attempt >= 1),
    outcome VARCHAR NOT NULL,
    duration_ms DOUBLE,
    failure_type VARCHAR,
    failure_fingerprint VARCHAR,
    log_ref VARCHAR,
    metadata JSON,
    PRIMARY KEY (run_id, test_id, attempt)
);

CREATE TABLE IF NOT EXISTS pipeline_events (
    event_id VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL REFERENCES pipeline_runs(run_id),
    occurred_at TIMESTAMPTZ NOT NULL,
    component VARCHAR NOT NULL,
    stage VARCHAR,
    severity VARCHAR NOT NULL,
    event_type VARCHAR NOT NULL,
    message_redacted VARCHAR,
    correlation_id VARCHAR,
    duration_ms DOUBLE,
    attributes JSON
);

CREATE TABLE IF NOT EXISTS measurements (
    measurement_id VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL REFERENCES pipeline_runs(run_id),
    stage VARCHAR,
    metric VARCHAR NOT NULL,
    value DOUBLE,
    unit VARCHAR NOT NULL,
    measured_at TIMESTAMPTZ NOT NULL,
    dimensions JSON
);

CREATE TABLE IF NOT EXISTS llm_usage (
    usage_id VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL REFERENCES pipeline_runs(run_id),
    stage VARCHAR,
    provider VARCHAR,
    model VARCHAR,
    purpose VARCHAR,
    input_tokens BIGINT CHECK (input_tokens IS NULL OR input_tokens >= 0),
    output_tokens BIGINT CHECK (output_tokens IS NULL OR output_tokens >= 0),
    cached_input_tokens BIGINT CHECK (cached_input_tokens IS NULL OR cached_input_tokens >= 0),
    estimated_cost_usd DOUBLE,
    latency_ms DOUBLE,
    prompt_template_sha256 VARCHAR,
    context_bytes BIGINT,
    measured_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS artifact_refs (
    artifact_sha256 VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL REFERENCES pipeline_runs(run_id),
    artifact_kind VARCHAR NOT NULL,
    uri VARCHAR NOT NULL,
    size_bytes BIGINT,
    content_type VARCHAR,
    model_id VARCHAR,
    model_revision VARCHAR,
    manifest_sha256 VARCHAR,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS query_benchmarks (
    benchmark_id VARCHAR PRIMARY KEY,
    workload_id VARCHAR NOT NULL,
    engine VARCHAR NOT NULL,
    engine_version VARCHAR,
    run_id VARCHAR NOT NULL REFERENCES pipeline_runs(run_id),
    query_id VARCHAR NOT NULL,
    correct BOOLEAN NOT NULL,
    latency_ms DOUBLE NOT NULL,
    rows_returned BIGINT NOT NULL,
    input_tokens BIGINT,
    output_tokens BIGINT,
    context_bytes BIGINT,
    result_sha256 VARCHAR,
    cache_state VARCHAR NOT NULL,
    hardware_fingerprint VARCHAR,
    measured_at TIMESTAMPTZ NOT NULL
);

CREATE VIEW IF NOT EXISTS v_run_outcomes AS
SELECT
    r.run_id,
    r.run_date,
    r.commit_sha,
    r.platform,
    r.status,
    s.passed,
    s.failed,
    s.skipped,
    s.total
FROM pipeline_runs AS r
LEFT JOIN test_summaries AS s USING (run_id);

CREATE VIEW IF NOT EXISTS v_token_efficiency AS
SELECT
    workload_id,
    engine,
    COUNT(*) AS query_count,
    AVG(latency_ms) AS mean_latency_ms,
    quantile_cont(latency_ms, 0.95) AS p95_latency_ms,
    AVG(input_tokens + COALESCE(output_tokens, 0)) AS mean_total_tokens,
    AVG(CASE WHEN correct THEN 1.0 ELSE 0.0 END) AS exact_success_rate
FROM query_benchmarks
GROUP BY workload_id, engine;

-- Resource-budget v1 reports have no source run ID or measurement timestamp.
-- Keep them in dedicated tables rather than inventing either field.
CREATE TABLE IF NOT EXISTS resource_budget_reports (
    report_id VARCHAR PRIMARY KEY,
    stage_id VARCHAR NOT NULL,
    schema_version INTEGER NOT NULL CHECK (schema_version = 1),
    status VARCHAR NOT NULL CHECK (status IN (
        'not_configured', 'not_evaluated', 'violations', 'incomplete', 'within_budget'
    )),
    source_sha256 VARCHAR NOT NULL,
    budget JSON NOT NULL,
    observation_count INTEGER NOT NULL CHECK (observation_count >= 0),
    finding_count INTEGER NOT NULL CHECK (finding_count >= 0),
    imported_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS resource_budget_observations (
    report_id VARCHAR NOT NULL REFERENCES resource_budget_reports(report_id),
    observation_index INTEGER NOT NULL CHECK (observation_index >= 0),
    candidate_index INTEGER NOT NULL CHECK (candidate_index >= 0),
    batch_size BIGINT CHECK (batch_size IS NULL OR batch_size > 0),
    sequence_length BIGINT CHECK (sequence_length IS NULL OR sequence_length > 0),
    tokens_per_second DOUBLE,
    latency_ms DOUBLE,
    host_rss_bytes BIGINT CHECK (host_rss_bytes IS NULL OR host_rss_bytes >= 0),
    accelerator_allocated_bytes BIGINT CHECK (
        accelerator_allocated_bytes IS NULL OR accelerator_allocated_bytes >= 0
    ),
    PRIMARY KEY (report_id, observation_index)
);

CREATE TABLE IF NOT EXISTS resource_budget_findings (
    report_id VARCHAR NOT NULL REFERENCES resource_budget_reports(report_id),
    finding_kind VARCHAR NOT NULL CHECK (
        finding_kind IN ('violation', 'missing_metric', 'candidate_outcome')
    ),
    finding_index INTEGER NOT NULL CHECK (finding_index >= 0),
    code VARCHAR NOT NULL,
    metric VARCHAR,
    budget_key VARCHAR,
    observed JSON,
    limit_value JSON,
    unit VARCHAR,
    details JSON NOT NULL,
    PRIMARY KEY (report_id, finding_kind, finding_index)
);

CREATE TABLE IF NOT EXISTS benchmark_imports (
    report_id VARCHAR PRIMARY KEY,
    report_kind VARCHAR NOT NULL CHECK (
        report_kind IN ('telemetry_engine_json_v1', 'query_benchmark_csv_v1')
    ),
    source_sha256 VARCHAR NOT NULL,
    record_count INTEGER NOT NULL CHECK (record_count > 0),
    imported_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS benchmark_measurements (
    measurement_id VARCHAR PRIMARY KEY,
    report_id VARCHAR NOT NULL REFERENCES benchmark_imports(report_id),
    source_record_index INTEGER NOT NULL CHECK (source_record_index >= 0),
    run_id VARCHAR REFERENCES pipeline_runs(run_id),
    workload_id VARCHAR NOT NULL,
    engine VARCHAR NOT NULL,
    engine_version VARCHAR,
    query_id VARCHAR NOT NULL,
    latency_statistic VARCHAR NOT NULL CHECK (
        latency_statistic IN ('observed', 'median', 'p95', 'first_after_parity')
    ),
    correct BOOLEAN NOT NULL,
    latency_ms DOUBLE NOT NULL CHECK (latency_ms >= 0),
    rows_returned BIGINT NOT NULL CHECK (rows_returned >= 0),
    input_tokens BIGINT CHECK (input_tokens IS NULL OR input_tokens >= 0),
    output_tokens BIGINT CHECK (output_tokens IS NULL OR output_tokens >= 0),
    context_bytes BIGINT CHECK (context_bytes IS NULL OR context_bytes >= 0),
    result_sha256 VARCHAR,
    cache_state VARCHAR CHECK (
        cache_state IS NULL OR cache_state IN ('cold', 'warm', 'after_parity', 'unknown', 'not_applicable')
    ),
    hardware_fingerprint VARCHAR,
    measured_at TIMESTAMPTZ,
    UNIQUE (report_id, source_record_index)
);

CREATE VIEW IF NOT EXISTS v_imported_benchmark_measurements AS
SELECT
    i.report_kind,
    i.source_sha256,
    i.imported_at,
    b.*
FROM benchmark_imports AS i
JOIN benchmark_measurements AS b USING (report_id);
