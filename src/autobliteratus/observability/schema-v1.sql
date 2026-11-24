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
