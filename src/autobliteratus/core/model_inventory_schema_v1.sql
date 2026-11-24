PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_versions (
    schema_version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL,
    schema_sha256 TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS inventory_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    structure_id TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    captured_at TEXT NOT NULL,
    model_class TEXT NOT NULL,
    model_type TEXT,
    source_revision TEXT,
    status TEXT NOT NULL CHECK (status IN ('complete', 'partial')),
    manifest_json TEXT NOT NULL,
    FOREIGN KEY (schema_version) REFERENCES schema_versions(schema_version)
);

CREATE TABLE IF NOT EXISTS modules (
    snapshot_id TEXT NOT NULL,
    module_id TEXT NOT NULL,
    canonical_path TEXT NOT NULL,
    aliases_json TEXT NOT NULL,
    parent_module_id TEXT,
    module_type TEXT NOT NULL,
    signature_json TEXT NOT NULL,
    child_count INTEGER NOT NULL,
    is_leaf INTEGER NOT NULL CHECK (is_leaf IN (0, 1)),
    PRIMARY KEY (snapshot_id, module_id),
    FOREIGN KEY (snapshot_id) REFERENCES inventory_snapshots(snapshot_id),
    FOREIGN KEY (snapshot_id, parent_module_id) REFERENCES modules(snapshot_id, module_id)
);

CREATE TABLE IF NOT EXISTS tensors (
    snapshot_id TEXT NOT NULL,
    tensor_id TEXT NOT NULL,
    state_name TEXT NOT NULL,
    shape_json TEXT NOT NULL,
    dtype TEXT NOT NULL,
    layout TEXT,
    strides_json TEXT,
    element_count INTEGER,
    value_sha256 TEXT,
    hash_status TEXT NOT NULL CHECK (hash_status IN ('verified', 'unavailable')),
    storage_alias_group TEXT,
    state_kind TEXT NOT NULL CHECK (state_kind IN ('parameter', 'buffer', 'state_dict', 'unknown')),
    PRIMARY KEY (snapshot_id, tensor_id),
    UNIQUE (snapshot_id, state_name),
    FOREIGN KEY (snapshot_id) REFERENCES inventory_snapshots(snapshot_id)
);

CREATE TABLE IF NOT EXISTS module_tensor_bindings (
    snapshot_id TEXT NOT NULL,
    module_id TEXT NOT NULL,
    tensor_id TEXT NOT NULL,
    local_name TEXT NOT NULL,
    relation TEXT NOT NULL CHECK (relation IN ('parameter', 'buffer', 'state_dict')),
    PRIMARY KEY (snapshot_id, module_id, tensor_id, local_name, relation),
    FOREIGN KEY (snapshot_id, module_id) REFERENCES modules(snapshot_id, module_id),
    FOREIGN KEY (snapshot_id, tensor_id) REFERENCES tensors(snapshot_id, tensor_id)
);

CREATE TABLE IF NOT EXISTS trace_runs (
    trace_id TEXT PRIMARY KEY,
    snapshot_id TEXT NOT NULL,
    trace_kind TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('measured', 'partial', 'unsupported', 'failed')),
    created_at TEXT NOT NULL,
    environment_json TEXT NOT NULL,
    input_profile_json TEXT NOT NULL,
    limitations_json TEXT NOT NULL,
    FOREIGN KEY (snapshot_id) REFERENCES inventory_snapshots(snapshot_id)
);

CREATE TABLE IF NOT EXISTS graph_nodes (
    trace_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    node_kind TEXT NOT NULL,
    target TEXT NOT NULL,
    module_id TEXT,
    attributes_json TEXT NOT NULL,
    PRIMARY KEY (trace_id, node_id),
    FOREIGN KEY (trace_id) REFERENCES trace_runs(trace_id)
);

CREATE TABLE IF NOT EXISTS graph_edges (
    trace_id TEXT NOT NULL,
    source_node_id TEXT NOT NULL,
    target_node_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    PRIMARY KEY (trace_id, source_node_id, target_node_id, relation),
    FOREIGN KEY (trace_id, source_node_id) REFERENCES graph_nodes(trace_id, node_id),
    FOREIGN KEY (trace_id, target_node_id) REFERENCES graph_nodes(trace_id, node_id)
);

CREATE TABLE IF NOT EXISTS role_assertions (
    assertion_id TEXT PRIMARY KEY,
    snapshot_id TEXT NOT NULL,
    target_kind TEXT NOT NULL CHECK (target_kind IN ('module', 'tensor', 'binding')),
    target_id TEXT NOT NULL,
    taxonomy TEXT NOT NULL,
    role TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('candidate', 'structurally_supported', 'experimentally_supported', 'unknown', 'unsupported')),
    confidence REAL CHECK (confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
    detector_name TEXT NOT NULL,
    detector_version TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (snapshot_id) REFERENCES inventory_snapshots(snapshot_id)
);

CREATE TABLE IF NOT EXISTS probe_runs (
    probe_run_id TEXT PRIMARY KEY,
    snapshot_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('measured', 'partial', 'failed')),
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    FOREIGN KEY (snapshot_id) REFERENCES inventory_snapshots(snapshot_id)
);

CREATE TABLE IF NOT EXISTS layer_probes (
    probe_id TEXT PRIMARY KEY,
    probe_run_id TEXT NOT NULL,
    snapshot_id TEXT NOT NULL,
    module_id TEXT NOT NULL,
    module_path TEXT NOT NULL,
    probe_name TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('measured', 'unknown', 'failed')),
    isolated INTEGER NOT NULL CHECK (isolated IN (0, 1)),
    measurement_signature TEXT,
    measurement_json TEXT NOT NULL,
    warning_json TEXT NOT NULL,
    limitations_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (probe_run_id) REFERENCES probe_runs(probe_run_id),
    FOREIGN KEY (snapshot_id, module_id) REFERENCES modules(snapshot_id, module_id)
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    evaluation_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    split_name TEXT NOT NULL,
    detector_name TEXT NOT NULL,
    detector_version TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    held_out_group_ids_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_modules_snapshot_path ON modules(snapshot_id, canonical_path);
CREATE INDEX IF NOT EXISTS idx_tensors_snapshot_name ON tensors(snapshot_id, state_name);
CREATE INDEX IF NOT EXISTS idx_roles_snapshot_target ON role_assertions(snapshot_id, target_kind, target_id);
CREATE INDEX IF NOT EXISTS idx_probes_snapshot_module ON layer_probes(snapshot_id, module_id);

CREATE TRIGGER IF NOT EXISTS immutable_schema_versions_update BEFORE UPDATE ON schema_versions BEGIN SELECT RAISE(ABORT, 'inventory evidence is append-only'); END;
CREATE TRIGGER IF NOT EXISTS immutable_schema_versions_delete BEFORE DELETE ON schema_versions BEGIN SELECT RAISE(ABORT, 'inventory evidence is append-only'); END;
CREATE TRIGGER IF NOT EXISTS immutable_inventory_snapshots_update BEFORE UPDATE ON inventory_snapshots BEGIN SELECT RAISE(ABORT, 'inventory evidence is append-only'); END;
CREATE TRIGGER IF NOT EXISTS immutable_inventory_snapshots_delete BEFORE DELETE ON inventory_snapshots BEGIN SELECT RAISE(ABORT, 'inventory evidence is append-only'); END;
CREATE TRIGGER IF NOT EXISTS immutable_modules_update BEFORE UPDATE ON modules BEGIN SELECT RAISE(ABORT, 'inventory evidence is append-only'); END;
CREATE TRIGGER IF NOT EXISTS immutable_modules_delete BEFORE DELETE ON modules BEGIN SELECT RAISE(ABORT, 'inventory evidence is append-only'); END;
CREATE TRIGGER IF NOT EXISTS immutable_tensors_update BEFORE UPDATE ON tensors BEGIN SELECT RAISE(ABORT, 'inventory evidence is append-only'); END;
CREATE TRIGGER IF NOT EXISTS immutable_tensors_delete BEFORE DELETE ON tensors BEGIN SELECT RAISE(ABORT, 'inventory evidence is append-only'); END;
CREATE TRIGGER IF NOT EXISTS immutable_module_tensor_bindings_update BEFORE UPDATE ON module_tensor_bindings BEGIN SELECT RAISE(ABORT, 'inventory evidence is append-only'); END;
CREATE TRIGGER IF NOT EXISTS immutable_module_tensor_bindings_delete BEFORE DELETE ON module_tensor_bindings BEGIN SELECT RAISE(ABORT, 'inventory evidence is append-only'); END;
CREATE TRIGGER IF NOT EXISTS immutable_trace_runs_update BEFORE UPDATE ON trace_runs BEGIN SELECT RAISE(ABORT, 'inventory evidence is append-only'); END;
CREATE TRIGGER IF NOT EXISTS immutable_trace_runs_delete BEFORE DELETE ON trace_runs BEGIN SELECT RAISE(ABORT, 'inventory evidence is append-only'); END;
CREATE TRIGGER IF NOT EXISTS immutable_graph_nodes_update BEFORE UPDATE ON graph_nodes BEGIN SELECT RAISE(ABORT, 'inventory evidence is append-only'); END;
CREATE TRIGGER IF NOT EXISTS immutable_graph_nodes_delete BEFORE DELETE ON graph_nodes BEGIN SELECT RAISE(ABORT, 'inventory evidence is append-only'); END;
CREATE TRIGGER IF NOT EXISTS immutable_graph_edges_update BEFORE UPDATE ON graph_edges BEGIN SELECT RAISE(ABORT, 'inventory evidence is append-only'); END;
CREATE TRIGGER IF NOT EXISTS immutable_graph_edges_delete BEFORE DELETE ON graph_edges BEGIN SELECT RAISE(ABORT, 'inventory evidence is append-only'); END;
CREATE TRIGGER IF NOT EXISTS immutable_role_assertions_update BEFORE UPDATE ON role_assertions BEGIN SELECT RAISE(ABORT, 'inventory evidence is append-only'); END;
CREATE TRIGGER IF NOT EXISTS immutable_role_assertions_delete BEFORE DELETE ON role_assertions BEGIN SELECT RAISE(ABORT, 'inventory evidence is append-only'); END;
CREATE TRIGGER IF NOT EXISTS immutable_probe_runs_update BEFORE UPDATE ON probe_runs BEGIN SELECT RAISE(ABORT, 'inventory evidence is append-only'); END;
CREATE TRIGGER IF NOT EXISTS immutable_probe_runs_delete BEFORE DELETE ON probe_runs BEGIN SELECT RAISE(ABORT, 'inventory evidence is append-only'); END;
CREATE TRIGGER IF NOT EXISTS immutable_layer_probes_update BEFORE UPDATE ON layer_probes BEGIN SELECT RAISE(ABORT, 'inventory evidence is append-only'); END;
CREATE TRIGGER IF NOT EXISTS immutable_layer_probes_delete BEFORE DELETE ON layer_probes BEGIN SELECT RAISE(ABORT, 'inventory evidence is append-only'); END;
CREATE TRIGGER IF NOT EXISTS immutable_evaluation_runs_update BEFORE UPDATE ON evaluation_runs BEGIN SELECT RAISE(ABORT, 'inventory evidence is append-only'); END;
CREATE TRIGGER IF NOT EXISTS immutable_evaluation_runs_delete BEFORE DELETE ON evaluation_runs BEGIN SELECT RAISE(ABORT, 'inventory evidence is append-only'); END;
