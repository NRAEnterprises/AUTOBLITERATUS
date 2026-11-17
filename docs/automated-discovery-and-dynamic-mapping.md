# Automated Discovery and Dynamic Mapping

## Scope

The automated discovery pipeline is sandbox-only. It is separate from `autobliteratus.pipeline`, rejects live/runtime contexts, requires a sandbox-bound adapter, and writes only inside the declared sandbox root.

## Experiment contract

`run_unknown_factor_battery` receives the base evidence report and a bounded context. Candidate factors may be supplied explicitly as a category-to-path mapping, or the runner derives candidates from areas that the base discovery report marked unknown.

The battery is bounded by:

- `unknown_factor_max_experiments`, default `64`, maximum `512`;
- `unknown_factor_max_seconds`, default `30`, maximum `600`.

The adapter may provide `sandbox_removal_probe(category, path, context)`. That probe must perform its removal experiment in an isolated copy and return measured booleans:

- `removal_supported: true`;
- `quality_preserved: true`.

The runner strips user-label and promotion fields from measurements, records the experiment identity, compares live model fingerprints before and during the probe, and restores the exact state snapshot after every probe. A probe that mutates live state is failed and cannot promote a mapping. Missing probes, unsupported removal, quality drift, exceptions, and budget truncation remain unknown or failed and are never promoted.

## Dynamic mapping updates

Only experiments with `status=verified` create dynamic mapping records. Records are keyed by the model `identity_sha256`, include the experiment digest, category, path, evidence source, and verified status, and are deduplicated by experiment ID. Static Python mappings remain conservative hypotheses; dynamic evidence is stored separately under the JSON database’s `dynamic_mappings` object.

This database update is atomic and remains inside `sandbox_root`. The runtime pipeline does not consume or promote this database automatically. A future promotion gate must validate replayability, schema signatures, model revision, resource bounds, and quality/regression evidence before any mapping capability can authorize mutation.

## Failure integration

Transaction integration tests inject a simulated downstream quality-gate failure after a state change. The checkpoint restores the exact pre-mutation fingerprint and records a durable `rolled_back` manifest with the failure reason. Discovery probes have an analogous self-restoring boundary so experimental faults cannot leak into the sandbox model instance.
