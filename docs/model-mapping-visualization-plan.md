# Model Mapping Visualization UI Plan

## Product scope

Build a separate web UI for inspecting evidence-backed AUTOBLITERATUS model mappings as an interactive 3D neural-matrix and mind-map view. The first version is read-only for model weights: users can select layers, inspect connections, filter by mapping category and evidence status, and create a proposed modification plan. The UI must never mutate weights directly; proposed operations are exported as a versioned plan for the sandbox/discovery or runtime pipeline to validate.

The interface will use React, Vite, Three.js/WebGL, and Tailwind CSS. Node.js will serve the development and optional API boundary. Blender, XMind, MindNode, Unity, Triton, CUDA, and Vulkan are represented as import/export or future backend adapters, not assumed to be available in the browser or current Python environment.

## Three visual directions

- **Neural Observatory / 0.07**: dark scientific instrument with cyan topology, amber evidence markers, and restrained grid overlays.
- **Cognitive Cartography / 0.04**: atlas-like mapping surface with warm paper tones, route lines, and layered annotations.
- **Systems Control Room / 0.08**: high-contrast operations console with dense telemetry, compact panels, and red/green gate states.

Chosen direction: **Neural Observatory**. It best communicates that the UI is an inspection instrument rather than a toy brain visualization.

## Design decisions

The design movement is scientific visualization with observatory instrumentation. Core principles are evidence before decoration, spatial context before lists, calm density, and explicit safety boundaries. The palette uses near-black navy for the analysis field, cold blue/cyan for structure, violet for active selection, amber for unverified evidence, and red only for blocked or unsafe operations. The layout is a split observatory: a persistent left command rail, a dominant 3D viewport, and a right evidence inspector with a lower timeline/status strip rather than a centered dashboard grid.

Signature elements are a faint orbital grid behind the neural graph, thin luminous connection filaments whose opacity reflects evidence confidence, and a vertical “mapping ledger” that shows verified, unknown, and blocked states. Interactions are deliberate: orbit, pan, and zoom are smooth; selecting a node causes connected paths to breathe once; unsafe actions are visually distinct and produce a plan artifact instead of executing. Animation is sparse and purposeful, with low-frequency node pulses and short selection transitions. Typography uses Inter for operational UI and IBM Plex Mono for hashes, tensor names, dimensions, and evidence IDs. Brand essence: “A forensic instrument for understanding model structure before changing it.” Personality: precise, restrained, investigative.

Brand voice uses direct copy such as “Inspect evidence before editing” and “This action creates a proposal; it does not modify weights.” The wordmark is a compact orbital-bracket mark around “AB”, and the ownable color is Evidence Cyan, reserved for measured structure rather than generic actions.

## Architecture

The UI is a separate `ui/model-mapping-observatory` web project. The visualization model is normalized JSON containing model identity, architecture, tensor/module nodes, typed edges, evidence status, fingerprints, and optional benchmark metrics. Three.js renders nodes and edges; React owns selection, filters, panels, and plan state; Tailwind owns layout and styling. A small Node API boundary can later serve mapping reports, but the initial app uses deterministic fixture data and file-import/export so it is useful without a live server.

The browser never receives unrestricted model weights. It receives metadata, tensor shapes, hashes, evidence records, and bounded visualization summaries. A proposal contains selected node IDs, intended operation, preconditions, and evidence references. Export is JSON only until a pipeline-side validator accepts the proposal.

Adapters are kept behind explicit contracts: Blender for scene export, XMind/MindNode for hierarchical map exchange, Unity for future interactive scene review, and Triton/CUDA/Vulkan for future measured backend telemetry. None is enabled merely by naming it in the UI.

## Project structure

```text
ui/model-mapping-observatory/
  src/
    app/                 route shell and application state
    components/          viewport, inspector, ledger, controls
    data/                deterministic fixture and schema helpers
    lib/                 graph layout, plan validation, import/export
    styles/              Tailwind entry and observatory tokens
  public/manus-routes.json
  plan.md
  TODO.md
```

## Delivery boundary

The first build is an inspection and proposal UI. It will not claim to render every hidden activation or full model weight matrix; it renders the mapping evidence available in the supplied JSON contract. Real model reports can be imported after schema validation. Runtime modification remains in AUTOBLITERATUS pipelines and requires their existing evidence, resource, and rollback gates.
