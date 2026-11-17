export type EvidenceStatus = 'verified' | 'observed' | 'unknown' | 'blocked'
export type NodeKind = 'model' | 'layer' | 'attention' | 'projection' | 'embedding' | 'norm' | 'output'

export type MappingNode = {
  id: string
  label: string
  shortLabel: string
  kind: NodeKind
  group: string
  position: [number, number, number]
  dimensions: string
  parameters: string
  evidence: EvidenceStatus
  confidence: number
  summary: string
  tensor: string
  tags: string[]
  metrics: { activation: string; latency: string; memory: string }
}

export type MappingEdge = {
  source: string
  target: string
  label: string
  weight: number
  relation: 'data-flow' | 'residual' | 'route' | 'projection'
}

export const evidenceMeta: Record<EvidenceStatus, { label: string; color: string; dot: string }> = {
  verified: { label: 'Verified', color: '#5be7c4', dot: 'bg-[#5be7c4]' },
  observed: { label: 'Observed', color: '#58b9ff', dot: 'bg-[#58b9ff]' },
  unknown: { label: 'Unknown', color: '#f1b861', dot: 'bg-[#f1b861]' },
  blocked: { label: 'Blocked', color: '#fa6d82', dot: 'bg-[#fa6d82]' },
}

export const mappingNodes: MappingNode[] = [
  {
    id: 'model-root', label: 'Qwen2.5 / 0.5B', shortLabel: 'Qwen2.5 · 0.5B', kind: 'model', group: 'Model identity', position: [0, 0.2, 0], dimensions: '24 layers · 896 hidden', parameters: '494.0M', evidence: 'verified', confidence: 0.99,
    summary: 'Root identity and architecture envelope bound to a pinned model revision.', tensor: 'model.config', tags: ['pinned revision', 'transformers', 'decoder-only'], metrics: { activation: '—', latency: '—', memory: '1.02 GB' },
  },
  {
    id: 'embed', label: 'Token Embeddings', shortLabel: 'Embeddings', kind: 'embedding', group: 'Input surface', position: [-3.8, 1.4, 0.4], dimensions: '151,936 × 896', parameters: '136.1M', evidence: 'verified', confidence: 0.98,
    summary: 'Vocabulary projection used to construct the initial residual stream.', tensor: 'model.embed_tokens.weight', tags: ['input', 'vocab', 'weight-bound'], metrics: { activation: '896 × 256', latency: '0.24 ms', memory: '519 MB' },
  },
  {
    id: 'layer-00', label: 'Decoder Layer 00', shortLabel: 'Layer 00', kind: 'layer', group: 'Decoder stack', position: [-2.3, 0.2, 0.2], dimensions: '896 hidden · 14 heads', parameters: '18.7M', evidence: 'verified', confidence: 0.97,
    summary: 'First decoder block with attention, normalization, and gated MLP paths.', tensor: 'model.layers.0', tags: ['block', 'residual', 'layer-0'], metrics: { activation: '896 × 256', latency: '1.83 ms', memory: '72 MB' },
  },
  {
    id: 'attn-00', label: 'Attention / Layer 00', shortLabel: 'Attn · 00', kind: 'attention', group: 'Attention path', position: [-0.8, 1.55, 0.6], dimensions: '14 heads · 64 KV', parameters: '4.8M', evidence: 'observed', confidence: 0.92,
    summary: 'Self-attention path traced from QKV projections through output projection.', tensor: 'model.layers.0.self_attn', tags: ['qkv', 'kv-cache', 'observed'], metrics: { activation: '14 × 256 × 64', latency: '0.88 ms', memory: '18 MB' },
  },
  {
    id: 'mlp-00', label: 'Gated MLP / Layer 00', shortLabel: 'MLP · 00', kind: 'projection', group: 'Feed-forward path', position: [-0.7, -1.3, -0.4], dimensions: '4864 intermediate', parameters: '13.7M', evidence: 'verified', confidence: 0.95,
    summary: 'Gated feed-forward branch with down projection returning to the residual stream.', tensor: 'model.layers.0.mlp', tags: ['gate', 'up', 'down'], metrics: { activation: '4864 × 256', latency: '0.95 ms', memory: '42 MB' },
  },
  {
    id: 'norm-00', label: 'RMSNorm / Layer 00', shortLabel: 'Norm · 00', kind: 'norm', group: 'Stability path', position: [0.6, 2.9, -0.5], dimensions: '896 scale values', parameters: '896', evidence: 'verified', confidence: 0.99,
    summary: 'Pre-attention normalization that stabilizes the first block input.', tensor: 'model.layers.0.input_layernorm', tags: ['norm', 'scale'], metrics: { activation: '896 × 256', latency: '0.08 ms', memory: '0.02 MB' },
  },
  {
    id: 'layer-12', label: 'Decoder Layer 12', shortLabel: 'Layer 12', kind: 'layer', group: 'Decoder stack', position: [1.9, 0.1, 0.5], dimensions: '896 hidden · 14 heads', parameters: '18.7M', evidence: 'observed', confidence: 0.88,
    summary: 'Mid-stack layer selected as a representative cross-layer tracing checkpoint.', tensor: 'model.layers.12', tags: ['block', 'mid-stack', 'observed'], metrics: { activation: '896 × 256', latency: '1.79 ms', memory: '72 MB' },
  },
  {
    id: 'attn-12', label: 'Attention / Layer 12', shortLabel: 'Attn · 12', kind: 'attention', group: 'Attention path', position: [3.3, 1.6, 0.1], dimensions: '14 heads · 64 KV', parameters: '4.8M', evidence: 'unknown', confidence: 0.41,
    summary: 'Candidate attention region requiring sandbox experiments before surgical edits.', tensor: 'model.layers.12.self_attn', tags: ['candidate', 'unknown', 'sandbox-only'], metrics: { activation: 'pending', latency: 'pending', memory: 'pending' },
  },
  {
    id: 'mlp-12', label: 'Gated MLP / Layer 12', shortLabel: 'MLP · 12', kind: 'projection', group: 'Feed-forward path', position: [3.2, -1.25, -0.1], dimensions: '4864 intermediate', parameters: '13.7M', evidence: 'unknown', confidence: 0.38,
    summary: 'Unverified MLP region. The UI will only draft a proposal, never mutate this area.', tensor: 'model.layers.12.mlp', tags: ['candidate', 'unknown', 'proposal-only'], metrics: { activation: 'pending', latency: 'pending', memory: 'pending' },
  },
  {
    id: 'layer-23', label: 'Decoder Layer 23', shortLabel: 'Layer 23', kind: 'layer', group: 'Decoder stack', position: [5.3, 0.15, 0.4], dimensions: '896 hidden · 14 heads', parameters: '18.7M', evidence: 'observed', confidence: 0.82,
    summary: 'Final decoder checkpoint before output normalization and language head.', tensor: 'model.layers.23', tags: ['block', 'final-stack', 'observed'], metrics: { activation: '896 × 256', latency: '1.86 ms', memory: '72 MB' },
  },
  {
    id: 'final-norm', label: 'Final RMSNorm', shortLabel: 'Final Norm', kind: 'norm', group: 'Output surface', position: [6.45, 2.0, -0.3], dimensions: '896 scale values', parameters: '896', evidence: 'verified', confidence: 0.99,
    summary: 'Output stabilization layer feeding the vocabulary projection.', tensor: 'model.norm', tags: ['norm', 'output'], metrics: { activation: '896 × 256', latency: '0.08 ms', memory: '0.02 MB' },
  },
  {
    id: 'lm-head', label: 'Language Model Head', shortLabel: 'LM Head', kind: 'output', group: 'Output surface', position: [7.3, 0.25, 0.2], dimensions: '896 → 151,936', parameters: '136.1M', evidence: 'verified', confidence: 0.98,
    summary: 'Vocabulary logits projection. Weight tying is confirmed in the current fixture report.', tensor: 'lm_head.weight', tags: ['logits', 'tied', 'output'], metrics: { activation: '151,936 × 256', latency: '1.24 ms', memory: '519 MB' },
  },
]

export const mappingEdges: MappingEdge[] = [
  { source: 'model-root', target: 'embed', label: 'input schema', weight: 1, relation: 'data-flow' },
  { source: 'embed', target: 'layer-00', label: 'residual stream', weight: 0.96, relation: 'data-flow' },
  { source: 'layer-00', target: 'attn-00', label: 'pre-norm input', weight: 0.92, relation: 'projection' },
  { source: 'layer-00', target: 'mlp-00', label: 'residual branch', weight: 0.91, relation: 'residual' },
  { source: 'norm-00', target: 'attn-00', label: 'normalized query', weight: 0.88, relation: 'data-flow' },
  { source: 'attn-00', target: 'layer-12', label: 'attention output', weight: 0.84, relation: 'data-flow' },
  { source: 'mlp-00', target: 'layer-12', label: 'gated output', weight: 0.8, relation: 'data-flow' },
  { source: 'layer-12', target: 'attn-12', label: 'candidate input', weight: 0.47, relation: 'projection' },
  { source: 'layer-12', target: 'mlp-12', label: 'candidate branch', weight: 0.42, relation: 'residual' },
  { source: 'attn-12', target: 'layer-23', label: 'unverified route', weight: 0.31, relation: 'route' },
  { source: 'mlp-12', target: 'layer-23', label: 'unverified route', weight: 0.29, relation: 'route' },
  { source: 'layer-23', target: 'final-norm', label: 'final residual', weight: 0.89, relation: 'data-flow' },
  { source: 'final-norm', target: 'lm-head', label: 'normalized logits', weight: 0.98, relation: 'data-flow' },
]

export const matrixFor = (node: MappingNode) => {
  const seed = node.id.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0)
  return Array.from({ length: 8 }, (_, row) => Array.from({ length: 16 }, (_, col) => {
    const wave = Math.sin(seed * 0.07 + row * 0.8 - col * 0.37)
    return Math.max(-1, Math.min(1, wave * 0.82 + Math.cos(col * 0.7 + row) * 0.13))
  }))
}
