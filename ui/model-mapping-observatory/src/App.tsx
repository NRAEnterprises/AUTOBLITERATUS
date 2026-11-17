import { useMemo, useState } from 'react'
import {
  Activity, ArrowDownRight, ArrowUpRight, Atom, BrainCircuit, ChevronDown, CircleDot, ClipboardCheck, Download, Eye, FileJson, Filter, GitBranch, Layers3, LockKeyhole, Maximize2, Network, Orbit, PanelRight, Plus, Search, ShieldCheck, SlidersHorizontal, Sparkles, Table2, Workflow, X,
} from 'lucide-react'
import { NeuralViewport } from './components/NeuralViewport'
import { evidenceMeta, mappingEdges, mappingNodes, matrixFor, type EvidenceStatus, type MappingNode } from './data/modelGraph'

const backendStatuses = [
  { name: 'Three.js / WebGL', status: 'active', detail: '3D viewport' },
  { name: 'Node.js bridge', status: 'staged', detail: 'fixture transport' },
  { name: 'Blender / Unity', status: 'adapter', detail: 'scene export' },
  { name: 'Triton / CUDA / Vulkan', status: 'context', detail: 'compute metadata' },
]

const statusOrder: EvidenceStatus[] = ['verified', 'observed', 'unknown', 'blocked']

function StatusDot({ status }: { status: EvidenceStatus }) {
  return <span className={`inline-block h-2 w-2 rounded-full ${evidenceMeta[status].dot}`} />
}

function App() {
  const [selectedId, setSelectedId] = useState('attn-00')
  const [query, setQuery] = useState('')
  const [activeStatus, setActiveStatus] = useState<EvidenceStatus | 'all'>('all')
  const [view, setView] = useState<'graph' | 'matrix' | 'mindmap'>('graph')
  const [showProposal, setShowProposal] = useState(false)
  const [proposalType, setProposalType] = useState('Inspect in sandbox')
  const [proposalNote, setProposalNote] = useState('')
  const [proposals, setProposals] = useState<string[]>([])

  const selected = mappingNodes.find((node) => node.id === selectedId) ?? mappingNodes[0]
  const filteredNodes = useMemo(() => mappingNodes.filter((node) => {
    const text = `${node.label} ${node.tensor} ${node.group} ${node.tags.join(' ')}`.toLowerCase()
    return text.includes(query.toLowerCase()) && (activeStatus === 'all' || node.evidence === activeStatus)
  }), [activeStatus, query])
  const visibleIds = new Set(filteredNodes.map((node) => node.id))
  const visibleEdges = mappingEdges.filter((edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target))
  const matrix = matrixFor(selected)

  const createProposal = () => {
    const label = `${proposalType} · ${selected.shortLabel}`
    setProposals((items) => [label, ...items])
    setShowProposal(false)
    setProposalNote('')
  }

  const exportPlan = () => {
    const payload = {
      schema_version: 1,
      plan_id: `proposal-${Date.now()}`,
      model_identity: 'Qwen2.5 / 0.5B · fixture mapping report',
      safety: { direct_weight_mutation: false, requires_pipeline_validation: true },
      selected_node: selected.id,
      proposals: proposals.map((label) => ({ label, evidence: selected.evidence, evidence_ref: selected.tensor })),
    }
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }))
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = 'autobliteratus-mapping-proposal.json'
    anchor.click()
    URL.revokeObjectURL(url)
  }

  const exportTopology = () => {
    const payload = {
      schema_version: 1,
      export_kind: 'model_mapping_observatory_topology',
      generated_at: new Date().toISOString(),
      model_identity: { name: 'Qwen2.5 / 0.5B', revision: 'fixture-report-2026.11.17-A', architecture: 'decoder-only' },
      safety: { direct_weight_mutation: false, downstream_mode: 'sandboxed_proposal', requires_pipeline_validation: true },
      viewport: { view, selected_node_id: selected.id, filter: { query, evidence: activeStatus } },
      nodes: mappingNodes,
      edges: mappingEdges,
      mind_map: { root_id: 'model-root', branch_ids: ['layer-00', 'layer-12', 'layer-23', 'final-norm'], interchange: ['TheBrain', 'XMind', 'MindNode'] },
      backend_context: backendStatuses,
    }
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }))
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = 'autobliteratus-3d-mind-map-topology.json'
    anchor.click()
    URL.revokeObjectURL(url)
  }

  return (
    <main className="min-h-screen bg-[#050b16] text-[#dcecff] selection:bg-cyan-300/20">
      <header className="flex h-[74px] items-center justify-between border-b border-white/[0.08] bg-[#07101f]/95 px-6 backdrop-blur-xl">
        <div className="flex items-center gap-3">
          <div className="brand-mark"><Orbit size={20} strokeWidth={1.5} /></div>
          <div className="header-title">
            <div className="flex items-center gap-2 text-[11px] font-semibold tracking-[0.26em] text-cyan-200/80"><span>AB</span><span className="text-white/20">/</span><span>MODEL MAPPING OBSERVATORY</span></div>
            <div className="mt-1 flex items-center gap-2 text-[10px] text-white/40"><span className="h-1.5 w-1.5 rounded-full bg-[#5be7c4] shadow-[0_0_12px_#5be7c4]" /> READ-ONLY INSPECTION SURFACE <span className="text-white/20">·</span> RUN 2026.11.17-A</div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div className="hidden items-center gap-2 rounded-full border border-white/[0.08] bg-white/[0.03] px-3 py-2 text-[10px] text-white/55 md:flex"><LockKeyhole size={12} className="text-amber-300" /> SOURCE WEIGHTS LOCKED</div>
          <button className="icon-button" title="Toggle fullscreen framing"><Maximize2 size={16} /></button>
          <button className="secondary-button hidden sm:inline-flex" onClick={exportTopology}><FileJson size={14} /> Export topology</button>
          <button className="primary-button" onClick={exportPlan}><Download size={14} /> Export plan</button>
        </div>
      </header>

      <div className="observatory-grid">
        <aside className="border-r border-white/[0.07] bg-[#07101f]/75 px-4 py-5">
          <div className="eyebrow"><Layers3 size={13} /> MODEL EXPLORER</div>
          <div className="relative mt-4">
            <Search size={14} className="absolute left-3 top-3 text-white/35" />
            <input className="field pl-9" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search tensors, layers..." />
          </div>
          <div className="mt-4 flex items-center justify-between text-[10px] uppercase tracking-[0.16em] text-white/35"><span>Evidence filter</span><Filter size={12} /></div>
          <div className="mt-2 space-y-1">
            <button className={`filter-row ${activeStatus === 'all' ? 'filter-row-active' : ''}`} onClick={() => setActiveStatus('all')}><span className="flex items-center gap-2"><CircleDot size={12} /> All mapped areas</span><span>{mappingNodes.length}</span></button>
            {statusOrder.map((status) => <button key={status} className={`filter-row ${activeStatus === status ? 'filter-row-active' : ''}`} onClick={() => setActiveStatus(status)}><span className="flex items-center gap-2"><StatusDot status={status} /> {evidenceMeta[status].label}</span><span>{mappingNodes.filter((node) => node.evidence === status).length}</span></button>)}
          </div>
          <div className="mt-6 flex items-center justify-between"><div className="eyebrow"><SlidersHorizontal size={13} /> AREAS</div><span className="mono text-white/35">{filteredNodes.length.toString().padStart(2, '0')}</span></div>
          <div className="mt-2 space-y-1 overflow-y-auto pb-4 lg:max-h-[calc(100vh-410px)]">
            {filteredNodes.map((node) => <button key={node.id} className={`node-row ${selected.id === node.id ? 'node-row-active' : ''}`} onClick={() => setSelectedId(node.id)}><span className="min-w-0"><span className="block truncate text-[11px] font-medium text-white/85">{node.shortLabel}</span><span className="mt-1 block truncate text-[9px] text-white/35">{node.group} · {node.dimensions}</span></span><StatusDot status={node.evidence} /></button>)}
          </div>
          <div className="mt-auto border-t border-white/[0.07] pt-4"><div className="eyebrow"><Network size={13} /> INTERCHANGE</div><div className="mt-3 space-y-2 text-[10px] text-white/45"><div className="flex items-center justify-between"><span>TheBrain</span><span className="status-chip chip-staged">MAPPING</span></div><div className="flex items-center justify-between"><span>XMind</span><span className="status-chip chip-staged">MAPPING</span></div><div className="flex items-center justify-between"><span>MindNode</span><span className="status-chip chip-staged">MAPPING</span></div></div></div>
        </aside>

        <section className="min-w-0 px-4 py-4 lg:px-5">
          <div className="flex flex-wrap items-end justify-between gap-3 px-1"><div><div className="eyebrow text-cyan-200/70"><BrainCircuit size={14} /> SPATIAL MAPPING / {selected.kind.toUpperCase()}</div><h1 className="mt-2 text-2xl font-light tracking-[-0.03em] text-white">{selected.label}</h1><p className="mt-1 text-xs text-white/40">Pinned to <span className="mono text-white/65">{selected.tensor}</span> · {selected.dimensions} · {selected.parameters}</p></div><div className="view-switch"><button className={view === 'graph' ? 'view-active' : ''} onClick={() => setView('graph')}><Orbit size={13} /> Graph</button><button className={view === 'matrix' ? 'view-active' : ''} onClick={() => setView('matrix')}><Table2 size={13} /> Matrix</button><button className={view === 'mindmap' ? 'view-active' : ''} onClick={() => setView('mindmap')}><Workflow size={13} /> Mind map</button></div></div>
          <div className="viewport-shell mt-4">
            <div className="viewport-toolbar"><div className="flex items-center gap-3"><span className="status-chip chip-live"><span className="h-1.5 w-1.5 rounded-full bg-cyan-300 shadow-[0_0_8px_#58b9ff]" /> LIVE SCENE</span><span className="text-[10px] text-white/35">{filteredNodes.length} nodes · {visibleEdges.length} connections</span></div><div className="flex items-center gap-3 text-[10px] text-white/35"><span>Drag to orbit</span><span>Scroll to zoom</span><span className="text-cyan-300/70">Click to inspect</span></div></div>
            <div className="h-[min(56vh,560px)] min-h-[430px]">
              {view === 'graph' && <NeuralViewport nodes={filteredNodes} edges={visibleEdges} selectedId={selected.id} onSelect={setSelectedId} />}
              {view === 'matrix' && <MatrixView node={selected} matrix={matrix} />}
              {view === 'mindmap' && <MindMapView selected={selected} onSelect={setSelectedId} />}
            </div>
            <div className="scene-legend"><span><span className="legend-dot bg-[#5be7c4]" /> verified</span><span><span className="legend-dot bg-[#58b9ff]" /> observed</span><span><span className="legend-dot bg-[#f1b861]" /> unknown</span><span><span className="legend-dot bg-[#fa6d82]" /> blocked</span></div>
          </div>
          <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-4">{backendStatuses.map((backend) => <div key={backend.name} className="backend-card"><div className="flex items-center justify-between"><span className="text-[10px] font-medium text-white/70">{backend.name}</span><span className={`status-chip ${backend.status === 'active' ? 'chip-live' : 'chip-staged'}`}>{backend.status}</span></div><div className="mt-2 text-[9px] uppercase tracking-[0.12em] text-white/30">{backend.detail}</div></div>)}</div>
          <div className="mt-4 flex items-center justify-between border-t border-white/[0.07] px-1 pt-3 text-[10px] text-white/35"><span className="flex items-center gap-2"><ShieldCheck size={13} className="text-[#5be7c4]" /> Evidence-first render · no direct weight writes</span><span className="mono">schema v1 · fixture report</span></div>
        </section>

        <aside className="border-l border-white/[0.07] bg-[#07101f]/70 px-4 py-5">
          <div className="flex items-center justify-between"><div className="eyebrow"><PanelRight size={13} /> EVIDENCE INSPECTOR</div><button className="icon-button-sm"><ChevronDown size={14} /></button></div>
          <div className="mt-5 flex items-start justify-between gap-3"><div><div className="text-lg font-medium text-white">{selected.shortLabel}</div><div className="mt-1 mono text-[10px] text-white/35">{selected.id}</div></div><span className="status-chip" style={{ color: evidenceMeta[selected.evidence].color, borderColor: `${evidenceMeta[selected.evidence].color}40`, backgroundColor: `${evidenceMeta[selected.evidence].color}12` }}><StatusDot status={selected.evidence} /> {evidenceMeta[selected.evidence].label}</span></div>
          <p className="mt-4 text-xs leading-5 text-white/55">{selected.summary}</p>
          <div className="mt-5 grid grid-cols-2 gap-2">{[['Confidence', `${Math.round(selected.confidence * 100)}%`], ['Parameters', selected.parameters], ['Activation', selected.metrics.activation], ['Memory', selected.metrics.memory]].map(([label, value]) => <div key={label} className="metric-box"><div className="eyebrow text-[9px]">{label}</div><div className="mt-2 mono text-xs text-white/80">{value}</div></div>)}</div>
          <div className="mt-5"><div className="eyebrow">TENSOR HANDLE</div><div className="mono mt-2 break-all rounded-lg border border-white/[0.08] bg-black/20 p-3 text-[10px] leading-5 text-cyan-100/65">{selected.tensor}</div></div>
          <div className="mt-5"><div className="eyebrow">LABELS</div><div className="mt-2 flex flex-wrap gap-1.5">{selected.tags.map((tag) => <span key={tag} className="tag">{tag}</span>)}</div></div>
          <div className="mt-5 border-t border-white/[0.07] pt-4"><div className="eyebrow"><ClipboardCheck size={13} /> CHANGE PROPOSAL</div><p className="mt-2 text-[10px] leading-4 text-white/40">Draft a reviewable action. The source model remains locked.</p><button className="secondary-button mt-3 w-full" onClick={() => setShowProposal(true)}><Plus size={14} /> Draft proposal</button>{proposals.length > 0 && <div className="mt-3 space-y-2">{proposals.map((proposal, index) => <div key={`${proposal}-${index}`} className="proposal-row"><span className="h-1.5 w-1.5 rounded-full bg-violet-300" /><span>{proposal}</span></div>)}</div>}</div>
          <div className="mt-6 border-t border-white/[0.07] pt-4"><div className="eyebrow"><Activity size={13} /> RUNTIME SIGNAL</div><div className="mt-3 flex items-center justify-between text-[10px] text-white/40"><span>Latency</span><span className="mono text-white/75">{selected.metrics.latency}</span></div><div className="mt-2 h-1 overflow-hidden rounded-full bg-white/[0.07]"><div className="h-full w-[68%] rounded-full bg-gradient-to-r from-cyan-400 to-violet-400" /></div></div>
        </aside>
      </div>

      {showProposal && <div className="modal-backdrop" onClick={() => setShowProposal(false)}><div className="proposal-modal" onClick={(event) => event.stopPropagation()}><div className="flex items-start justify-between"><div><div className="eyebrow text-violet-200/80"><Sparkles size={13} /> PROPOSAL DRAFT</div><h2 className="mt-2 text-xl text-white">Draft a safe next step</h2><p className="mt-1 text-xs text-white/40">{selected.shortLabel} · {selected.evidence} evidence</p></div><button className="icon-button" onClick={() => setShowProposal(false)}><X size={16} /></button></div><label className="eyebrow mt-6 block">ACTION TYPE<select className="field mt-2" value={proposalType} onChange={(event) => setProposalType(event.target.value)}><option>Inspect in sandbox</option><option>Compare against adjacent layer</option><option>Trace downstream impact</option><option>Draft surgical modification</option></select></label><label className="eyebrow mt-4 block">RATIONALE<textarea className="field mt-2 min-h-[90px] resize-none" value={proposalNote} onChange={(event) => setProposalNote(event.target.value)} placeholder="What should the pipeline verify next?" /></label><div className="mt-5 flex items-center justify-between rounded-lg border border-amber-300/20 bg-amber-300/[0.06] p-3 text-[10px] leading-4 text-amber-100/65"><LockKeyhole size={14} className="shrink-0 text-amber-300" /><span>This creates a plan artifact only. No model weights are changed by the observatory.</span></div><div className="mt-5 flex justify-end gap-2"><button className="secondary-button" onClick={() => setShowProposal(false)}>Cancel</button><button className="primary-button" onClick={createProposal}>Add to plan <ArrowUpRight size={14} /></button></div></div></div>}
    </main>
  )
}

function MatrixView({ node, matrix }: { node: MappingNode; matrix: number[][] }) {
  return <div className="matrix-view"><div className="matrix-header"><div><div className="eyebrow"><Table2 size={13} /> VALUE FIELD / {node.shortLabel.toUpperCase()}</div><p className="mt-2 text-xs text-white/45">Deterministic preview of the selected tensor surface · values normalized to [-1, 1]</p></div><div className="matrix-scale"><span>-1</span><i /><span>0</span><i className="scale-hot" /><span>+1</span></div></div><div className="matrix-grid">{matrix.flatMap((row, rowIndex) => row.map((value, colIndex) => <div key={`${rowIndex}-${colIndex}`} className="matrix-cell" style={{ backgroundColor: value > 0 ? `rgba(88,185,255,${0.12 + value * 0.34})` : `rgba(179,113,255,${0.12 + Math.abs(value) * 0.32})` }} title={`r${rowIndex} c${colIndex}: ${value.toFixed(3)}`} />))}</div><div className="matrix-footer"><span>rows 08 · cols 16 · sample window 128 values</span><span className="mono text-cyan-200/60">tensor fingerprint 7f:19:ac</span></div></div>
}

function MindMapView({ selected, onSelect }: { selected: MappingNode; onSelect: (id: string) => void }) {
  const branches = mappingNodes.filter((node) => ['layer-00', 'layer-12', 'layer-23', 'final-norm'].includes(node.id))
  return <div className="mindmap-view"><div className="eyebrow"><Workflow size={13} /> LINKED MIND MAP / EVIDENCE TOPOLOGY</div><div className="mindmap-stage"><div className="mindmap-core"><Atom size={18} /><span>Qwen2.5<br /><b>model root</b></span></div>{branches.map((node, index) => <button key={node.id} className={`mindmap-node mindmap-node-${index} ${selected.id === node.id ? 'mindmap-selected' : ''}`} onClick={() => onSelect(node.id)}><span className="mindmap-line" /><StatusDot status={node.evidence} /><span>{node.shortLabel}</span><small>{node.group}</small></button>)}</div><div className="mindmap-footer"><span>Interchange framing ready</span><span className="flex items-center gap-2"><FileJson size={13} /> TheBrain · XMind · MindNode</span></div></div>
}

export default App
