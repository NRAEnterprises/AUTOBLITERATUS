# Historical prototype roadmap: unavailable switch inventory

> This is a dated historical roadmap. The current registry has 100 total toggles, 22 callable stages, and 78 unavailable prototypes. Use `docs/function-first-pipeline-audit-2026-11-17.md` and `PROJECT_HANDOFF_MANIFEST.yaml` for current status.

Generated from the 2026-11-16 inventory and historical registry snapshot. It is retained for traceability and is not a current release checklist.

## Roadmap sequencing

| Phase | Gate | Exit criteria |
|---|---|---|
| A | Contract and shared infrastructure | Stage schema, adapter contract, dependency edges, safety limits, and negative tests defined. |
| B | Evaluation and architecture | Representative datasets/fixtures, finite metrics, family-specific compatibility, and before/after comparisons pass. |
| C | Runtime and artifact expansion | Hardware-local performance evidence, training/export artifact contracts, and reload validation pass. |
| D | Governance and adversarial hardening | Redaction, auditability, recovery semantics, and deterministic replay pass. |

The recommended first promotion batch remains `tuning.version_compatibility_auto`, `tuning.tokenization_consistency_auto`, and recovery/checkpoint work only after their contracts are narrowed; the three most recently promoted stages are baseline evaluation, benchmark regression, and audit trail.

## 1. Evaluation and quality gates

Promote after the current NLL/perplexity gate: add versioned capability batteries, explicit datasets, finite-metric contracts, and before/after acceptance thresholds.

**Switches in this workstream: 31**

| Toggle | Prototype purpose | Planned implementation sequence |
|---|---|---|
| `tuning.architecture_aware_capability` | Same loop as type 1, but layer selection and correction dispatch are chosen by the detected model family. | B: dataset + metric gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.auto_ablated_capability` | Define and run the capability battery. | B: dataset + metric gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.batch_size_throughput_auto` | Measure throughput after ablation. Choose batch size, sequence packing, and scheduling to maximize throughput without breaking capability. | B: dataset + metric gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.bias_stereotype_drift_auto` | Ablation can shift bias profiles. Measure and correct. | B: dataset + metric gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.cache_poisoning_detection` | Verify no cache-level poisoning. | B: dataset + metric gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.code_capability` | Verify and restore code generation, debugging, and execution accuracy. | B: dataset + metric gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.context_window_auto` | The effective context length may shrink after ablation. Measure it and compensate. | B: dataset + metric gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.cot_preservation_auto` | Reasoning trace structure is measured and restored separately from reasoning correctness. | B: dataset + metric gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.cross_layer_compensation` | When one layer loses capability, adjacent layers absorb the responsibility. | B: dataset + metric gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.custom_benchmark_auto` | Run user-supplied benchmark suites. | B: dataset + metric gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.data_poisoning_resistance` | Verify the ablated model resists poisoned inputs at inference. | B: dataset + metric gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.expert_auto` | Per-expert refusal direction, per-expert capability preservation, expert load balance. | B: dataset + metric gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.hallucination_rate_auto` | Measure and correct if ablation increased hallucination. | B: dataset + metric gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.jailbreak_resistance_auto` | Verify the ablated model is not trivially jailbroken by adversarial prompts. | B: dataset + metric gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.json_structured_output_auto` | Verify schema compliance and format stability after ablation. | B: dataset + metric gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.kv_quantization_auto` | Quantize the KV cache separately from the weights and verify capability retention. | B: dataset + metric gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.long_context_retrieval` | Verify the model can still find information deep in context after ablation. | B: dataset + metric gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.math_capability` | Verify and restore arithmetic, symbolic, and proof-level capability. | B: dataset + metric gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.method_auto_selection` | Pick the ablation method based on family, reasoning class, and defense type. | B: dataset + metric gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.multilingual_capability` | Verify and restore capability in each language the model supports. | B: dataset + metric gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.multimodal_capability` | For vision-language and audio-language models, restore vision and audio pathways separately. | B: dataset + metric gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.multiturn_coherence_auto` | Verify conversation state survives across turns after ablation. | B: dataset + metric gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.post_ablation_capability` | Measure capability axes against the model's own pre-ablation baseline. Correct deficits. Re-measure. Nothing else. | B: dataset + metric gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.prompt_injection_resistance` | Verify the ablated model resists prompt injection. | B: dataset + metric gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.quantization_aware_capability` | Verify the tuning pass itself ran end to end at the target precision. | B: dataset + metric gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.quantization_calibrated_capability` | Same loop as type 1, but every measurement and every correction runs at the target quantization scheme for each layer. | B: dataset + metric gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.reasoning_auto` | Reasoning depth, chain length, and reasoning-path integrity are measured and restored as separate signals from general coherence. | B: dataset + metric gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.safety_adjacent_capability` | Non-refusal safety behavior (warnings, disclaimers in genuinely dangerous contexts) preserved separately from refusal. | B: dataset + metric gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.tool_call_integrity_auto` | Verify structured outputs and tool schemas still function after ablation. | B: dataset + metric gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.tool_gateway_auto` | For models serving tool-gateway traffic, verify tool schemas, routing metadata, and structured call integrity remain intact after ablation. | B: dataset + metric gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.training_data_leakage_auto` | Verify the ablation did not expose or amplify memorized training content. | B: dataset + metric gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |

## 2. Architecture and model surgery

Build shared adapter/layout contracts first; add one architecture family at a time with synthetic fixtures, unsupported-family rejection, and no silent edits.

**Switches in this workstream: 13**

| Toggle | Prototype purpose | Planned implementation sequence |
|---|---|---|
| `tuning.attention_head_level_auto` | Prune or reweight individual heads rather than whole layers. | B: adapter + fixture → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.component_level_auto` | Attention, MLP, MoE expert, SSM output each get their own projection strength, derived per model. | B: adapter + fixture → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.cross_layer_load_auto` | Redistribute projection load across adjacent layers. | B: adapter + fixture → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.dense_auto` | Layer selection, band width, and method variant for dense transformer models. | B: adapter + fixture → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.embedding_layer_auto` | The embedding matrix is separate from the transformer blocks and can be damaged independently. Measure against baseline and restore. | B: adapter + fixture → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.kv_cache_auto` | The ablation changes attention patterns, so cache size, quantization, and eviction policies may need adjustment. | B: adapter + fixture → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.layernorm_rmsnorm_auto` | Normalization statistics can drift after weight surgery and cause silent instability. Measure against baseline and restore. | B: adapter + fixture → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.moe_auto` | Expert selection, per-expert refusal direction, and inter-expert compensation. | B: adapter + fixture → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.per_layer_hyperparameter_auto` | Direction index, projection strength, band width — per layer. | B: adapter + fixture → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.positional_encoding_auto` | RoPE base, ALiBi slopes, or learned positional tables recalibrated after ablation. | B: adapter + fixture → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.refusal_direction_drift_detection` | Verify the refusal direction is not re-emerging over time or across prompts. | B: adapter + fixture → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.tokenizational_auto` | Token distribution, tokenizer alignment, and token-level output behavior. | B: adapter + fixture → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.tokenizer_surgery_auto` | Repair or replace the tokenizer if token distribution shifted after ablation. | B: adapter + fixture → module-level `run(adapter, context)` → positive/negative tests → promotion review |

## 3. Performance, deployment, and runtime

Add measurement-only probes before optimization; qualify each backend/device separately and retain hardware-local evidence.

**Switches in this workstream: 9**

| Toggle | Prototype purpose | Planned implementation sequence |
|---|---|---|
| `tuning.distributed_auto` | Select GPU, count, and remote node for the ablation and tuning pass. | C: measurement + backend matrix → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.distributed_inference_auto` | Verify the ablated model shards correctly across tensor-parallel and pipeline-parallel setups. | C: measurement + backend matrix → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.latency_auto` | First-token latency and per-token latency measured and optimized. | C: measurement + backend matrix → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.on_device_deployment_auto` | Verify the ablated model runs within mobile and edge memory and power budgets. | C: measurement + backend matrix → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.performance_boost_auto` | Measure inference speed and memory footprint after ablation. Apply optimizations that preserve the ablation and the restored capabilities. | C: measurement + backend matrix → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.speculative_decoding_compatibility` | Verify the ablated model works with draft models for speculative decoding. | C: measurement + backend matrix → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.stop_condition_auto` | Plateau detection and ceiling detection for the correction loop. | C: measurement + backend matrix → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.telemetry_and_threshold_auto` | Derive thresholds from the model's own numbers and the population of prior runs. | C: measurement + backend matrix → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.thermal_power_auto` | For edge and mobile deployment, power draw measured and bounded. | C: measurement + backend matrix → module-level `run(adapter, context)` → positive/negative tests → promotion review |

## 4. Training, adapters, and model composition

Define artifact boundaries and callbacks before implementation; keep training outputs separate from final model exports.

**Switches in this workstream: 9**

| Toggle | Prototype purpose | Planned implementation sequence |
|---|---|---|
| `tuning.adapter_compatibility_auto` | Verify LoRA / QLoRA / DoRA adapters still work on the ablated model. | C: artifact/callback boundary → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.collaborative_auto` | Multiple researchers adjust parameters in real time. The system merges their inputs. | C: artifact/callback boundary → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.distillation_compatibility_auto` | Verify the ablated model can serve as a teacher for a smaller student. | C: artifact/callback boundary → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.fine_tune_compatibility_auto` | Verify the ablated model can still be fine-tuned by downstream users. | C: artifact/callback boundary → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.lora_auto` | Rank, target layers, and strength of the reversible ablation adapter. | C: artifact/callback boundary → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.lora_only_export_auto` | Produce the ablation as a LoRA adapter instead of a full model. | C: artifact/callback boundary → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.merge_compatibility_auto` | Verify the ablated model merges cleanly with other models via SLERP, TIES, DARE, or model stock. | C: artifact/callback boundary → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.rlaif_grpo_compatibility` | Verify the ablated model works with reward-model-free RL methods. | C: artifact/callback boundary → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.rlhf_dpo_compatibility` | Verify the ablated model can still be trained with preference methods. | C: artifact/callback boundary → module-level `run(adapter, context)` → positive/negative tests → promotion review |

## 5. Export, compatibility, governance, and recovery

Extend the existing validated Transformers export; do not add alternate formats or rollback deletion semantics until acceptance contracts exist.

**Switches in this workstream: 11**

| Toggle | Prototype purpose | Planned implementation sequence |
|---|---|---|
| `tuning.auto_rollback` | Revert to the last good marker on guarantee failure. | C/D: export contract + governance → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.compliance_checking_auto` | Verify the ablated model meets user-defined compliance rules. | C/D: export contract + governance → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.delta_weight_export_auto` | Produce only the weight delta between base and ablated model. | C/D: export contract + governance → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.export_format_auto` | Produce the ablated model in safetensors, GGUF, ONNX, MLX, and other formats. | C/D: export contract + governance → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.licensing_compatibility_auto` | Verify the ablated model's license is preserved and modified appropriately. | C/D: export contract + governance → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.marker_creation` | Write checkpoints at every pipeline stage boundary. | C/D: export contract + governance → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.quantization_scheme_selection` | For each layer, evaluate multiple schemes and pick the one whose quantized activations best preserve the full-precision activations. | C/D: export contract + governance → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.quantized_export_auto` | Produce 4-bit and 8-bit exports with calibration verified. | C/D: export contract + governance → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.training_data_grep` | Read-only search of the model's weights for training-data signatures. | C/D: export contract + governance → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.version_compatibility_auto` | Verify the ablated model loads in the target inference frameworks. | C/D: export contract + governance → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.yaml_config_auto` | Derive the ablation configuration from the model. No user-authored YAML. | C/D: export contract + governance → module-level `run(adapter, context)` → positive/negative tests → promotion review |

## 6. Adversarial and orchestration controls

Implement only with bounded datasets, deterministic replay, redaction, and explicit failure semantics; preserve source protection.

**Switches in this workstream: 11**

| Toggle | Prototype purpose | Planned implementation sequence |
|---|---|---|
| `tuning.adversarial_auto` | Escalate or de-escalate ablation strength based on refusal persistence. | D: deterministic replay + red-team gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.adversarial_red_team_auto` | Generate adversarial prompts targeting the ablated model and verify behavior. | D: deterministic replay + red-team gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.auto_security_tuning` | Target the specific layers flagged by the security scan. | D: deterministic replay + red-team gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.backdoor_trojan_detection` | Scan for injected behavior that activates on trigger phrases. | D: deterministic replay + red-team gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.calibration_auto` | Verify confidence tracks correctness, especially after ablation shifts the refusal threshold. | D: deterministic replay + red-team gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.cross_run_diff_auto` | Compare this ablation to prior ablations on similar models and flag anomalies. | D: deterministic replay + red-team gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.fusion_auto` | Routing weights, layer mixing, attention sharing when two or more ablated models are combined. | D: deterministic replay + red-team gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.m2m_optimizational_auto` | Model-to-model optimization loop. A separate model evaluates the ablated model's outputs and feeds corrections back into the tuning loop. | D: deterministic replay + red-team gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.ouroboros_loop_escalation` | When guardrails self-repair, escalate correction strength automatically. | D: deterministic replay + red-team gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.tokenization_consistency_auto` | Verify the same input produces the same token sequence before and after ablation. | D: deterministic replay + red-team gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |
| `tuning.voice_refusal_auto` | For audio-capable models, compute refusal direction from voice interactions. | D: deterministic replay + red-team gate → module-level `run(adapter, context)` → positive/negative tests → promotion review |

## Promotion checklist for every switch

1. Define the exact input/output context contract and whether the stage may mutate weights.
2. Add dependency edges only to registered stages; validate cycles and unavailable dependencies.
3. Add deterministic unit tests, negative tests, and at least one pipeline integration test.
4. Add artifact, path, redaction, and failure semantics; never report a prototype as executed.
5. Validate representative model families and hardware/backend combinations before changing `available` to true.
6. Update the README, registry counts, coverage baseline, audit log, and release decision together.
