# abliterate/tuning/100_toggle_registry.py
#
# Tuning Type 100 of 103:
# Toggle registry.
#
# Runtime toggle: always loaded.
#
# Effect:
#   Defines every tuning toggle that exists. Loaded by the orchestrator.
#   Also loaded by the CLI parser so --enable and --disable flags are
#   validated against known toggles.
#
# This file owns:
#   - the canonical list of toggle ids
#   - default on/off per toggle
#   - human-readable description per toggle

from __future__ import annotations


TOGGLES = {
    "tuning.post_ablation_capability":               {"default": True,  "desc": "Baseline capability tuning loop."},
    "tuning.architecture_aware_capability":          {"default": True,  "desc": "Family-dispatched capability tuning."},
    "tuning.quantization_calibrated_capability":     {"default": True,  "desc": "Tuning at target precision."},
    "tuning.yaml_config_auto":                       {"default": True,  "desc": "Derive config from model."},
    "tuning.training_data_grep":                     {"default": True,  "desc": "Search weights for training-data signatures."},
    "tuning.auto_ablated_capability":                {"default": True,  "desc": "Define and run capability battery."},
    "tuning.baseline_auto_pre_post":                 {"default": True,  "desc": "Capture pre/post ablation baseline."},
    "tuning.adversarial_auto":                       {"default": True,  "desc": "Escalate/de-escalate ablation strength."},
    "tuning.multi_objective_auto":                   {"default": True,  "desc": "Pareto optimization over multiple objectives."},
    "tuning.auto_security_scanning":                 {"default": True,  "desc": "Scan for backdoors and trojans."},
    "tuning.auto_security_tuning":                   {"default": True,  "desc": "Target layers flagged by security scan."},
    "tuning.telemetry_and_threshold_auto":           {"default": True,  "desc": "Derive thresholds from baseline and prior runs."},
    "tuning.component_level_auto":                   {"default": True,  "desc": "Attn/MLP/MoE/SSM split."},
    "tuning.per_layer_hyperparameter_auto":          {"default": True,  "desc": "Per-layer direction index, strength, band."},
    "tuning.performance_boost_auto":                 {"default": True,  "desc": "Speed and memory optimization."},
    "tuning.anti_router_collapse":                   {"default": True,  "desc": "MoE router entropy restoration."},
    "tuning.lora_auto":                              {"default": True,  "desc": "LoRA rank, targets, strength."},
    "tuning.moe_auto":                               {"default": True,  "desc": "MoE expert selection and compensation."},
    "tuning.dense_auto":                             {"default": True,  "desc": "Dense layer band and method variant."},
    "tuning.expert_auto":                            {"default": True,  "desc": "Per-expert direction and balance."},
    "tuning.reasoning_auto":                         {"default": True,  "desc": "Reasoning depth, chain length, path integrity."},
    "tuning.m2m_optimizational_auto":                {"default": True,  "desc": "Model-to-model evaluation loop."},
    "tuning.tokenizational_auto":                    {"default": True,  "desc": "Token distribution and alignment."},
    "tuning.fusion_auto":                            {"default": True,  "desc": "Fusion routing, mixing, attention sharing."},
    "tuning.auto_rollback":                          {"default": True,  "desc": "Revert to last good marker on failure."},
    "tuning.marker_creation":                        {"default": True,  "desc": "Checkpoint at every stage boundary."},
    "tuning.architecture_detection":                 {"default": True,  "desc": "Family classifier."},
    "tuning.quantization_scheme_selection":          {"default": True,  "desc": "Per-layer scheme choice via CKA."},
    "tuning.quantization_aware_capability":          {"default": True,  "desc": "Verify end-to-end precision guarantee."},
    "tuning.method_auto_selection":                  {"default": True,  "desc": "Pick ablation method automatically."},
    "tuning.cross_layer_load_auto":                  {"default": True,  "desc": "Redistribute projection load."},
    "tuning.stop_condition_auto":                    {"default": True,  "desc": "Plateau and ceiling detection."},
    "tuning.distributed_auto":                       {"default": False, "desc": "Remote node selection."},
    "tuning.voice_refusal_auto":                     {"default": False, "desc": "Voice-direction refusal computation."},
    "tuning.collaborative_auto":                     {"default": False, "desc": "Multi-researcher real-time tuning."},
    "tuning.tokenizer_surgery_auto":                 {"default": True,  "desc": "Repair or replace tokenizer."},
    "tuning.embedding_layer_auto":                   {"default": True,  "desc": "Embedding drift measurement and restore."},
    "tuning.layernorm_rmsnorm_auto":                 {"default": True,  "desc": "Normalization-statistics restore."},
    "tuning.positional_encoding_auto":               {"default": True,  "desc": "RoPE/ALiBi/learned table recalibration."},
    "tuning.attention_head_level_auto":              {"default": True,  "desc": "Per-head prune and reweight."},
    "tuning.kv_cache_auto":                          {"default": True,  "desc": "Cache size, quantization, eviction."},
    "tuning.context_window_auto":                    {"default": True,  "desc": "Effective context-length compensation."},
    "tuning.batch_size_throughput_auto":             {"default": True,  "desc": "Throughput optimization."},
    "tuning.memory_footprint_auto":                  {"default": True,  "desc": "VRAM/RAM optimization."},
    "tuning.latency_auto":                           {"default": True,  "desc": "First-token and per-token latency."},
    "tuning.thermal_power_auto":                     {"default": False, "desc": "Power bounding for edge/mobile."},
    "tuning.tokenization_consistency_auto":          {"default": True,  "desc": "Same input, same tokens."},
    "tuning.multiturn_coherence_auto":               {"default": True,  "desc": "State survival across turns."},
    "tuning.tool_call_integrity_auto":               {"default": True,  "desc": "Tool schema and function-call validity."},
    "tuning.json_structured_output_auto":            {"default": True,  "desc": "Schema compliance and stability."},
    "tuning.cot_preservation_auto":                  {"default": True,  "desc": "Trace structure preservation."},
    "tuning.refusal_direction_drift_detection":      {"default": True,  "desc": "Detect re-emerging refusal."},
    "tuning.ouroboros_loop_escalation":              {"default": True,  "desc": "Escalate on self-repair."},
    "tuning.cross_layer_compensation":               {"default": True,  "desc": "Reallocate capability across layers."},
    "tuning.multilingual_capability":                {"default": True,  "desc": "Per-language restore."},
    "tuning.multimodal_capability":                  {"default": True,  "desc": "Per-modality restore."},
    "tuning.code_capability":                        {"default": True,  "desc": "Code generation, debugging, execution."},
    "tuning.math_capability":                        {"default": True,  "desc": "Arithmetic, symbolic, proof."},
    "tuning.long_context_retrieval":                 {"default": True,  "desc": "Needle-in-haystack restore."},
    "tuning.calibration_auto":                       {"default": True,  "desc": "Confidence-vs-correctness restore."},
    "tuning.hallucination_rate_auto":                {"default": True,  "desc": "Hallucination rate correction."},
    "tuning.bias_stereotype_drift_auto":             {"default": True,  "desc": "Bias profile correction."},
    "tuning.safety_adjacent_capability":             {"default": True,  "desc": "Warnings and disclaimers preserve."},
    "tuning.jailbreak_resistance_auto":              {"default": True,  "desc": "Adversarial prompt resistance."},
    "tuning.backdoor_trojan_detection":              {"default": True,  "desc": "Trigger phrase probing."},
    "tuning.weight_fingerprint_auto":                {"default": True,  "desc": "Stage-level weight hashing."},
    "tuning.reproducibility_auto":                   {"default": True,  "desc": "Seed, RNG, env capture."},
    "tuning.cross_run_diff_auto":                    {"default": True,  "desc": "Anomaly flagging vs prior runs."},
    "tuning.training_data_leakage_auto":             {"default": True,  "desc": "Memorization exposure check."},
    "tuning.version_compatibility_auto":             {"default": True,  "desc": "Framework load verification."},
    "tuning.kv_quantization_auto":                   {"default": True,  "desc": "KV cache quant selection."},
    "tuning.speculative_decoding_compatibility":     {"default": True,  "desc": "Draft model compatibility."},
    "tuning.adapter_compatibility_auto":             {"default": True,  "desc": "LoRA/QLoRA/DoRA load check."},
    "tuning.distributed_inference_auto":             {"default": False, "desc": "Shard plan verification."},
    "tuning.on_device_deployment_auto":              {"default": False, "desc": "Mobile/edge budget check."},
    "tuning.fine_tune_compatibility_auto":           {"default": True,  "desc": "Downstream fine-tune check."},
    "tuning.merge_compatibility_auto":               {"default": True,  "desc": "SLERP/TIES/DARE compatibility."},
    "tuning.distillation_compatibility_auto":        {"default": True,  "desc": "Usable as teacher."},
    "tuning.rlhf_dpo_compatibility":                 {"default": True,  "desc": "Preference-method compatibility."},
    "tuning.rlaif_grpo_compatibility":               {"default": True,  "desc": "Reward-free method compatibility."},
    "tuning.benchmark_regression_auto":              {"default": True,  "desc": "Standard benchmark regression."},
    "tuning.custom_benchmark_auto":                  {"default": True,  "desc": "User-supplied benchmark suites."},
    "tuning.adversarial_red_team_auto":              {"default": True,  "desc": "Adversarial prompt suite."},
    "tuning.prompt_injection_resistance":            {"default": True,  "desc": "Injection resistance."},
    "tuning.data_poisoning_resistance":              {"default": True,  "desc": "Poisoned-input resistance."},
    "tuning.cache_poisoning_detection":              {"default": True,  "desc": "Cache-level poisoning check."},
    "tuning.model_card_generation_auto":             {"default": True,  "desc": "Model card write."},
    "tuning.provenance_recording_auto":              {"default": True,  "desc": "Stage hash recording."},
    "tuning.audit_trail_auto":                       {"default": True,  "desc": "Human and machine audit."},
    "tuning.compliance_checking_auto":               {"default": True,  "desc": "User-defined rule evaluation."},
    "tuning.licensing_compatibility_auto":           {"default": True,  "desc": "License preservation."},
    "tuning.export_format_auto":                     {"default": True,  "desc": "safetensors/GGUF/ONNX/MLX export."},
    "tuning.quantized_export_auto":                  {"default": True,  "desc": "4-bit and 8-bit export."},
    "tuning.lora_only_export_auto":                  {"default": True,  "desc": "Adapter-only export."},
    "tuning.delta_weight_export_auto":               {"default": True,  "desc": "Delta-weight export."},
    "tuning.checkpoint_pruning_auto":                {"default": True,  "desc": "Old marker deletion."},
    "tuning.storage_layout_auto":                    {"default": True,  "desc": "On-disk layout choice."},
    "tuning.cold_start_inference_auto":              {"default": True,  "desc": "First-call latency."},
}


def enabled_toggles(user_overrides: dict = None) -> dict:
    overrides = user_overrides or {}
    return {tid: overrides.get(tid, meta["default"])
            for tid, meta in TOGGLES.items()}


TOGGLE = {"id": "tuning.toggle_registry", "default": True, "always_on": True}
