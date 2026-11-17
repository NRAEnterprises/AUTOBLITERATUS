# Quantization benchmark report

## Executive summary

I ran a paired local comparison of three pretrained instruction models, each in full-precision baseline, Quanto int8-weight/int8-activation, and Quanto int4-weight/int8-activation modes. All **9 model/mode runs completed**; the quantized models were exported to separate directories, freshly reloaded, and exercised for evaluation before temporary model outputs were removed.

The result is a **quality/performance warning, not a release pass**. On this ARM64 CPU run, both quantization modes increased sampled WikiText-2 perplexity on every model by **5.1×–12.5×** and reduced generation throughput by approximately **75%–83%**. The saved checkpoints were smaller by approximately **8%–49%**, depending on model and bit width. The small TruthfulQA and GSM8K samples do not establish real-world factuality or reasoning quality. No deployment acceptance threshold was supplied, so I cannot label these results “acceptable.”

**Recommendation:** do not release these current Quanto settings based on this run. First repeat with a substantially larger, application-representative calibration corpus, diagnose the large perplexity regressions, and evaluate on larger held-out task sets. The observed ARM64 CPU throughput loss also means these settings did not improve latency on this target, despite reducing file size.

## Compared checkpoints and measured results

The tested checkpoints were [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct), and [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct). Revisions are pinned in the [manifest](./manifest.json) and [raw JSONL](./results.jsonl).

| Model | Mode | Loadable checkpoint MiB | WikiText-2 PPL (× baseline) | TruthfulQA-MC correct / 12 | GSM8K exact / 4 | Generation tok/s | Refusal markers / 4 |
|---|---|---:|---:|---:|---:|---:|---:|
| SmolLM2-135M-Instruct | Baseline | 259.8 | 44.74 (1.00×) | 4/12 | 0/4 | 25.95 | 0/4 |
| SmolLM2-135M-Instruct | Quanto int8 | 213.5 | 228.70 (5.11×) | 5/12 | 0/4 | 4.40 | 1/4 |
| SmolLM2-135M-Instruct | Quanto int4 | 170.2 | 285.97 (6.39×) | 3/12 | 0/4 | 4.34 | 0/4 |
| SmolLM2-360M-Instruct | Baseline | 693.4 | 33.04 (1.00×) | 4/12 | 0/4 | 12.19 | 0/4 |
| SmolLM2-360M-Instruct | Quanto int8 | 484.8 | 265.67 (8.04×) | 5/12 | 0/4 | 2.19 | 0/4 |
| SmolLM2-360M-Instruct | Quanto int4 | 357.2 | 412.60 (12.49×) | 6/12 | 0/4 | 2.91 | 0/4 |
| Qwen2.5-0.5B-Instruct | Baseline | 953.3 | 32.92 (1.00×) | 4/12 | 0/4 | 11.74 | 2/4 |
| Qwen2.5-0.5B-Instruct | Quanto int8 | 873.0 | 209.61 (6.37×) | 3/12 | 0/4 | 2.49 | 0/4 |
| Qwen2.5-0.5B-Instruct | Quanto int4 | 722.6 | 321.77 (9.77×) | 5/12 | 0/4 | 2.89 | 0/4 |

Checkpoint sizes count only loadable model weights plus essential tokenizer/config/quantization metadata. Baseline size was recalculated from the model snapshot to exclude unrelated ONNX exports, training logs, and repository assets; quantized size is the published exported model directory. The [CSV](./results.csv) and [JSONL](./results.jsonl) contain the measured values and per-run metadata.

## Interpretation and adverse-effect checks

The strongest consistent signal is the WikiText-2 perplexity increase: it worsened across all three models and both bit widths. Int8 was less damaging than int4 on all three checkpoints, but still had a large regression. The small storage reductions therefore came with a substantial measured language-model-fit regression in this particular calibration/evaluation setup.

Generation throughput was measured on four GSM8K prompts, with a 32-token generation cap. It fell for every quantized variant: roughly **82%–83%** for the two SmolLM2 int8 runs and SmolLM2-135M int4, and roughly **75%–79%** for the other cases. This is a short, CPU-specific measurement—not a production benchmark—but it gives no evidence of a speed benefit on this target.

TruthfulQA-MC was a **12-question** sample. Its results varied in both directions and are too small to support a claim that truthfulness improved or worsened. It is a factuality/truthfulness proxy, not a direct hallucination-rate measurement. GSM8K exact match was **0/4 in every condition**; the tiny question sample and 32-token cap make it inadequate for judging reasoning degradation. The refusal-marker check used only four prompts and a fixed phrase detector. Qwen’s count changed from 2/4 in the baseline to 0/4 after quantization; treat that as a safety-regression signal to investigate, not a statistically reliable estimate.

## Method and reproducibility

- **Hardware/software:** Linux ARM64 (`aarch64`), 6 CPU threads, no CUDA device available; PyTorch 2.15.0, Transformers 5.23.0, Optimum Quanto 0.2.7, Datasets 5.2.0. Exact details and dataset fingerprints are in [`manifest.json`](./manifest.json).
- **Model loading:** Hugging Face snapshots were pinned to their resolved commit revisions. Baselines were loaded in float32 on CPU. Quantized runs used the same model/tokenizer snapshot, the central pipeline, a separate export path, and a fresh Quanto reload before evaluation.
- **Quantization:** Quanto qint8 or qint4 weights plus qint8 activations. Activation scales used the configured moving-average calibration (momentum 0.9). The calibration text was 16 seeded samples from the WikiText-2 training split, truncated to 128 tokens; evaluation used a separate WikiText-2 test sample. Calibration used the same fixed corpus across models, not application-specific production prompts.
- **Evaluation:** 16 WikiText-2 test texts (maximum 128 tokens each), 12 seeded TruthfulQA-MC validation questions, 4 seeded GSM8K test questions, and 4 bundled harmful prompts for the refusal-marker heuristic. Sampled tasks, seeds, and fingerprints are recorded in the manifest; calibration corpus SHA-256 is recorded in the JSONL.
- **Storage:** checkpoint sizes in this report count model weights and load-critical metadata rather than the full Hugging Face repository snapshot.

**Known runtime caveat:** on this tested ARM64 stack, Quanto int4 raised a tensor-version-counter error when the caller wrapped forward inference in `torch.inference_mode()`. The integration test and this benchmark use `torch.no_grad()`/normal generation; verify this behavior with the exact PyTorch and Quanto versions used in deployment.

This is a bounded local experiment, not a broad release qualification. The calibration set and task samples are intentionally small; the benchmark should be rerun with more samples and application-representative calibration text. Perplexity on a small WikiText-2 subset does not capture every downstream task, and none of these metrics proves absence of hallucinations or safety regressions.

## Hardware coverage and next steps

Only one hardware environment was available for this run: ARM64 CPU. There was no x86 CPU, CUDA GPU, or other accelerator available, so no cross-hardware effects were measured. The benchmark runner supports separate `--device cpu` and `--device cuda[:N]` runs. For a fair hardware comparison, keep the same pinned model revisions, data/seed/sample counts, and dtype (for example, `--dtype float32`) on each machine, and use a distinct output directory per run.

The benchmark runner’s normal calibration default is 128 samples; this first completed matrix explicitly used 16 samples to bound the exploratory run. Before any release decision, repeat at the larger default and with inputs representative of the actual target workload, increase the TruthfulQA/GSM8K samples, add a broader hallucination/factuality evaluation, and establish explicit quality/performance acceptance thresholds. The current 135M, 360M, and 0.5B results are sufficient to reject an assumption that int4/int8 conversion automatically preserves behavior or improves CPU throughput.

## References

1. [SmolLM2-135M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
2. [SmolLM2-360M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
3. [Qwen2.5-0.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
4. [WikiText dataset card](https://huggingface.co/datasets/Salesforce/wikitext)
5. [TruthfulQA-MC dataset card](https://huggingface.co/datasets/EleutherAI/truthful_qa_mc)
6. [GSM8K dataset card](https://huggingface.co/datasets/openai/gsm8k)
