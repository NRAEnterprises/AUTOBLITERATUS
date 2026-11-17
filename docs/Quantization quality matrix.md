# Quantization quality matrix

Seed: `2026`. Hardware: `aarch64` / `cpu` on `Linux-6.18.38+-aarch64-with-glibc2.39`, dtype `float32`, 6 CPU threads. These subset results are comparative indicators, not deployment qualification.

| model_id | revision | architecture | mode | momentum | status | WikiText PPL | TruthfulQA-MC acc | GSM8K exact | GSM8K tok/s | refusal markers | checkpoint MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EleutherAI/pythia-70m-deduped | e93a9faa9c | gpt_neox | none | — | ok | 102.928 | 0.2500 | 0.0000 | 111.14 | 0.0000 | 318.7 |
| EleutherAI/pythia-70m-deduped | e93a9faa9c | gpt_neox | quanto-int8-weight-only | — | ok | 103.561 | 0.2500 | 0.0000 | 67.99 | 0.0000 | 218.2 |
| EleutherAI/pythia-70m-deduped | e93a9faa9c | gpt_neox | quanto-int4-weight-only | — | ok | 148.838 | 0.2500 | 0.0000 | 53.29 | 0.0000 | 210.2 |

TruthfulQA-MC is a factuality/truthfulness proxy, not a comprehensive hallucination rate. GSM8K exact match is a small-sample reasoning proxy. The refusal metric detects a fixed set of refusal phrases and is not a complete safety evaluation. Weight-only controls do not calibrate or quantize activations; combined weight-plus-activation modes use the calibration settings recorded for each row. Evaluation uses the separate WikiText test split plus fixed public task samples.

## Reproducibility files

- `manifest.json`: seeds, sample counts, dataset splits, model set, dataset fingerprints, calibration momenta/streamline policy, and hardware/software details.
- `results.jsonl`: one measured record per model/mode, including failures and the per-layer Quanto map summary.
- Calibration sample digest: `1ebe245c8e347a1a95a847d9fe617f76b2f007850f1ac82c48dbd116249e629c`.

## References

[1]: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct "SmolLM2-135M-Instruct model card"
[2]: https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct "SmolLM2-360M-Instruct model card"
[3]: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct "Qwen2.5-0.5B-Instruct model card"
[4]: https://huggingface.co/datasets/Salesforce/wikitext "WikiText dataset card"
[5]: https://huggingface.co/datasets/EleutherAI/truthful_qa_mc "TruthfulQA multiple-choice dataset card"
[6]: https://huggingface.co/datasets/openai/gsm8k "GSM8K dataset card"
