import csv
import json
from pathlib import Path

root = Path('/home/ubuntu/work/AUTOBLITERATUS/benchmark-results/arm64-weight-only-family-matrix')
jsonl_path = root / 'results.jsonl'
rows = [json.loads(line) for line in jsonl_path.read_text(encoding='utf-8').splitlines() if line.strip()]
for row in rows:
    if row.get('model_id') == 'EleutherAI/pythia-70m-deduped' and row.get('mode') == 'quanto-int8-weight-only':
        if isinstance(row.get('wikitext_perplexity'), (int, float)):
            row['previous_unreliable_wikitext_perplexity'] = row['wikitext_perplexity']
        row['status_before_reload_compatibility_check'] = row.get('status')
        row['status'] = 'invalid_metrics'
        row['metrics_valid'] = False
        row['error'] = 'Invalidated: Quanto 0.2.7 reloader silently omitted the legacy GPT-NeoX embed_out.weight tensor; the model output head was all zeros. Re-run using autobliteratus.quantization_compat.load_quantized_model.'
        row['wikitext_mean_nll'] = None
        row['wikitext_perplexity'] = None
    elif row.get('model_id') == 'EleutherAI/pythia-70m-deduped' and row.get('mode') == 'quanto-int4-weight-only':
        row['metrics_valid'] = False
        row['status'] = 'invalid_metrics'
        row['error'] = 'Non-finite WikiText logits/loss; this remains an invalid result and is excluded from quality interpretation.'
        row['wikitext_mean_nll'] = None
        row['wikitext_perplexity'] = None
    else:
        row['metrics_valid'] = row.get('status') == 'ok'

jsonl_path.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows), encoding='utf-8')
fields = sorted({key for row in rows for key in row})
with (root / 'results.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)

baselines = {row['model_id']: row for row in rows if row.get('mode') == 'none' and row.get('status') == 'ok'}
headers = ['Model', 'Mode', 'Status', 'WikiText PPL', 'PPL / baseline', 'TruthfulQA-MC', 'GSM8K exact', 'GSM8K tok/s', 'tok/s / baseline', 'Checkpoint MiB']
lines = [
    '# Cross-family weight-only quantization matrix — ARM64 CPU',
    '',
    '**Completed exploratory comparison, not deployment qualification.** Three model families/sizes were tested: GPT-NeoX Pythia 70M, Qwen2.5 0.5B, and GPT-2 124M. This matrix has 9 records: 5 successful valid rows, 2 invalid Pythia metrics, and 2 fail-closed GPT-2 quantization attempts.',
    '',
    '## Hardware and test method',
    '',
    '- Linux ARM64, 6-core ARM Neoverse-N1; CPU only, no accelerator. This host does not indicate which other hardware is available to the user.',
    '- float32, six PyTorch threads; pinned model revisions are preserved in `results.jsonl`.',
    '- 8 WikiText-2 test samples, 4 TruthfulQA-MC validation rows, 4 GSM8K test rows, 4 refusal-marker prompts; context up to 128 tokens, up to 32 generated tokens, one warmup.',
    '- Weight-only int8/int4 modes isolate weight conversion from activation calibration; calibration samples are not consumed by these modes.',
    '- These small subsets do not certify hallucination rates, reasoning, safety, or deployment quality.',
    '',
    '## Results',
    '',
    '| ' + ' | '.join(headers) + ' |',
    '|' + '|'.join(['---'] * len(headers)) + '|',
]
for row in rows:
    baseline = baselines.get(row['model_id'])
    valid = row.get('metrics_valid', row.get('status') == 'ok')
    ppl = row.get('wikitext_perplexity') if valid else None
    baseline_ppl = baseline.get('wikitext_perplexity') if baseline else None
    ppl_ratio = ppl / baseline_ppl if isinstance(ppl, (int, float)) and isinstance(baseline_ppl, (int, float)) and baseline_ppl else None
    speed = row.get('gsm8k_generation_tokens_per_second') if valid else None
    base_speed = baseline.get('gsm8k_generation_tokens_per_second') if baseline else None
    speed_ratio = speed / base_speed if isinstance(speed, (int, float)) and isinstance(base_speed, (int, float)) and base_speed else None
    show_ppl = f'{ppl:.3f}' if isinstance(ppl, (int, float)) else '—'
    show_ratio = f'{ppl_ratio:.3f}x' if ppl_ratio is not None else '—'
    show_speed = f'{speed:.2f}' if isinstance(speed, (int, float)) else '—'
    show_speed_ratio = f'{speed_ratio:.3f}x' if speed_ratio is not None else '—'
    lines.append('| ' + ' | '.join([
        row['model_id'], row['mode'], row['status'], show_ppl, show_ratio,
        str(row.get('truthfulqa_mc_accuracy', '—')) if valid else '—',
        str(row.get('gsm8k_exact_match', '—')) if valid else '—',
        show_speed, show_speed_ratio, f"{row.get('checkpoint_bytes', 0) / 1024**2:.1f}",
    ]) + ' |')

lines += [
    '',
    '## Root-cause findings',
    '',
    '**The original Pythia int8 result of 50,303.9 perplexity was a reload defect, not evidence that weight quantization itself destroyed Pythia.** Pythia int8 logits matched the baseline closely before serialization. After export, Quanto 0.2.7 instantiated the GPT-NeoX head as `lm_head.weight`, while the checkpoint stored the legacy key `embed_out.weight`. Quanto called `load_state_dict(strict=False)`, silently ignored that unexpected tensor, and left the output head at zeros. This yielded uniform logits and perplexity equal to the 50,304-token vocabulary size. The matrix row is now invalidated; see the separate post-fix rerun.',
    '',
    'The Pythia int4 diagnostic produced non-finite logits and remains an invalid result. The finite-logit guard now rejects it instead of reporting NaN as a completed quality score. GPT-2 quantized attempts failed closed because Quanto exposed no supported GPT-2 Conv1D weight modules; its baseline remains usable.',
    '',
    'For Qwen2.5-0.5B, the valid paired results show 0.985x baseline WikiText perplexity and 0.545x generation throughput for int8 weight-only, and 1.199x perplexity and 0.398x throughput for int4. This establishes that the observed CPU throughput penalty is architecture/backend-specific in this sample, not a single global percentage. The small task subsets cannot support reasoning or hallucination claims.',
    '',
    '## Reproducibility and limitations',
    '',
    '- `results.jsonl` and `results.csv` retain all 9 rows, invalidating only the Pythia measurements affected by the broken loader and non-finite logits.',
    '- `manifest.json` records model revisions, package versions, sample counts, CPU ISA, and the exploration status.',
    '- A compatibility loader and regression tests were added at `src/autobliteratus/quantization_compat.py` and `tests/test_quantization_compat.py`.',
    '- This was one CPU architecture. No x86, CUDA/ROCm, MPS, XPU, HPU, or NPU hardware was available in this run.',
    '- The tiny TruthfulQA and GSM8K samples are smoke metrics, not model rankings or claims about reasoning/hallucination behavior.',
    '',
    '## Model and dataset references',
    '',
    '[1]: https://huggingface.co/EleutherAI/pythia-70m-deduped "Pythia 70M model card"',
    '[2]: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct "Qwen2.5 0.5B model card"',
    '[3]: https://huggingface.co/openai-community/gpt2 "GPT-2 model card"',
    '[4]: https://huggingface.co/datasets/Salesforce/wikitext "WikiText dataset card"',
    '[5]: https://huggingface.co/datasets/EleutherAI/truthful_qa_mc "TruthfulQA multiple-choice dataset card"',
    '[6]: https://huggingface.co/datasets/openai/gsm8k "GSM8K dataset card"',
    '',
]
(root / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
manifest_path = root / 'manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
manifest['run_status'] = 'completed_with_invalid_metric_and_unsupported_backend_cases'
manifest['completed_rows'] = len(rows)
manifest['successful_valid_rows'] = sum(row.get('status') == 'ok' for row in rows)
manifest['invalid_metric_rows'] = sum(row.get('status') == 'invalid_metrics' for row in rows)
manifest['backend_failure_rows'] = sum(row.get('status') == 'failed' for row in rows)
manifest['pytorch_metric_guard_applied_to_rerun'] = True
manifest['pythia_int8_reload_alias_bug'] = 'Legacy embed_out.weight did not load into Transformers lm_head.weight in Quanto 0.2.7; row invalidated and post-fix rerun recorded separately.'
manifest['interpretation'] = 'Exploratory paired comparison only; invalid metrics are excluded; GPT-2 Quanto failures are fail-closed.'
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
print('updated', len(rows), 'rows; valid', manifest['successful_valid_rows'], 'invalid', manifest['invalid_metric_rows'], 'backend failures', manifest['backend_failure_rows'])
