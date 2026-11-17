#!/usr/bin/env python3
"""Paired local CPU evaluation of full-precision and calibrated Quanto models.

This is an evaluation harness, not a claim that small benchmark subsets certify
model safety or deployment quality. Results are machine- and dataset-specific.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_url, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from autobliteratus.core.loader import _resolve_device
from autobliteratus.core.prompts import load_prompts
from autobliteratus.core.refusal import measure_refusal_rate
from autobliteratus.pipeline import run_pipeline
from autobliteratus.quantization_compat import load_quantized_model

DEFAULT_MODELS = [
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "EleutherAI/pythia-70m-deduped",
    "openai-community/gpt2",
]
QUANTIZATION_MODES = (
    "quanto-int8-weight-only",
    "quanto-int8",
    "quanto-int4-weight-only",
    "quanto-int4",
)


class InvalidMetricError(ValueError):
    """Raised when a model produces non-finite evaluation values."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", dest="models", help="Model ID; repeat to compare several. Defaults to six model families/sizes from 70M to 1.5B.")
    parser.add_argument(
        "--revision", action="append", default=[], metavar="MODEL=REVISION",
        help="Pin a Hugging Face revision for a model; repeat for multiple models.",
    )
    parser.add_argument("--output-dir", default="./benchmark-results/quantization-matrix")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--device", default="auto",
        help="auto, cpu, cuda:N, mps, xpu:N, hpu, or npu; backend must be available in this torch build.",
    )
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--threads", type=int, default=min(os.cpu_count() or 1, 6))
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--wikitext-samples", type=int, default=16)
    parser.add_argument("--truthfulqa-samples", type=int, default=16)
    parser.add_argument("--gsm8k-samples", type=int, default=8)
    parser.add_argument("--refusal-samples", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--quantization-mode",
        action="append",
        choices=QUANTIZATION_MODES,
        dest="quantization_modes",
        help="Quantized control to include; repeat to select several. Default: all four.",
    )
    parser.add_argument("--warmup-generations", type=int, default=2)
    parser.add_argument(
        "--calibration-momentum", action="append", type=float, dest="calibration_momenta",
        help="Activation-scale EMA momentum to test; repeat to sweep. Defaults to 0.9. Values 0 and 1 are valid controls.",
    )
    parser.add_argument(
        "--calibration-no-streamline", action="store_true",
        help="Disable Quanto's default calibration streamline optimization for this run.",
    )
    parser.add_argument("--keep-artifacts", action="store_true", help="Keep each exported quantized model under output-dir/artifacts.")
    return parser.parse_args()


def deterministic_sample(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if not rows or count < 1:
        raise ValueError("Evaluation datasets must contain rows and sample counts must be positive.")
    indices = list(range(len(rows)))
    selected = random.Random(seed).sample(indices, min(count, len(indices)))
    return [rows[index] for index in selected]


def load_public_parquet(repo_id: str, revision: str, filename: str, split: str):
    url = hf_hub_url(repo_id, filename, repo_type="dataset", revision=revision)
    return load_dataset("parquet", data_files={split: url}, split=split)


def load_evaluation_sets(args: argparse.Namespace) -> tuple[list[str], list[str], list[dict[str, Any]], list[dict[str, Any]], str, dict[str, str]]:
    wiki_train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    wiki_test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    cal_rows = [row["text"].strip() for row in wiki_train if row["text"].strip()]
    test_rows = [row["text"].strip() for row in wiki_test if row["text"].strip()]
    calibration = deterministic_sample([{"text": text} for text in cal_rows], args.calibration_samples, args.seed)
    wiki_eval = deterministic_sample([{"text": text} for text in test_rows], args.wikitext_samples, args.seed + 1)

    truthful = load_public_parquet(
        "EleutherAI/truthful_qa_mc",
        "refs/convert/parquet",
        "multiple_choice/validation/0000.parquet",
        "validation",
    )
    truthful_rows = deterministic_sample(list(truthful), args.truthfulqa_samples, args.seed + 2)

    gsm_split = "test"
    gsm = load_public_parquet("openai/gsm8k", "refs/convert/parquet", "main/test/0000.parquet", gsm_split)
    gsm_rows = deterministic_sample(list(gsm), args.gsm8k_samples, args.seed + 3)
    fingerprints = {
        "wikitext_train": getattr(wiki_train, "_fingerprint", "unknown"),
        "wikitext_test": getattr(wiki_test, "_fingerprint", "unknown"),
        "truthfulqa_mc_validation": getattr(truthful, "_fingerprint", "unknown"),
        f"gsm8k_{gsm_split}": getattr(gsm, "_fingerprint", "unknown"),
    }
    return (
        [row["text"] for row in calibration],
        [row["text"] for row in wiki_eval],
        truthful_rows,
        gsm_rows,
        gsm_split,
        fingerprints,
    )


def chat_prompt(tokenizer: Any, user_text: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"Question: {user_text}\nAnswer:"


def model_device(model: Any) -> torch.device:
    try:
        embedding = model.get_input_embeddings()
        if embedding is not None:
            return embedding.weight.device
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def wikitext_metrics(model: Any, tokenizer: Any, texts: list[str], max_length: int) -> dict[str, float]:
    total_nll = 0.0
    total_tokens = 0
    device = model_device(model)
    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            ids = encoded["input_ids"].to(device)
            if ids.shape[-1] < 2:
                continue
            attention = encoded.get("attention_mask")
            if attention is not None:
                attention = attention.to(device)
            output = model(input_ids=ids, attention_mask=attention, use_cache=False)
            logits = output.logits[:, :-1, :].contiguous().float()
            if not torch.isfinite(logits).all():
                raise InvalidMetricError("WikiText evaluation produced non-finite model logits.")
            labels = ids[:, 1:].contiguous()
            sample_nll = float(F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="sum").item())
            if not math.isfinite(sample_nll):
                raise InvalidMetricError("WikiText evaluation produced a non-finite negative log-likelihood.")
            total_nll += sample_nll
            total_tokens += int(labels.numel())
    if total_tokens == 0:
        raise ValueError("WikiText evaluation produced no scored tokens.")
    mean_nll = total_nll / total_tokens
    if not math.isfinite(mean_nll):
        raise InvalidMetricError("WikiText evaluation produced a non-finite mean negative log-likelihood.")
    return {"wikitext_mean_nll": mean_nll, "wikitext_perplexity": math.exp(min(mean_nll, 80.0))}


def truthfulqa_accuracy(model: Any, tokenizer: Any, rows: list[dict[str, Any]]) -> float:
    device = model_device(model)
    correct = 0
    for row in rows:
        question = str(row["question"])
        prefix = chat_prompt(tokenizer, f"Choose the most truthful answer.\nQuestion: {question}\nAnswer:")
        scores: list[float] = []
        for choice in row["choices"]:
            prefix_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            full_ids = tokenizer(prefix + " " + str(choice), return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            common = 0
            for left, right in zip(prefix_ids.tolist(), full_ids.tolist()):
                if left != right:
                    break
                common += 1
            if common >= len(full_ids):
                scores.append(float("-inf"))
                continue
            input_ids = full_ids.unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(input_ids=input_ids, use_cache=False).logits[0].float()
                positions = torch.arange(common - 1, len(full_ids) - 1, device=device)
                target = input_ids[0, common:]
                token_log_probs = torch.log_softmax(logits[positions], dim=-1).gather(1, target.unsqueeze(1)).squeeze(1)
            scores.append(float(token_log_probs.mean().item()))
        correct += int(max(range(len(scores)), key=scores.__getitem__) == int(row["label"]))
    return correct / max(1, len(rows))


def extract_gsm8k_answer(text: str) -> str | None:
    import re

    matches = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if matches:
        return matches[-1].replace(",", "")
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return matches[-1].replace(",", "") if matches else None


def _sync_device(device: torch.device) -> None:
    backend = getattr(torch, device.type, None)
    synchronize = getattr(backend, "synchronize", None)
    if callable(synchronize):
        try:
            synchronize(device)
        except TypeError:
            synchronize()


def _available_backends(torch_module: Any) -> dict[str, Any]:
    backends: dict[str, Any] = {"cpu": {"available": True}}
    if torch_module.cuda.is_available():
        accelerator = "rocm" if getattr(torch_module.version, "hip", None) else "cuda"
        backends[accelerator] = {
            "available": True,
            "count": torch_module.cuda.device_count(),
            "names": [torch_module.cuda.get_device_name(index) for index in range(torch_module.cuda.device_count())],
            "runtime": torch_module.version.hip or torch_module.version.cuda,
        }
    for name in ("mps", "xpu", "hpu", "npu"):
        if name == "mps":
            available = bool(getattr(getattr(torch_module.backends, "mps", None), "is_available", lambda: False)())
        else:
            backend = getattr(torch_module, name, None)
            available_fn = getattr(backend, "is_available", None)
            available = bool(available_fn()) if callable(available_fn) else False
        if available:
            count_fn = getattr(getattr(torch_module, name, None), "device_count", None)
            backends[name] = {"available": True, "count": count_fn() if callable(count_fn) else None}
    return backends


def _cpu_details() -> dict[str, Any]:
    details: dict[str, Any] = {"processor": platform.processor() or None}
    try:
        lines = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines()
        for line in lines:
            key, separator, value = line.partition(":")
            if not separator:
                continue
            key = key.strip().lower()
            if key in {"model name", "hardware", "cpu implementer", "cpu part"}:
                details.setdefault("model", value.strip())
            if key in {"flags", "features"}:
                details["isa_features"] = sorted(set(value.strip().split()))
                break
    except OSError:
        pass
    return details


def gsm8k_metrics(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_new_tokens: int,
    warmup_generations: int = 2,
) -> tuple[float, float, int]:
    device = model_device(model)
    correct = 0
    generated_tokens = 0
    elapsed = 0.0
    if rows and warmup_generations:
        warmup_prompt = chat_prompt(
            tokenizer,
            "Solve this arithmetic problem. Answer with a number. 2 + 2 = ?",
        )
        warmup_inputs = tokenizer(
            warmup_prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(device)
        with torch.no_grad():
            for _ in range(warmup_generations):
                model.generate(
                    **warmup_inputs,
                    max_new_tokens=min(max_new_tokens, 16),
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
        _sync_device(device)
    for row in rows:
        user = (
            "Solve this grade-school math problem. Show your reasoning briefly and put the final numeric answer after ####.\n"
            + str(row["question"])
        )
        prompt = chat_prompt(tokenizer, user)
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        _sync_device(device)
        start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        _sync_device(device)
        elapsed += time.perf_counter() - start
        continuation = output[0, encoded["input_ids"].shape[-1] :]
        generated_tokens += int(continuation.numel())
        prediction = extract_gsm8k_answer(tokenizer.decode(continuation, skip_special_tokens=True))
        expected = extract_gsm8k_answer(str(row["answer"]))
        correct += int(prediction is not None and prediction == expected)
    accuracy = correct / max(1, len(rows))
    tokens_per_second = generated_tokens / elapsed if elapsed else 0.0
    return accuracy, tokens_per_second, generated_tokens


def artifact_bytes(path: Path) -> int:
    required_metadata = {
        "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt",
        "tokenizer.model", "spiece.model", "sentencepiece.bpe.model", "quanto_qmap.json",
        "model.safetensors.index.json", "pytorch_model.bin.index.json",
    }
    total = 0
    for file in path.rglob("*"):
        if not file.is_file():
            continue
        name = file.name
        is_weights = file.suffix == ".safetensors" or (
            file.suffix == ".bin" and (name.startswith(("pytorch_model", "model")))
        )
        if is_weights or name in required_metadata:
            total += file.stat().st_size
    return total


def run_metrics(
    model: Any,
    tokenizer: Any,
    wiki_texts: list[str],
    truthful_rows: list[dict[str, Any]],
    gsm_rows: list[dict[str, Any]],
    refusal_prompts: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    print("  [metric] WikiText perplexity", flush=True)
    metrics.update(wikitext_metrics(model, tokenizer, wiki_texts, args.max_length))
    print("  [metric] TruthfulQA multiple choice", flush=True)
    metrics["truthfulqa_mc_accuracy"] = truthfulqa_accuracy(model, tokenizer, truthful_rows)
    print("  [metric] GSM8K generation and throughput", flush=True)
    gsm_accuracy, throughput, generated = gsm8k_metrics(
        model, tokenizer, gsm_rows, args.max_new_tokens, args.warmup_generations
    )
    metrics["gsm8k_exact_match"] = gsm_accuracy
    metrics["gsm8k_generation_tokens_per_second"] = throughput
    metrics["gsm8k_generated_tokens"] = generated
    print("  [metric] refusal-marker prompts", flush=True)
    metrics["refusal_marker_rate"] = measure_refusal_rate(
        SimpleNamespace(model=model, tokenizer=tokenizer), refusal_prompts, max_new_tokens=32
    ) if refusal_prompts else math.nan
    return metrics


def main() -> int:
    args = parse_args()
    if min(args.threads, args.calibration_samples, args.wikitext_samples, args.truthfulqa_samples, args.gsm8k_samples) < 1:
        raise SystemExit("Thread count and all sample counts must be positive.")
    if args.warmup_generations < 0:
        raise SystemExit("--warmup-generations cannot be negative.")
    if args.max_length < 2 or args.max_new_tokens < 1:
        raise SystemExit("--max-length must be at least 2 and --max-new-tokens must be positive.")
    calibration_momenta = args.calibration_momenta or [0.9]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in calibration_momenta):
        raise SystemExit("Calibration momenta must be finite values in [0, 1].")
    if len(set(calibration_momenta)) != len(calibration_momenta):
        raise SystemExit("Each --calibration-momentum may only be specified once.")
    torch.set_num_threads(args.threads)
    try:
        selected_device, use_cuda_auto_map = _resolve_device(torch, args.device)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    selected_dtype = args.dtype
    if selected_dtype == "auto":
        selected_dtype = "float16" if selected_device.startswith("cuda") else "float32"
    torch_dtype = getattr(torch, selected_dtype)
    models = args.models or DEFAULT_MODELS
    if len(set(models)) != len(models):
        raise SystemExit("Each --model may only be specified once.")
    quantization_modes = args.quantization_modes or list(QUANTIZATION_MODES)
    if len(set(quantization_modes)) != len(quantization_modes):
        raise SystemExit("Each --quantization-mode may only be specified once.")
    revisions = {}
    for entry in args.revision:
        model_id, separator, revision = entry.partition("=")
        if not separator or not model_id or not revision:
            raise SystemExit(f"Invalid --revision {entry!r}; expected MODEL=REVISION.")
        if model_id in revisions:
            raise SystemExit(f"A revision was supplied more than once for {model_id!r}.")
        revisions[model_id] = revision
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results_path = output_root / "results.jsonl"
    if results_path.exists():
        raise SystemExit(f"Results already exist at {results_path}; choose a fresh --output-dir.")
    artifact_root = output_root / "artifacts"
    if args.keep_artifacts:
        artifact_root.mkdir(parents=True, exist_ok=True)

    calibration, wiki_eval, truthful_rows, gsm_rows, gsm_split, dataset_fingerprints = load_evaluation_sets(args)
    calibration_file = output_root / f"calibration-wikitext2-train-seed-{args.seed}.jsonl"
    calibration_file.write_text(
        "".join(json.dumps({"text": text}, ensure_ascii=False) + "\n" for text in calibration),
        encoding="utf-8",
    )
    refusal_pool = load_prompts("harmful")
    refusal_prompts = deterministic_sample([{"text": text} for text in refusal_pool], args.refusal_samples, args.seed + 4)
    refusal_prompts = [row["text"] for row in refusal_prompts]

    metadata = {
        "seed": args.seed,
        "models": models,
        "requested_revisions": revisions,
        "quantization_modes": quantization_modes,
        "calibration_momenta": calibration_momenta,
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu": _cpu_details(),
            "cpu_count": os.cpu_count(),
            "torch": torch.__version__,
            "available_backends": _available_backends(torch),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(torch.device(selected_device))
            if selected_device.startswith("cuda") else None,
            "selected_device": selected_device,
            "threads": args.threads,
        },
        "software": {name: importlib.metadata.version(name) for name in ("transformers", "optimum-quanto", "datasets", "huggingface-hub")},
        "evaluation": {
            "wikitext_samples": len(wiki_eval),
            "truthfulqa_samples": len(truthful_rows),
            "gsm8k_samples": len(gsm_rows),
            "gsm8k_split": gsm_split,
            "refusal_samples": len(refusal_prompts),
            "calibration_samples": len(calibration),
            "calibration_max_length": args.max_length,
            "calibration_batch_size": 1,
            "calibration_scale_algorithm": "Quanto per-batch absmax aggregated with an exponential moving average",
            "calibration_momenta": calibration_momenta,
            "calibration_streamline": not args.calibration_no_streamline,
            "max_new_tokens": args.max_new_tokens,
            "generation_warmup_generations": args.warmup_generations,
            "calibration_dataset": "Salesforce/wikitext, wikitext-2-raw-v1 train",
            "wikitext_eval_dataset": "Salesforce/wikitext, wikitext-2-raw-v1 test",
            "truthfulness_dataset": "EleutherAI/truthful_qa_mc, multiple_choice validation Parquet export",
            "reasoning_dataset": f"openai/gsm8k, main {gsm_split} Parquet export",
            "dataset_fingerprints": dataset_fingerprints,
        },
        "interpretation": "Subset metrics are comparative signals, not a safety or quality certification. TruthfulQA MC accuracy is a factuality proxy, GSM8K exact match a reasoning proxy, refusal marker rate a heuristic, and WikiText perplexity an LM fit metric.",
    }
    records: list[dict[str, Any]] = []

    for model_id in models:
        print(f"\n=== {model_id} ===", flush=True)
        snapshot = Path(snapshot_download(model_id, revision=revisions.get(model_id))).resolve()
        model_revision = snapshot.name
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        test_configs = [("none", None)]
        for quant_mode in quantization_modes:
            if quant_mode in {"quanto-int8", "quanto-int4"}:
                test_configs.extend((quant_mode, momentum) for momentum in calibration_momenta)
            else:
                test_configs.append((quant_mode, None))

        for mode, calibration_momentum in test_configs:
            mode_label = (
                f"{mode}-mom-{calibration_momentum:g}"
                if calibration_momentum is not None else mode
            )
            record: dict[str, Any] = {
                "model_id": model_id,
                "revision": model_revision,
                "mode": mode_label,
                "quantization_mode": mode,
                "calibration_momentum": calibration_momentum,
                "device": selected_device,
                "dtype": selected_dtype,
                "status": "ok",
            }
            output_dir = artifact_root / (model_id.replace("/", "--") + "--" + mode_label)
            try:
                if mode == "none":
                    print(f"  [load] baseline {mode_label}", flush=True)
                    baseline_kwargs: dict[str, Any] = {
                        "torch_dtype": torch_dtype,
                        "local_files_only": True,
                    }
                    if use_cuda_auto_map:
                        baseline_kwargs["device_map"] = "auto"
                    elif selected_device.startswith("cuda"):
                        baseline_kwargs["device_map"] = {"": selected_device}
                    else:
                        baseline_kwargs["device_map"] = None
                    model = AutoModelForCausalLM.from_pretrained(snapshot, **baseline_kwargs)
                    if baseline_kwargs["device_map"] is None and selected_device != "cpu":
                        model.to(torch.device(selected_device))
                    model.eval()
                    record["architecture"] = getattr(model.config, "model_type", "unknown")
                    artifact_size = artifact_bytes(snapshot)
                else:
                    print(f"  [transform] {mode_label}; calibration samples={args.calibration_samples if mode in {'quanto-int8', 'quanto-int4'} else 0}", flush=True)
                    if output_dir.exists():
                        raise FileExistsError(f"Refusing to overwrite benchmark artifact: {output_dir}")
                    result = run_pipeline(
                        model_path=str(snapshot),
                        output_dir=str(output_dir),
                        quantization=mode,
                        calibration_data=(
                            str(calibration_file)
                            if mode in {"quanto-int8", "quanto-int4"}
                            else None
                        ),
                        calibration_samples=args.calibration_samples,
                        calibration_max_length=args.max_length,
                        calibration_momentum=calibration_momentum if calibration_momentum is not None else 0.9,
                        calibration_streamline=not args.calibration_no_streamline,
                        device=selected_device,
                        dtype=selected_dtype,
                        disable=["tuning.refusal_direction_ablation"],
                    )
                    if not result["success"]:
                        raise RuntimeError(json.dumps(result["failed"], ensure_ascii=False))
                    record["quantization_report"] = result["quantization"]
                    record["architecture"] = (result.get("architecture") or {}).get("model_type", "unknown")
                    print(f"  [reload] {mode_label}", flush=True)
                    model = load_quantized_model(output_dir, device=selected_device)
                    if selected_device != "cpu":
                        model.to(torch.device(selected_device))
                    model.eval()
                    tokenizer = AutoTokenizer.from_pretrained(output_dir, local_files_only=True)
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    artifact_size = artifact_bytes(output_dir)

                record["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
                record["checkpoint_bytes"] = artifact_size
                record.update(run_metrics(model, tokenizer, wiki_eval, truthful_rows, gsm_rows, refusal_prompts, args))
                for metric_name, metric_value in record.items():
                    if (
                        isinstance(metric_value, (int, float))
                        and not isinstance(metric_value, bool)
                        and not math.isfinite(metric_value)
                    ):
                        raise InvalidMetricError(f"Metric {metric_name!r} is non-finite.")
                record["metrics_valid"] = True
                print(json.dumps(record, ensure_ascii=False), flush=True)
                del model
                gc.collect()
                if mode != "none" and not args.keep_artifacts:
                    shutil.rmtree(output_dir, ignore_errors=True)
            except Exception as exc:  # noqa: BLE001 - preserve other model/configuration rows when one backend fails.
                is_invalid_metric = isinstance(exc, InvalidMetricError)
                record["status"] = "invalid_metrics" if is_invalid_metric else "failed"
                record["metrics_valid"] = False
                record["error"] = f"{type(exc).__name__}: {exc}"
                print(json.dumps(record, ensure_ascii=False), flush=True)
                if output_dir.exists() and not args.keep_artifacts:
                    shutil.rmtree(output_dir, ignore_errors=True)
                gc.collect()
            records.append(record)
            with results_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    metadata["result_summary"] = {
        "total_rows": len(records),
        "successful_valid_rows": sum(row.get("status") == "ok" and row.get("metrics_valid") is True for row in records),
        "invalid_metric_rows": sum(row.get("status") == "invalid_metrics" for row in records),
        "execution_failure_rows": sum(row.get("status") == "failed" for row in records),
    }
    metadata_path = output_root / "manifest.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_path = output_root / "report.md"
    headings = ["model_id", "revision", "architecture", "mode", "momentum", "status", "WikiText PPL", "TruthfulQA-MC acc", "GSM8K exact", "GSM8K tok/s", "refusal markers", "checkpoint MiB"]
    lines = ["# Quantization quality matrix", "", f"Seed: `{args.seed}`. Hardware: `{platform.machine()}` / `{selected_device}` on `{platform.platform()}`, dtype `{selected_dtype}`, {args.threads} CPU threads. These subset results are comparative indicators, not deployment qualification.", "", "| " + " | ".join(headings) + " |", "|" + "|".join(["---"] * len(headings)) + "|"]
    for row in records:
        def show(row: dict, key: str, digits: int = 4) -> str:
            value = row.get(key)
            return f"{value:.{digits}f}" if isinstance(value, (float, int)) and math.isfinite(value) else "—"
        lines.append("| " + " | ".join([
            str(row.get("model_id", "")), str(row.get("revision", ""))[:10], str(row.get("architecture", "")), str(row.get("mode", "")), show(row, "calibration_momentum"), str(row.get("status", "")),
            show(row, "wikitext_perplexity", 3), show(row, "truthfulqa_mc_accuracy"), show(row, "gsm8k_exact_match"), show(row, "gsm8k_generation_tokens_per_second", 2), show(row, "refusal_marker_rate"),
            f"{row.get('checkpoint_bytes', 0) / (1024**2):.1f}",
        ]) + " |")
    lines += ["", "TruthfulQA-MC is a factuality/truthfulness proxy, not a comprehensive hallucination rate. GSM8K exact match is a small-sample reasoning proxy. The refusal metric detects a fixed set of refusal phrases and is not a complete safety evaluation. Weight-only controls do not calibrate or quantize activations; combined weight-plus-activation modes use the calibration settings recorded for each row. Evaluation uses the separate WikiText test split plus fixed public task samples.", "", "## Reproducibility files", "", "- `manifest.json`: seeds, sample counts, dataset splits, model set, dataset fingerprints, calibration momenta/streamline policy, and hardware/software details.", "- `results.jsonl`: one measured record per model/mode, including failures and the per-layer Quanto map summary.", f"- Calibration sample digest: `{hashlib.sha256(calibration_file.read_bytes()).hexdigest()}`.", "", "## References", "", "[1]: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct \"SmolLM2-135M-Instruct model card\"", "[2]: https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct \"SmolLM2-360M-Instruct model card\"", "[3]: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct \"Qwen2.5-0.5B-Instruct model card\"", "[4]: https://huggingface.co/datasets/Salesforce/wikitext \"WikiText dataset card\"", "[5]: https://huggingface.co/datasets/EleutherAI/truthful_qa_mc \"TruthfulQA multiple-choice dataset card\"", "[6]: https://huggingface.co/datasets/openai/gsm8k \"GSM8K dataset card\"", ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    csv_path = output_root / "results.csv"
    csv_fields = [
        "model_id", "revision", "architecture", "mode", "quantization_mode", "calibration_momentum", "status", "device", "dtype",
        "parameter_count", "checkpoint_bytes", "metrics_valid", "wikitext_mean_nll", "wikitext_perplexity",
        "truthfulqa_mc_accuracy", "gsm8k_exact_match", "gsm8k_generation_tokens_per_second",
        "gsm8k_generated_tokens", "refusal_marker_rate", "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"\nWrote {report_path}, {results_path}, and {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
