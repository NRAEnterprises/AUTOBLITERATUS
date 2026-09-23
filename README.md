# AUTOBLITERATUS

AUTOBLITERATUS is a Python toolkit for automated language-model ablation, capability tuning, quantization-aware optimization, and evaluation.

The project is designed to organize multiple model-tuning strategies behind a configurable pipeline. It includes architecture-aware tuning, capability evaluation, runtime selection, rollback support, provenance tracking, and model export workflows.

> **Project status:** Experimental and under active development. Some tuning modules and pipeline integrations may be incomplete or subject to change.

## Features

AUTOBLITERATUS is intended to support:

- Automated model ablation and post-ablation capability tuning
- Architecture detection and architecture-aware optimization
- Quantization scheme selection and quantization-aware tuning
- LoRA and fine-tuning compatibility workflows
- Latency, throughput, memory-footprint, and batch-size tuning
- Attention-head, embedding-layer, normalization, positional-encoding, and KV-cache tuning
- Safety and robustness evaluation
- Jailbreak-resistance and prompt-injection testing
- Backdoor, poisoning, leakage, and drift detection
- Multilingual, multimodal, mathematical, coding, and reasoning capability evaluation
- Runtime feature selection through toggle configuration
- Dependency-aware pipeline orchestration
- Rollback markers and experiment reproducibility
- Model-card, provenance, audit-trail, and export support

## Requirements

- Python 3.10 or newer
- PyTorch 2.0 or newer
- Hugging Face Transformers 4.40 or newer
- Accelerate 0.30 or newer
- PEFT 0.10 or newer
- Safetensors 0.4 or newer

The complete dependency list is available in [`pyproject.toml`](pyproject.toml).

## Installation

Clone the repository:

```bash
git clone https://github.com/NRAEnterprises/AUTOBLITERATUS.git
cd AUTOBLITERATUS
