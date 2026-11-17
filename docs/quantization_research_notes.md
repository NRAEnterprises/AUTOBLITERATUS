# Quantization backend research notes

Research status: current comparative documentation review for the Linux ARM64, six-CPU, no-NVIDIA sandbox. No backend has been executed from this source review; compatibility claims below are documentation-level until the integration experiments run.

## BitsAndBytes
- Transformers documents BitsAndBytes 8-bit LLM.int8 and 4-bit FP4/NF4 loading, save, and reload integration. It quantizes during loading and does **not** perform representative-dataset calibration.
- Official support materials list Linux aarch64 CPU and x86_64 CPU; CUDA requires a compatible NVIDIA device and build. Local CUDA tests are impossible here.
- Source URLs:
  - https://huggingface.co/docs/transformers/quantization/bitsandbytes
  - https://huggingface.co/docs/bitsandbytes/installation
  - https://github.com/bitsandbytes-foundation/bitsandbytes#system-requirements
  - https://github.com/bitsandbytes-foundation/bitsandbytes/releases/tag/0.50.0

## GPTQ / GPTQModel
- GPTQ is calibration-based; its official example uses representative text samples, quantizes, saves packed weights/metadata, and reloads via GPTQModel or the documented Transformers integration.
- NVIDIA CUDA and x86 CPU paths are documented. Linux ARM64 CPU calibration is not established by the project platform matrix; a portable ATen CPU path is mentioned but does not certify ARM64 end-to-end support.
- Source URLs:
  - https://github.com/ModelCloud/GPTQModel/releases/tag/v7.10.0
  - https://github.com/ModelCloud/GPTQModel/blob/v7.10.0/README.md
  - https://github.com/ModelCloud/GPTQModel/wiki/Installation-and-Quickstart
  - https://github.com/ModelCloud/GPTQModel/wiki/Inference-Backends
  - https://github.com/ModelCloud/GPTQModel/blob/main/docs/torch_fused_int4_transformations.md
  - https://huggingface.co/docs/transformers/en/quantization/gptq
  - https://github.com/ModelCloud/GPTQModel/blob/v7.10.0/requirements.txt
  - https://github.com/ModelCloud/GPTQModel/wiki/HuggingFace-Integration

## AWQ / AutoAWQ
- AutoAWQ documents calibration, packed 4-bit quantization, and Transformers reload. It is officially deprecated/unmaintained.
- Its documented CPU path is Intel/IPEX x86, not ARM64. CUDA is supported on compatible NVIDIA hardware. No ARM64 CPU support was verified.
- Source URLs:
  - https://huggingface.co/docs/transformers/en/quantization/awq
  - https://huggingface.co/docs/transformers/en/quantization/overview
  - https://github.com/casper-hansen/AutoAWQ
  - https://github.com/casper-hansen/AutoAWQ/blob/main/docs/examples.md
  - https://github.com/casper-hansen/AutoAWQ/blob/main/setup.py
  - https://github.com/mit-han-lab/llm-awq

## TorchAO
- Transformers supports TorchAO weight-only/dynamic quantization and save/reload integration. The simple cited recipes are not representative-data calibration by themselves.
- ARM64 Linux low-bit support is possible through specific KleidiAI source-build paths, not guaranteed by generic wheels. CUDA and x86 CPU configurations are documented; some packed int4 formats are device-specific.
- Source URLs:
  - https://huggingface.co/docs/transformers/en/quantization/torchao
  - https://docs.pytorch.org/ao/stable/workflows/inference.html
  - https://docs.pytorch.org/ao/stable/eager_tutorials/torchao_hf_integration.html
  - https://github.com/pytorch/ao/releases/tag/v0.18.0
  - https://github.com/pytorch/ao/pull/2169
  - https://github.com/pytorch/ao/issues/2143

## Optimum Quanto
- The direct Quanto API supports representative activation calibration: `quantize(..., activations=qint8)`, run representative inputs inside `Calibration(momentum=...)`, then `freeze` and serialize. The official Transformers `QuantoConfig` path is weight-only; direct Quanto is needed for activation calibration.
- Official wrapper support is `QuantizedModelForCausalLM.quantize/save_pretrained/from_pretrained`. Direct low-level save uses Safetensors plus `quanto_qmap.json`; wrapper reload reconstructs the quantized model from that map.
- Project is in maintenance mode; CPU is documented/CI-tested, but Linux ARM64 is not specifically certified. Test it on the actual host.
- Sources:
  - https://github.com/huggingface/optimum-quanto
  - https://huggingface.co/docs/transformers/en/quantization/quanto
  - https://raw.githubusercontent.com/huggingface/optimum-quanto/main/optimum/quanto/models/transformers_models.py
  - https://raw.githubusercontent.com/huggingface/optimum-quanto/main/optimum/quanto/calibrate.py
  - https://github.com/huggingface/optimum-quanto/releases/tag/v0.2.7
  - https://raw.githubusercontent.com/huggingface/optimum-quanto/v0.2.7/pyproject.toml

## Test interpretation
- The current host is Linux AArch64, six CPU cores, no NVIDIA device, and initially no ML/test packages. Actual GPU or x86 measurements require additional machines.
- Tiny random weights can validate conversion and reload plumbing only. They cannot establish accuracy, hallucination, refusal, reasoning, or meaningful throughput quality.
- Compare real pretrained checkpoints on identical prompts/data/settings, keep each artifact distinct, report source/backend versions and hardware, and treat unavailable hardware as untested rather than simulated.
