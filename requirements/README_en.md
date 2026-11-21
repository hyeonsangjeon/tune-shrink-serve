# Install files

[한국어](README.md) | **English**

Pick one based on what you want to do. The GPU side is based on Python 3.10; the CPU check works on 3.10 or later. The API comparison (which uses `datasets`) breaks on Python 3.14, so use 3.12 or earlier.

| File | When | Where |
|---|---|---|
| [`cpu.txt`](cpu.txt) | Tests, checking numbers against results, redrawing charts (without a GPU) | Laptop, CI |
| [`gpu.txt`](gpu.txt) | INT4 conversion, evaluation, serving measurement, transformers+PEFT training | `~/venv-q` on the GPU VM |
| [`unsloth.txt`](unsloth.txt) | LoRA training with Unsloth | `~/venv-unsloth` on the GPU VM (created separately) |
| [`slides.txt`](slides.txt) | Building the talk slides | Laptop |

- Before `gpu.txt`, install torch for CUDA 13.0 first: `pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.11.0`
- On the GPU VM, [`scripts/gpu/bootstrap.sh`](../scripts/gpu/bootstrap.sh) builds both virtual environments from these files.
- The install file for the earlier experiment (Qwen3-8B · KorQuAD) is [`archive/qwen3_korquad/requirements.txt`](../archive/qwen3_korquad/requirements.txt).
- The versions actually installed are recorded in [`env_h100.json`](../quantization/results/kormedmcqa_midm/env_h100.json).
