# Step-by-step guide
[한국어](README.md) | **English**

This is the order for checking and reproducing this repo's results yourself. Go through it from the top. Only step 1 works without a GPU; steps 2-5 and 7 need a GPU, and step 6 needs an Azure AI Foundry account.

| Step | What it does | What you need | Time taken (this run) | Output | Result doc |
|---|---|---|---|---|---|
| [1. Check without a GPU](01_cpu_check_en.md) | Tests, check the numbers in the docs against the result JSONs, redraw the charts | Python 3.10 or later | about 1 min | `OK` on screen | — |
| [2. Prepare the GPU VM](02_gpu_setup_en.md) | Create the VM, connect, install two virtual environments | 1 GPU with 80GB or more | install about 6 min | `env_<gpu>.json` | [0. Experiment setup](../talk/00_setup_en.md) |
| [3. LoRA training](03_finetune_en.md) | Untuned model score, LoRA training (transformers+PEFT, Unsloth), dev check | Step 2 | 24.3 min / 15.7 min | `A_train_seed42.json` | [1. Fine-tuning](../talk/01_finetune_unsloth_en.md) |
| [4. Shrink to INT4 and score](04_int4_en.md) | INT4 PTQ, INT4 QAT, score once on the 2022-2024 questions, same-question comparison | Step 3 | QAT training 102 min, scoring 1-6 min per model | `eval_vllm_*_h100.json`, `compare_final_*.json` | [2. INT4 quantization](../talk/02_int4_quantization_en.md) |
| [5. Measure serving speed](05_serving_en.md) | Offline throughput, online server (prefix caching on/off) | Step 4 | online, both models about 30 min | `bench_*.json`, `online_*.json` | [3. vLLM serving](../talk/03_vllm_serving_en.md) |
| [6. Compare with API models](06_managed_api_en.md) | Have managed API models answer the same questions | Azure AI Foundry, `az login` | 2.6-28.7 min per model | `managed/*.json` | [4. Cost](../talk/04_cost_en.md) |
| [7. (Optional) Score-raising experiments](07_recipe_sweep_en.md) | Change the training method and compare on the 2021 questions | Step 3 | about 4-5 h | `sweep/summary_dev.json` | [5. Raising the score](../talk/05_recipe_sweep_en.md) |
| [8. When it fails, and what to commit](08_troubleshooting_en.md) | Known pitfalls, what to do after changing results | — | — | — | — |

- **Cost:** for GPU steps, time taken × GPU price per hour. This run used 1 Azure H100 NVL at an on-demand list price of $6.98 per hour. Model download (about 23GB) and install time are not included in the table above.
- **Settings are in one file:** [`quantization/configs/kormedmcqa_midm.yaml`](../../quantization/configs/kormedmcqa_midm.yaml). The model, data split, prompt, training values and result folder are all here.
- **Run every command from the repo root.**
- **To run one step at a time while reading the code, use [`steps/`](../../steps/).** Training, INT4, scoring and serving are one Python file per step, with an explanation for each function.
- **GPU steps run through [`scripts/gpu/stage.sh`](../../scripts/gpu/stage.sh).** `bash scripts/gpu/stage.sh <stage>` runs one stage and leaves a log and its finish status. How to run directly with Python commands, without `stage.sh`, is in the “How to rerun” section of each result doc.
- **Reading the names:** the `a`·`b`·`c` and `A_bf16`·`B_int4_ptq`·`C_int4_qat` left in stage and file names are, in order, the BF16 LoRA model, INT4 PTQ and INT4 QAT. `base` is the untuned model. In file names, `selection_dev` means the 2021 questions (for selection), and scoring files with `final_holdout` or `_h100` are the 2022-2024 questions (once, at the end).
