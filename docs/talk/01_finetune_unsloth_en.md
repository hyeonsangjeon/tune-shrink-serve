# 1. Fine-tuning — the same LoRA with transformers+PEFT and with Unsloth

[한국어](01_finetune_unsloth.md) | **English**

Shared conditions (model, data, prompt, scoring, hardware) are in [0. Experiment setup](00_setup_en.md).

## What was changed

**The only thing changed is the training engine.**

- transformers `Trainer` + PEFT LoRA
- Unsloth `FastLanguageModel` + `get_peft_model`

Data, prompt, LoRA settings, training schedule, seed and GPU were kept the same. After training, both engines merge the LoRA into the original weights and save a single BF16 model. The INT4 conversion in ② and the serving in ③ take this model as is. ② and ③ used the model trained with transformers+PEFT.

## Training settings

| Item | Value |
|---|---|
| Training data | 3,401 past questions from 2012-2020, 2 epochs ([example question](00_setup_en.md#what-one-question-looks-like)) |
| optimizer steps | 426 (batch 8 × gradient accumulation 2 = 16 questions per step) |
| LoRA | rank 16, alpha 32, dropout 0, targets `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |
| Learning rate | 2e-4, warmup 3%, weight decay 0.01, gradient clipping 1.0 |
| Precision | BF16 (not loaded in 4-bit) |
| Maximum length | 1,024 tokens |
| gradient checkpointing | On. transformers+PEFT uses the default method; Unsloth uses its own method (`use_gradient_checkpointing="unsloth"`) |
| Loss | Applied only to the answer number and `<|eot_id|>` |

> **Measurement conditions**
>
> - Run: 2026-11-10 to 11-11, Azure H100 NVL 94GB, 1 GPU, seed 42, 1 run per engine
> - Training time: measures only the training loop (`trainer.train`). Model loading and merge/save are not included
> - GPU memory: read directly with `nvidia-smi` during training. It is not the peak but the value at the moment of reading. It includes free memory that PyTorch has reserved
> - Accuracy: measured with greedy decoding while served with vLLM 0.23. 1,059 questions from 2021, 3,009 questions from 2022-2024
> - Result files: [`A_train_seed42.json`](../../quantization/results/kormedmcqa_midm/A_train_seed42.json), [`unsloth_A/A_train_seed42.json`](../../quantization/results/kormedmcqa_midm/unsloth_A/A_train_seed42.json), [`train_gpu_memory_observed_h100.json`](../../quantization/results/kormedmcqa_midm/train_gpu_memory_observed_h100.json)

## Results

![Training time and GPU memory of transformers+PEFT and Unsloth](figures/finetune_engines_en.svg)

| | transformers+PEFT | Unsloth |
|---|---|---|
| Training time (426 steps) | 24.3 min (1,457.6 s) | **15.7 min** (943.9 s) |
| GPU memory during training (`nvidia-smi`) | 82.3 GiB | **25.5 GiB** |
| Mean training loss | 1.879 | 1.874 |
| Accuracy, 1,059 questions from 2021 | 70.4% (67.6-73.1) | 70.7% (67.9-73.4) |
| Accuracy, 3,009 questions from 2022-2024 | 72.3% (70.7-73.9) | 72.2% (70.6-73.8) |

Parentheses show the 95% confidence interval. The untuned model scored 63.3% on the same 3,009 questions.

When paired question by question, the difference between the two engines can be explained by chance.

| Comparison (paired by question) | Questions only one side got right | McNemar p |
|---|---|---|
| transformers+PEFT vs Unsloth, 2021 | 36 vs 39 | 0.82 |
| transformers+PEFT vs Unsloth, 2022-2024 | 113 vs 110 | 0.89 |
| untuned vs transformers+PEFT LoRA, 2022-2024 | 164 vs 436 | below 0.000001 |

"Questions only one side got right" gives the number that only the first-listed side got right vs the number that only the second-listed side got right.

**Observed.**
- Training time fell by 35%.
- Memory as seen by `nvidia-smi` fell below 1/3.
- Accuracy and training loss were the same for the two engines.
- LoRA training alone raised accuracy by 9.0 percentage points (pt) over the untuned model, and all four professions improved (per-profession numbers are in [2. INT4 quantization](02_int4_quantization_en.md#accuracy-by-profession)).

**Plausible reason.** Unsloth uses its own kernels that fuse the LoRA computation, and its own gradient checkpointing that offloads activations to the CPU. The speed and memory differences are likely to come from these two. However, we did not separate them by profiling.

**What this result does not show.**
- **The memory numbers must be compared on the same basis.** Both values are single readings from `nvidia-smi`. For transformers+PEFT, the peak allocation as counted by PyTorch was not recorded. For Unsloth, the peak allocation as counted by PyTorch is 24.5 GiB.
- **The gradient checkpointing methods differ.** Part of the difference may come from this setting rather than from the engine, and it was not measured separately.
- **The software environments differ slightly.** The Unsloth environment uses torch 2.14.1; the other uses 2.11.0.
- **A single seed, one GPU type.** We do not know how much the score on the 2022-2024 questions changes when training is repeated. Running the Unsloth training once more with seed 43 and measuring on the 2021 questions gave a difference of 0.9-1.5 pt ([5. Raising the score](05_recipe_sweep_en.md)). In the table above, the accuracy difference between the two engines (0.3 pt on the 2021 questions, 0.1 pt on the 2022-2024 questions) is smaller than that.

## What it took to use Unsloth

Loading Mi:dm 2.0 as is with Unsloth 2026.11.3 stops at the tokenizer. If you want to try this again, knowing the two points below will save time.

1. **Pin transformers to 4.57.6.** Installing Unsloth also installs transformers 5.17, and 5.17 cannot read the Mi:dm tokenizer (`TypeError: argument of type 'tokenizers.AddedToken' is not iterable`).
2. **Load the tokenizer separately and pass it in.** Even with 4.57.6, running `import unsloth` while a GPU is visible breaks tokenizer loading again (`AttributeError: 'tokenizers.AddedToken' object has no attribute 'keys'`). It did not reproduce in an environment without a GPU, and we did not find the patch that causes it.
   - Workaround: `train.unsloth_external_tokenizer: true` in [`kormedmcqa_midm_unsloth.yaml`](../../quantization/configs/kormedmcqa_midm_unsloth.yaml)
   - With this option on, a separate process that has not imported Unsloth reads the tokenizer with plain transformers and passes it in.
   - Model loading and training are done by Unsloth.

After the workaround, two things were checked: that the merged model's vocabulary is still 131,384, and that its tokenizer produces the same prompt tokens and stop tokens as the untuned model (`scripts/gpu/stage.sh a_smoke`).

## How to rerun

GPU environment setup and the checks for each stage are in [step-by-step guide 2 and 3](../guide/02_gpu_setup_en.md). The training commands are:

```bash
# transformers + PEFT
python -m quantization.v2_run a --config quantization/configs/kormedmcqa_midm.yaml --seed 42 --backend hf

# Unsloth (in the separate virtual environment; results go to the unsloth_A/ folder)
python -m quantization.v2_run a --config quantization/configs/kormedmcqa_midm_unsloth.yaml --seed 42

# dev check on the 2021 questions
python -m quantization.eval_vllm --config quantization/configs/kormedmcqa_midm.yaml --method A_bf16 --split selection_dev
```

The trained model is published on Hugging Face: [`HyeonSang/Midm-2.0-KorMedMCQA`](https://huggingface.co/HyeonSang/Midm-2.0-KorMedMCQA) (the merged BF16 LoRA model trained with transformers+PEFT).
