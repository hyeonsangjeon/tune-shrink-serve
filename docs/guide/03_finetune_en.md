# 3. LoRA training
[한국어](03_finetune.md) | **English**

Measure the untuned model's score, train LoRA on Mi:dm 2.0 Base, and merge it into a single BF16 model. The same training is done once with each of two engines (transformers+PEFT, Unsloth). Later steps (INT4, serving) use the model trained with transformers+PEFT.

| What you need | Time taken (H100 NVL) | Result doc |
|---|---|---|
| A VM that has finished [step 2](02_gpu_setup_en.md) | Training: transformers+PEFT 24.3 min, Unsloth 15.7 min | [1. Fine-tuning](../talk/01_finetune_unsloth_en.md) |

To read the training code, see [`steps/1_lora_train.py`](../../steps/1_lora_train.py). It covers attaching LoRA with PEFT and Unsloth, training and merging, with an explanation for each function. You can also run it directly with `python steps/1_lora_train.py`.

## Steps

```bash
bash scripts/gpu/stage.sh selftest           # 0. check on a small matrix that QAT fake quantization uses the same scheme as the served INT4
bash scripts/gpu/stage.sh base_dev           # 1. untuned model, 1,059 questions from 2021
BACKEND=hf bash scripts/gpu/stage.sh a_smoke # 2. 5-step trial training (when it finishes, its outputs are moved to *_smoke)
BACKEND=hf bash scripts/gpu/stage.sh a       # 3. main training, 426 steps (2 epochs)
BACKEND=hf bash scripts/gpu/stage.sh a_dev   # 4. trained model, 2021 questions
bash scripts/gpu/stage.sh gate               # 5. did it go up by at least 2 percentage points (pt) over the untuned model? (stops if not)
```

To do the same training with Unsloth, change only the config. Its results go to a separate `unsloth_A/` folder and do not overwrite the results above.

```bash
CONFIG=quantization/configs/kormedmcqa_midm_unsloth.yaml bash scripts/gpu/stage.sh a_smoke
CONFIG=quantization/configs/kormedmcqa_midm_unsloth.yaml bash scripts/gpu/stage.sh a
```

- `BACKEND=hf` trains with transformers+PEFT in `~/venv-q`. Without it, training uses Unsloth in `~/venv-unsloth` (the config default).
- The training values (LoRA rank 16 and alpha 32, learning rate 2e-4, 16 questions per step) are under `train:` in [`kormedmcqa_midm.yaml`](../../quantization/configs/kormedmcqa_midm.yaml).

## Verify

Result files are written to `quantization/results/kormedmcqa_midm/`.

| Stage | File | What to look for |
|---|---|---|
| `selftest` | `qat_selftest.json` | `ok: true` |
| `base_dev` | `eval_vllm_base_seed42_selection_dev.json` | `n_unparsed` (answers whose option number could not be read) is close to 0. If it is high, fix the prompt first. This run's accuracy is in the same file |
| `a_smoke` | log `~/work/logs/a_smoke.log` | The merged folder is created without errors, with `vocab_ok: true` and `prompt_ids_same: true` |
| `a` | `A_train_seed42.json` | `global_step` 426, `train_seconds`. The log must have no `[tokenize] WARNING` (length exceeded) |
| `a_dev` | `eval_vllm_A_bf16_seed42_selection_dev.json` | `n_eval` 1,059, `n_unparsed` close to 0 |
| `gate` | `gate_dev_seed42.json` | `pass: true`. Also check that both results came from the same GPU |

The trained model is written to `quantization/artifacts/kormedmcqa_midm/A_bf16_seed42/` (BF16 safetensors, about 22GB). Do not commit the weights.

## When it fails

- **Unsloth cannot read the Mi:dm tokenizer:** `train.unsloth_external_tokenizer: true` in the config loads the tokenizer separately and passes it in (it is already set in the Unsloth config). Details are in [results doc 1](../talk/01_finetune_unsloth_en.md#what-it-took-to-use-unsloth).
- **Main training resumes from step 5:** you ran the trial training by hand and left a `*_run/` folder behind. Remove the folder and run again. `stage.sh a_smoke` does this for you.
- **The Spot VM was reclaimed:** start the VM again, then continue from the last checkpoint (every 100 steps) with `RESUME=1 BACKEND=hf bash scripts/gpu/stage.sh a`.
- **`gate` stops (exit code 3):** training did not raise the score on the 2021 questions by 2 pt or more. Look at the training log and `n_unparsed` first.

Next: [4. Shrink to INT4 and score](04_int4_en.md)
