# 7. (Optional) Score-raising experiments — recipe sweep

[한국어](07_recipe_sweep.md) | **English**

We change the training method of the BF16 LoRA model trained with Unsloth ("current LoRA" below and in [results doc 5](../talk/05_recipe_sweep_en.md)) one thing at a time, and check whether accuracy on the 2021 questions goes up. No other model's outputs are used. You do not need this to reproduce the numbers in the main talk.

| What you need | Time taken (H100 NVL) | Result doc |
|---|---|---|
| The Unsloth training run (`unsloth_A`) from [step 3](03_finetune_en.md) and its checkpoints | Queue q0 about 15 min, q1 about 3 h, q2 about 2 h (estimates written in the script) | [5. Raising the score](../talk/05_recipe_sweep_en.md) |

## What runs

The configs are in [`quantization/configs/sweep/`](../../quantization/configs/sweep/), and all of them inherit from the Unsloth config (`kormedmcqa_midm_unsloth.yaml`). The header comment of each config file describes that experiment. `s0` to `s8` in the file names are the experiment numbers.

| Config file | What changed |
|---|---|
| `s0_seed43.yaml` | Same recipe, only the seed is 43. Before comparing recipes, this measures how much the score changes when only the seed changes |
| `s1_ep4.yaml` | Epochs 2 → 4. Keeps an adapter for every epoch |
| `s2_perm4.yaml` | 4 epochs + training with the order of the answer options changed every epoch (the correct answer number changes with it) |
| `s3_lr1e-4.yaml` | Option shuffling + learning rate 1e-4 |
| `s4_lr4e-4.yaml` | Option shuffling + learning rate 4e-4 |
| `s5_roles_t2.yaml` | Option shuffling + adjusting how often each profession is seen (temperature 2) |
| `s6_plw01.yaml` (optional, not run this time) | Option shuffling + loss on the question and options as well, with weight 0.1 |
| `s7_neftune5.yaml` (optional, not run this time) | Option shuffling + NEFTune noise alpha 5 |
| `s8_self_explain.yaml` | 2 epochs + explanations written by the untuned Mi:dm itself, added as training examples (max length 1,536 tokens). Serving is unchanged: it answers with the number only |
| `refit_train_dev.yaml` (not run this time) | Retrains the chosen recipe on the 2012-2021 questions (4,460 questions). Because the 2021 questions go into training, it can be measured only once, on the 2022-2024 questions |

`s3`, `s4` and `s5` inherit from whichever of `s1` and `s2` has the higher 2021 score (`extends`, currently `s2_perm4.yaml`). If `s1` is higher, change `extends` in those three configs before running q2.

## Selection rule and pass criteria

The rules were fixed in [`quantization/sweep_report.py`](../../quantization/sweep_report.py) before the runs.

- **Selection uses the 2021 questions only.** `sweep_report` does not read the 2022-2024 results.
- **Within an experiment:** Pick the checkpoint with the highest 2021 score. If an earlier checkpoint is within 0.5 percentage points (pt) of the highest score, pick the earlier one (the less-trained one).
- **Pass criteria:** At least 2.0 pt higher than the current LoRA, and p below 0.05 in the same-question comparison (McNemar).
- **The 2022-2024 questions are measured only for the one model that passes, and only once.**
- **p is nominal.** How many times the 2021 scores were looked at is recorded in `dev_scorings_looked_at`.

## Steps

The queue script [`scripts/gpu/sweep_queue.sh`](../../scripts/gpu/sweep_queue.sh) runs `stage.sh` stages one after another. For each stage it checks that the GPU is free and that there is enough disk space, and it stops if any stage fails. bash reads a script while it runs it, so copy the queue script out of the repo and run the copy.

```bash
cp ~/work/repo/scripts/gpu/sweep_queue.sh ~/work/sweep_queue.sh
setsid nohup bash ~/work/sweep_queue.sh q0 </dev/null >>~/work/logs/queue_q0.log 2>&1 &
# when q0 finishes, run q1 the same way, then q2
```

| Queue | What it does |
|---|---|
| `q0` | Tests of new code paths (option-shuffled training data read in order, the loss for explanation training, scoring with vLLM + adapter), a 100-question explanation pilot (stops if fewer than 25 explanations remain) |
| `q1` | Scores the current LoRA adapter the same way, `s0` and its merged model, `s1`, `s2`, writing explanations for the 3,401 training questions (stops if fewer than 1,000), `s8` (including scoring in explanation mode), `sweep_report` |
| `q2` | `s3`, `s4`, `s5`, `sweep_report` |
| `final` | Merges the checkpoint that passed and measures it again on the 2021 questions: `PICK_CFG=<sweep config> PICK_CKPT=<checkpoint folder>`. Then `CONFIG=<sweep config> SEED=<seed> bash scripts/gpu/stage.sh final_A` measures the 2022-2024 questions once |

When the queues finish, run the summary once more so that results are compared with the current LoRA's adapter scored the same way (no GPU needed).

```bash
python -m quantization.sweep_report --reference-lora quantization/results/kormedmcqa_midm/unsloth_A/ckpt_dev/eval_vllm_lora_A_bf16_unsloth_seed42_run_checkpoint-426_selection_dev.json
```

In this run (2026-11-13) no experiment passed, so `final` and the 2022-2024 measurement were not run.

## Verify

File paths are relative to `quantization/results/kormedmcqa_midm/sweep/`.

| Stage | Files left | What to look for |
|---|---|---|
| `sweep_a` | `<experiment>/A_train_seed<S>.json` | Whether `recipe` matches that config (learning rate, epochs, max length); `global_step`. For experiments that shuffle options or add explanations: `n_views`, `n_explain_items`, `gold_position_share` in `augment` |
| `sweep_dev` | `<experiment>/ckpt_dev/eval_vllm_lora_*_selection_dev.json` | One per checkpoint; `n_eval` 1,059, `n_unparsed` close to 0 |
| `self_explain` | `s8_self_explain/self_explain_stats_seed42.json` | `n_kept` (number of questions whose explanation was kept), `kept_by_role`. The explanation texts stay under `quantization/artifacts/` and are not committed |
| `sweep_report` | `summary_dev.json` | For each experiment: `picked` and `passes_gate`; whether `skipped_not_full_2021` is empty; `dev_scorings_looked_at` |

## Safeguards and how to stop

- **A sweep stage stops if the config is not a sweep config.** This keeps it from overwriting the models and results used in the talk.
- **It stops if the seed in the config name differs from `SEED`.** `s0_seed43.yaml` runs only with `SEED=43`.
- **If the training folder already exists, it does not train again.** To continue, set `RESUME=1`; to start from scratch, move the folder.
- **The 2022-2024 questions are measured only once, even if the scoring method differs.** Adapter scoring (`--lora-dir`) is used only for the 2021 questions.
- **To stop, stop the whole process group:** `STOP=<stage[_TAG]> bash scripts/gpu/stage.sh stop` (e.g. `STOP=sweep_a_s3_lr1e-4`). The queue and the Python and vLLM processes under it stop together. The result is recorded in `~/work/logs/stop.log`.

Next: [8. When it fails, and what to commit](08_troubleshooting_en.md)
