# 5. Raising the score — what works without other models' outputs
[한국어](05_recipe_sweep.md) | **English**

The shared conditions are in [0. Experiment setup](00_setup_en.md). The starting point is the BF16 LoRA model trained with Unsloth in [1. Fine-tuning](01_finetune_unsloth_en.md); this document calls it "the current LoRA". The current LoRA scored 72.2% on the 2022-2024 questions.

We used the 2021 questions to measure whether changing only the training method could raise accuracy further. There is one condition: **no outputs from other models are used.** Training used only the past exam questions and their answers, plus explanations written by the untuned Mi:dm itself.

**Bottom line: no experiment met the pass criteria.** In every experiment, the difference was about the same size as the difference that came from changing only the seed. So we did not measure the 2022-2024 questions.

## What we changed

Starting from the current LoRA's recipe (2 epochs, learning rate 2e-4, 426 steps), we changed one thing at a time.

| Experiment | What changed | Config file |
|---|---|---|
| S0 seed 43 | Same recipe; only the seed changes, 42 → 43. Before comparing recipes, this shows how much the 2021 score moves when only the seed changes | [`s0_seed43.yaml`](../../quantization/configs/sweep/s0_seed43.yaml) |
| S1 4 epochs | Only the epochs change, 2 → 4 (852 steps) | [`s1_ep4.yaml`](../../quantization/configs/sweep/s1_ep4.yaml) |
| S2 shuffled options | S1 + shuffled answer-option order. The first epoch uses the original order; each later epoch uses a different shuffle, and the answer number moves with it. The 65 questions whose numeric options are listed in order of size are not shuffled | [`s2_perm4.yaml`](../../quantization/configs/sweep/s2_perm4.yaml) |
| S3 learning rate 1e-4 | S2 + learning rate 2e-4 → 1e-4 | [`s3_lr1e-4.yaml`](../../quantization/configs/sweep/s3_lr1e-4.yaml) |
| S4 learning rate 4e-4 | S2 + learning rate 2e-4 → 4e-4 | [`s4_lr4e-4.yaml`](../../quantization/configs/sweep/s4_lr4e-4.yaml) |
| S5 profession weights | S2 + a different number of passes for each profession. Profession weights follow the square root of the question count (temperature 2), so dentist goes from 8.7% → about 15% and doctor from 55.6% → about 41%. The number of steps is almost the same | [`s5_roles_t2.yaml`](../../quantization/configs/sweep/s5_roles_t2.yaml) |
| S6 loss on the question too (optional) | S2 + loss is also applied to the question and options, with weight 0.1. The KT system text and the date line stay excluded | [`s6_plw01.yaml`](../../quantization/configs/sweep/s6_plw01.yaml) |
| S7 NEFTune (optional) | S2 + noise added to the embeddings during training (NEFTune, alpha 5) | [`s7_neftune5.yaml`](../../quantization/configs/sweep/s7_neftune5.yaml) |
| S8 Mi:dm explanations | The current LoRA + explanations written by Mi:dm itself. For each training question, the untuned Mi:dm writes an explanation and an answer 4 times; if 2 or more of them are correct, one explanation is kept. The kept explanations become "explanation → answer" examples, trained on together with the rest, once per epoch. Serving is unchanged: it answers with the number only | [`s8_self_explain.yaml`](../../quantization/configs/sweep/s8_self_explain.yaml) |

- Everything else is the same as the current LoRA: Unsloth, LoRA rank 16 and alpha 32, 16 questions per step (batch 8 × accumulation 2), maximum length 1,024 tokens.
- S3, S4 and S5 were built on S2, which had the higher 2021 score of S1 and S2.
- S8 raised the maximum length to 1,536 tokens. The explanation examples are long, so the loss is counted with the same weight per example; this keeps the number-only examples from being drowned out in the loss.
- S6 and S7 were optional experiments to run if time allowed. They were not run this time.

**Selection rule and pass criteria.** These were fixed in code before the runs ([`sweep_report.py`](../../quantization/sweep_report.py)).
- Selection uses only the 2021 questions.
- For each experiment, pick the checkpoint with the highest 2021 score. If an earlier checkpoint is within 0.5 percentage points (pt) of the highest score, pick the earlier one (the one with less training).
- To pass, an experiment must clear both of two bars: at least 2.0 pt above the current LoRA, and p below 0.05 in the same-question comparison (McNemar).
- The plan was to measure the 2022-2024 questions only once, for the one model that passed.

> **Measurement conditions**
>
> - Run: 2026-11-13, one H100 NVL 94GB, 1 run per experiment. The seed is 43 for S0 only and 42 for the rest
> - Training: 3,401 questions from 2012-2020. The adapter was saved at the end of every epoch (every 200 steps for S8 only)
> - Evaluation: 1,059 questions from 2021. Each saved adapter was loaded on top of the untuned model and scored with greedy decoding on vLLM 0.23 (`eval_vllm.py --lora-dir`)
> - Baseline: the current LoRA's last adapter, measured again the same way. Only the current LoRA and S0 were also measured as merged models
> - Result files: [`summary_dev.json`](../../quantization/results/kormedmcqa_midm/sweep/summary_dev.json) (checkpoint scores, selected checkpoints, comparison with the current LoRA), `ckpt_dev/*.json` in each experiment folder (scoring per checkpoint) and `A_train_seed<S>.json` (training time), [`self_explain_stats_seed42.json`](../../quantization/results/kormedmcqa_midm/sweep/s8_self_explain/self_explain_stats_seed42.json) (number of S8 explanations)

## Results

| Experiment | Selected checkpoint | 2021 accuracy (95% confidence interval) | vs the current LoRA: difference · questions only one got right · p | Passed |
|---|---|---|---|---|
| The current LoRA (baseline) | 426 steps (end of epoch 2) | 70.4% (67.6-73.1) | — | — |
| S0 seed 43 | 213 steps (end of epoch 1) | **72.0%** (69.2-74.6) | +1.5 pt · 58 vs 74 · 0.19 | No |
| S1 4 epochs | 426 steps (end of epoch 2) | 70.6% (67.8-73.3) | +0.2 pt · 61 vs 63 · 0.93 | No |
| S2 shuffled options | 213 steps (end of epoch 1) | 71.6% (68.8-74.2) | +1.1 pt · 50 vs 62 · 0.30 | No |
| S3 learning rate 1e-4 | 851 steps (end of epoch 4) | 71.8% (69.0-74.4) | +1.3 pt · 47 vs 61 · 0.21 | No |
| S4 learning rate 4e-4 | 426 steps (end of epoch 2) | 68.9% (66.1-71.6) | −1.5 pt · 99 vs 83 · 0.27 | No |
| S5 profession weights | 216 steps (end of epoch 1) | 70.1% (67.2-72.7) | −0.4 pt · 61 vs 57 · 0.78 | No |
| S8 Mi:dm explanations | 729 steps (end of epoch 2) | 71.6% (68.8-74.2) | +1.1 pt · 50 vs 62 · 0.30 | No |

- "Questions only one got right" is the number that only the current LoRA got right vs the number that only that experiment got right.
- p is from the exact McNemar test. Many comparisons were made on the 2021 questions, but p is the nominal value, not corrected for them.
- Differences were computed from the values before rounding.

**Measured as merged models.** The merged current LoRA scored 70.7% (67.9-73.4), 3 more questions right than the adapter measurement. The merged S0 scored 71.7% (68.9-74.3). Merged model vs merged model, the difference is +0.9 pt, 48 vs 58, p=0.38. Comparing the selected checkpoints in the table above with the merged current LoRA (70.7%) also gives differences of −1.8 to +1.2 pt with every p at 0.18 or above, so no experiment passes.

### 2021 accuracy by checkpoint

Unit: %. Bold values are the selected checkpoints. For S1, S4 and S5, an earlier checkpoint was within 0.5 pt of the highest one, so the earlier one was selected. Training time covers only the training loop (15.7 min for the current LoRA).

| Experiment | End of epoch 1 | End of epoch 2 | End of epoch 3 | End of epoch 4 | Training time |
|---|---|---|---|---|---|
| S0 seed 43 | **72.0** | 71.8 | — | — | 15.8 min |
| S1 4 epochs | 68.2 | **70.6** | 69.8 | 70.6 | 31.3 min |
| S2 shuffled options | **71.6** | 71.1 | 71.1 | 71.4 | 31.3 min |
| S3 learning rate 1e-4 | 69.9 | 71.1 | 71.0 | **71.8** | 31.3 min |
| S4 learning rate 4e-4 | 67.3 | **68.9** | 68.8 | 69.4 | 31.3 min |
| S5 profession weights | **70.1** | 68.6 | 70.4 | 69.8 | 31.3 min |

### S8 — training on explanations written by Mi:dm itself

- Of the 3,401 training questions, 2,428 kept an explanation. The untuned Mi:dm got 1,570 questions right all 4 times, and never got 554 questions right.
- The kept explanations average 282 characters.
- S8 saved the adapter every 200 steps over 2 epochs (729 steps). The same adapters were measured in two ways. Training time was 31.7 min.

| Checkpoint | 200 steps | 400 steps | 600 steps | 729 steps (end) |
|---|---|---|---|---|
| Answer with the number only (%) | 68.2 | 69.6 | 70.9 | **71.6** |
| Write an explanation, then answer (%) | 67.7 | 69.6 | 68.4 | 70.3 |

With the 729-step adapter on the 1,059 questions, answering with the number only produced 2 tokens of output per question and took 7.1 s. Writing an explanation and then answering averaged 132.5 tokens per question and took 29.3 s. In explanation mode, 2 answers had a number that could not be read, and 1 answer hit the output limit (512 tokens).

**What we saw.**
- **No experiment met the pass criteria.** Differences from the current LoRA range from −1.5 to +1.5 pt, and every p is 0.19 or above. So we did not measure the 2022-2024 questions.
- **The highest score came from S0, which did not change the recipe (72.0%).** Changing only the seed already gave a difference of 0.9-1.5 pt from the current LoRA (+0.9 pt between the merged models, +1.3 pt between the epoch-2 adapters, +1.5 pt for the selected epoch-1 adapter). Among the experiments that changed the recipe, S3 was highest at 71.8%, and S2 and S8 were at 71.6%. All of these are within the range of differences from changing only the seed.
- **S1 and S2, which saw the same data in the first epoch, differed by 3.4 pt.** In the first epoch both saw the same 3,401 questions with the same option order; only the order in which the questions were seen differed (S2's option shuffling starts from the second epoch). Their end-of-epoch-1 scores were 68.2% and 71.6%.
- **S4 (learning rate 4e-4) was below the current LoRA at all four checkpoints (67.3-69.4%).** However, the p for the selected checkpoint is 0.27.
- **S1, with epochs raised to 4, scored the same 70.6% at the end of epoch 2 and at the end of epoch 4.**
- **Having the model write an explanation before answering did not raise the score.** At all four S8 checkpoints, it was equal to or lower than answering with the number only; at the last adapter it was 70.3%, 1.2 pt lower. Output grew from 2 tokens to 132.5 tokens per question.
- **We looked at a 2021 score 31 times in total.** 26 checkpoints, 4 in S8's explanation mode, and 1 for the merged S0.

**Plausible reasons.**
- **The score difference that comes with each retraining is larger than the effect of changing the recipe.** Changing only the seed or the order in which questions are seen moved the 2021 score by 1-3 pt (S0; the first epoch of S1 and S2). If a recipe change has a smaller effect than that, a single run will not reveal it. The pass criteria of +2.0 pt and p<0.05 were also set so that differences like these would not be read as improvements.
- **S1-S5 add no new information.** They see the same 3,401 questions for longer, in a different option order, or with a different learning rate or weighting. If 2 epochs had already learned what there was to learn, more training gains little. We did not measure this separately.
- **S4's learning rate may have been too high.** A higher learning rate changes the weights more in each step, which can make the model lose some of what the untuned model knew. We did not measure this separately either.
- **S8's explanations are concentrated on questions the untuned Mi:dm already gets right.** An explanation is kept only for questions answered correctly 2 or more times out of 4, so the harder the question, the more likely it has no explanation (the 554 questions it never got right have no explanation). Why writing an explanation before answering scored lower cannot be determined from these measurements.

**What this result does not show.**
- **We did not measure methods that use other models' outputs.** No other model's output was used anywhere: not in training, not in scoring, not in filtering the explanations. The managed API results ([4. Cost](04_cost_en.md#-comparison-with-managed-apis--slm-vs-api)) were used only as a reference when choosing which methods to try. These results cannot tell what training on explanations written by a larger model would do.
- **Each experiment was run once.** The only recipe repeated with a different seed is the current one (S0), so we do not know how much each recipe's score varies with the seed. For the same reason we cannot state rankings such as "S3 is better than S2" or "S4 is bad".
- **The selected scores may be higher than the true values.** We looked at the 2021 score 31 times and selected from among them. Picking the highest value tends to pick a value that came out high by chance. The p values are also nominal, not corrected for the multiple comparisons.
- **Scores measured with the adapter loaded and scores of the merged model differ slightly.** The current LoRA scored 70.4% with the adapter and 70.7% merged. S0 scored 71.8% with the epoch-2 adapter and 71.7% merged. The gap is 1-3 questions and its direction is not consistent, so adapter scores were compared only with adapter scores, and merged models only with merged models. Apart from S0, no experiment was measured as a merged model.
- **We do not know the scores on the 2022-2024 questions.** No experiment passed, so they were not measured. Whether the experiments that were a little over 1 pt higher on 2021 would also be higher on 2022-2024 is unknown.
- **Only the answer number in the explanations was checked.** S8 used only explanations written by the untuned Mi:dm itself, and kept explanations only for questions where 2 or more of the 4 ended with the correct answer number. No person checked whether the content of the explanations is medically correct.
- **The explanation texts are not in the repo.** They restate the content of past exam questions, so they were not committed, following the data license (CC BY-NC 2.0). The repo keeps only their counts and lengths (`self_explain_stats_seed42.json`).
- **We did not measure S6, S7, or combinations of recipes.** For example, there is no experiment that uses S3's learning rate together with S5's profession weights.

## To run it again

The order and safeguards are in [step-by-step guide 7](../guide/07_recipe_sweep_en.md). The queue runs on the GPU VM; summarizing the results needs no GPU.

```bash
cp ~/work/repo/scripts/gpu/sweep_queue.sh ~/work/sweep_queue.sh      # copy outside the repo so the script does not change while it runs
setsid nohup bash ~/work/sweep_queue.sh q0 </dev/null >>~/work/logs/queue_q0.log 2>&1 &   # when it finishes, q1, then q2
# Summarize the 2021 results: pass the current LoRA's adapter score as well, so that both sides are measured the same way
python -m quantization.sweep_report --reference-lora quantization/results/kormedmcqa_midm/unsloth_A/ckpt_dev/eval_vllm_lora_A_bf16_unsloth_seed42_run_checkpoint-426_selection_dev.json
```
