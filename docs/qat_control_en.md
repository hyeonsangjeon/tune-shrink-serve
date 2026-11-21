# QAT control experiment: is it the fake quantization or the extra training?
[한국어](qat_control.md) | **English**

This experiment was run separately from the talk. The "Plan" section of this document was written before the run (2026-11-17 05:31 UTC). Results will be appended below after the run finishes. The plan section is not edited after seeing the results.

## Question

INT4 QAT ([`steps/3_int4_qat.py`](../steps/3_int4_qat.py)) differs from PTQ in two ways.

1. During training, the weights are rounded to INT4 and then converted back before use (fake quantization). The model adjusts its weights while experiencing the rounding error.
2. It trains for 600 more steps (about 2.8 passes) on the same 3,401 training questions.

In the talk run, QAT was +0.2 percentage points (pt) above PTQ (p=0.67); in a run retrained with the same code, it was +1.0 pt (p=0.02). When QAT is better, the runs so far cannot tell whether that is due to item 1 or item 2. This experiment builds models that get item 2 but not item 1, and compares them.

## Plan (written before the run)

### What we run

All three start from the talk's BF16 LoRA model ([`HyeonSang/Midm-2.0-KorMedMCQA`](https://huggingface.co/HyeonSang/Midm-2.0-KorMedMCQA), 72.3% on the 2022-2024 questions).

| Name | What it does | seed |
|---|---|---|
| P | Convert straight to INT4 (PTQ, `steps/2_int4_ptq.py`) | — |
| Q | Train 600 steps with fake quantization → INT4 (the current QAT) | 42, 43, 44 |
| X | Train the same 600 steps without fake quantization → BF16 → INT4 (`--no-fake-quant`) | 42, 43, 44 |

For X, the BF16 model before the INT4 conversion is also scored. This checks whether "a BF16 model that only got more training" improved over the original BF16 LoRA model.

**The only thing that differs between Q and X is fake quantization.** Kept the same:

- Trained weights: all linear layers except the output layer (lm_head), about 10.5 billion. We checked with a small model that both paths train the same weights (same trained weight names and count).
- Training settings: learning rate 1e-5, 600 steps, batch 4 × accumulation 4, 8-bit AdamW, warmup 3%, gradient checkpointing.
- Training questions: the same 3,401 as in step 1 (2012-2020).
- INT4 settings: same as step 2 (`Int4WeightOnlyConfig`, group 128).
- Scoring: `steps/4_evaluate.py`, 3,009 questions from 2022-2024, once per model.
- GPU and software: one H100 NVL, the same environment as the 11-16 to 11-17 retraining.

**The seed changes only the order in which the training questions are shuffled.** Q and X with the same seed see the questions in the same order. So for each seed, Q and X are compared as a pair.

### Comparisons and decision rules

For each seed s, three values are computed (accuracy, pt).

| Comparison | Formula | Question it asks |
|---|---|---|
| ① Fake quantization effect | Q_s − X_s(INT4) | Did training while experiencing the rounding help on its own? |
| ② Extra training effect (INT4) | X_s(INT4) − P | Does more training alone raise the INT4 score? |
| ③ Extra training effect (BF16) | X_s(BF16) − 72.3 | Does more training alone raise the BF16 score? |

The same decision rule applies to all three comparisons.

- **Effect present:** all three seeds in the same direction, and a mean difference of 0.5 pt or more
- **No effect:** the size of the mean difference is 0.2 pt or less
- **Otherwise:** undecided

0.5 pt is the size of the variation that comes from the questions when two models are compared on the same 3,009 questions. When about 200 questions are answered correctly by only one of the two models, the standard deviation of the difference is √200 / 3,009 ≈ 0.47 pt. The McNemar p is also recorded for each pair, but it is not used for the verdict, because McNemar does not include the variation from retraining.

### Predictions (written before the run)

- ① Mean of +0.3 to 0.8 pt. The prediction is that fake quantization helps a little. We expect the verdict to fall around the boundary between "Undecided" and "Effect present".
- ② Mean near 0. We expect that training an already fine-tuned model further at learning rate 1e-5 will barely change the INT4 score.
- ③ Mean near 0, or slightly down.

These predictions are wrong if the following results appear.

- ① Mean of 0.2 pt or less: the amount by which QAT beat PTQ is explained by the extra training.
- ② As large as ① or larger: QAT's gain came from the extra training rather than from fake quantization.

### Known limitations

- **The starting point is fixed to a single BF16 LoRA model.** The variation from retraining the LoRA is not part of this experiment. When QAT−PTQ moved from +0.2 pt to +1.0 pt in the earlier retraining, that variation was part of the change too.
- **Even with the same seed, GPU computation differs slightly each time, so results are not byte-identical** (confirmed in the 11-16 to 11-17 retraining). This variation also goes into the differences between seeds.
- **With 3 seeds, the size of the variation is known only roughly.**
- **No settings are chosen using the 2022-2024 questions.** All settings are the talk values as they were, and each new model is scored only once.
- **Learning rate, number of steps, and trained layers are not varied.** The answer holds only for this setting.

### Order, stopping conditions, cost

1. Small check: train X for only 5 steps and score it on 50 of the 2021 questions. If the number of trained weights differs from the QAT run record (10,468,982,784), the main run is not done.
2. Build P and score it.
3. For seeds 42, 43, 44 in that order: X (BF16 scoring, INT4 scoring) → Q (INT4 scoring). Each seed finishes one pair, so even if the run stops partway, the pairs finished so far can be compared.

Expected time is about 11-12 hours, about $80 at $6.98/h for the H100 NVL. If it goes past 14 hours (about $100), stop and report with the pairs finished by then. When done, shut the VM down (deallocate).

Result files go in [`quantization/results/kormedmcqa_midm/qat_control/`](../quantization/results/kormedmcqa_midm/qat_control/). They contain only the per-question answers, not the question text.

## Plan addendum (2026-11-17 09:15 UTC, written after seeing only the seed 42 results)

Written after the seed 42 pair finished, before seeing the results for seeds 43 and 44.

**Two more pairs, seeds 45 and 46, will be run.** They will be run regardless of the results for seeds 43 and 44.

The reason is the variation between runs with the same settings. The seed 42 QAT has been trained three times with the same settings, and scored 69.69% (talk), 70.09% (11-16 to 11-17 retraining), and 69.96% (this run). Runs with the same settings differ by up to 0.4 pt, which is close to the 0.5 pt decision threshold. With 3 seeds, an "Undecided" verdict is likely, so we add seeds.

- **The first verdict is not changed.** The result of applying the decision rule above to seeds 42, 43, 44 is reported first.
- **The 5-seed result is written up separately.** The same rule is applied to the 5 seeds (all five seeds in the same direction, mean of 0.5 pt or more). The difference for each seed and the mean are also recorded. If the two results differ, we say so.
- The procedure is the same as above. For each seed, the order is X (BF16 scoring, INT4 scoring) → Q (INT4 scoring).
- **Time and cost:** one pair takes about 2.9 hours, so this adds about 5.8 hours and about $40. The cap changes to 17 hours from when the VM was started (about $120). If that time is exceeded, report with the pairs finished by then.

**Correction (2026-11-17 14:25 UTC, after seeing results up to seed 44).** Of the three scores given under "reason" above, 70.09% did not come from the same settings. The 11-16 to 11-17 retraining trained everything again starting from the BF16 LoRA model, so its QAT started from a different model (the same holds for that run's resumed-training run, which scored 69.69%). Only two runs share both the starting model and the seed: the talk QAT (69.69%) and this seed 42 QAT (69.96%). They differ by 0.27 pt. The grounds for the decision to add seeds were weaker than written. The decision to run seeds 45 and 46 and the rules above do not change.

## Results (written 2026-11-18)

### What happened during the run

- The VM was started at 2026-11-17 05:32 UTC, and the run began at 05:58. The small check passed (X trains 10,468,982,784 weights, the same as QAT). P, and X and Q for seeds 42, 43, 44, had all finished at 14:04.
- Next, X for seed 45 (training, BF16 scoring, INT4 scoring) finished at 14:51.
- **The VM was shut down at 15:50 UTC.** Automation run by the organization that manages the subscription deallocated the VM. The activity log does not record a reason. At that point Q for seed 45 had trained 346 of its 600 steps, and seed 46 had not started.
- **The per-question scoring files were lost.** Each model's answers to the 3,009 questions were saved on the VM's local NVMe, and that disk is wiped when the VM is deallocated. The run records and logs left on the OS disk were retrieved on 11-18 by starting the VM for about 3 minutes.
- So the numbers below come from the records the run printed. The counts of questions only one model got right, and the McNemar p, exist only for seeds 42-44. P gave the same answer as the talk's INT4 PTQ on all 3,009 questions. So P's per-question answers are the same as the talk's PTQ scoring file ([`eval_vllm_B_int4_ptq_seed42_h100.json`](../quantization/results/kormedmcqa_midm/eval_vllm_B_int4_ptq_seed42_h100.json)).
- The VM was on for 10.36 hours in total, and the cost was $72.3 (computed at $6.98/h for the H100 NVL, Bastion cost excluded). This is within the 17-hour cap.

### Accuracy (2022-2024, 3,009 questions)

The starting BF16 LoRA model (A) scored 72.316% (2,176 questions), and P scored 69.458% (2,090 questions).

| Model | seed 42 | seed 43 | seed 44 | seed 45 |
|---|---|---|---|---|
| Q: trained with fake quantization → INT4 | 69.957% (2,105) | 69.957% (2,105) | 69.724% (2,098) | stopped at step 346 |
| X: trained without fake quantization, kept in BF16 | 71.751% (2,159) | 72.051% (2,168) | 71.751% (2,159) | 71.851% (2,162) |
| X: trained without fake quantization → INT4 | 69.425% (2,089) | 69.392% (2,088) | 69.159% (2,081) | 69.325% (2,086) |

Numbers in parentheses are the questions answered correctly. X (INT4) for seed 43 has 2 questions whose answer number could not be read; all the others have 0. For seed 45's X, the run record has no count of correct answers. The per-profession accuracy in the scoring log (one decimal place, 435-885 questions per profession) narrows the correct count down to a single value, so that value is given. Recovering the counts for the 9 models of seeds 42-44 in the same way gives the same counts as the run record in every case.

### Comparisons

Differences are in pt. In parentheses: the number of questions only one model got right (first model vs second model), and the McNemar p.

| Comparison | seed 42 | seed 43 | seed 44 | Mean (42-44) | seed 45 |
|---|---|---|---|---|---|
| ① Fake quantization effect: Q − X(INT4) | +0.53 (114 vs 98, p=0.303) | +0.56 (120 vs 103, p=0.284) | +0.56 (120 vs 103, p=0.284) | **+0.55** | none |
| ② Extra training effect (INT4): X(INT4) − P | −0.03 (125 vs 126, p=1.000) | −0.07 (117 vs 119, p=0.948) | −0.30 (103 vs 112, p=0.585) | **−0.13** | −0.13 |
| ③ Extra training effect (BF16): X(BF16) − A | −0.56 (64 vs 81, p=0.184) | −0.27 (57 vs 65, p=0.526) | −0.56 (58 vs 75, p=0.165) | **−0.47** | −0.47 |
| For reference, not a verdict item: Q − P | +0.50 (107 vs 92, p=0.321) | +0.50 (98 vs 83, p=0.298) | +0.27 (106 vs 98, p=0.624) | +0.42 | none |

For seed 45 there is only the accuracy difference, because the per-question files had been lost by then.

### Verdicts (planned seeds 42, 43, 44)

| Comparison | Mean | Direction across the three seeds | Verdict | Prediction before the run |
|---|---|---|---|---|
| ① Fake quantization effect | +0.55 pt | all + | **Effect present** | +0.3 to 0.8 pt, around the boundary between "Undecided" and "Effect present" |
| ② Extra training effect (INT4) | −0.13 pt | all − | **No effect** | near 0 |
| ③ Extra training effect (BF16) | −0.47 pt | all − | **Undecided** | near 0, or slightly down |

Neither of the two results that we had said would make the predictions wrong (① mean of 0.2 pt or less; ② as large as ① or larger) occurred.

**The 5-seed verdict could not be made.** Q for seed 45 did not finish, and seed 46 did not run. Applying the same rule to the seeds that did finish gives ② a 4-seed mean of −0.13 pt, "No effect", and ③ a 4-seed mean of −0.47 pt, "Undecided". ① stays at 3 seeds. This is not the "5-seed result" defined in the plan addendum.

### What we saw

- In INT4 accuracy, the model trained with fake quantization (Q) was 0.53-0.56 pt higher than the one trained without it (X), for all three seeds.
- The model trained 600 more steps without fake quantization was almost the same as P once converted to INT4 (−0.03 to −0.30 pt across 4 seeds).
- That model's BF16 accuracy was below the starting point for all four seeds (−0.27 to −0.56 pt).
- In this setting X (INT4) and P are almost the same, so Q − X (+0.55 pt) and Q − P (+0.42 pt) are similar. The amount by which QAT scored above PTQ is not explained by the extra training.
- Training time for 600 steps was 37.6-37.7 min for X and 101.1 min for Q. With fake quantization, training was about 2.7 times slower.
- Training loss was slightly higher for Q. The mean loss over the 600 steps was 1.336-1.344 for Q and 1.317-1.320 for X.

### Plausible reasons

With fake quantization, the model experiences the INT4 rounding error during training and moves its weights in the direction where that error does less harm. This is the original explanation for QAT, and ① is consistent with it. However, we did not measure what changed (which layers, which questions). The drop in BF16 accuracy in ③ may be because the model saw the same 3,401 questions about 2.8 more times. ③ fell short of the decision threshold, and we did not measure this cause separately either.

### What this result does not show

- **The "Effect present" verdict for ① only just clears the threshold.** The mean of +0.55 pt is 0.05 pt above the 0.5 pt threshold. The McNemar p for each seed is 0.28-0.30, so any single pair on its own cannot be told apart from the variation that comes from the questions. The verdict rests on all three seeds pointing the same way.
- **Adding seeds does not reduce this uncertainty.** The difference in ① between seeds is 0.03 pt, much smaller than the variation that comes from the questions (about 0.47 pt). Because every seed was scored on the same 3,009 questions, the three seeds pointing the same way means the variation from retraining is small. It does not mean the same difference would appear when scoring on different questions.
- **There is a single starting point, one BF16 LoRA model.** In the 11-16 to 11-17 run, which retrained from the LoRA onward, QAT − PTQ was +1.0 pt; the mean here is +0.42 pt. The variation from retraining the LoRA is not included in this experiment.
- **This experiment does not show that QAT is better than PTQ either.** The Q − P mean of +0.42 pt, computed for reference, is "Undecided" under the same rule. What this experiment can say goes only as far as "the amount by which QAT scored above PTQ is not due to the extra training".
- **Only one value each was tried for learning rate, number of steps, and trained layers** (1e-5, 600 steps, all linear layers except lm_head).
- **Without the per-question files, some analyses cannot be redone.** Which questions fake quantization helped on, and the paired comparison for seed 45, cannot be computed.

### Records and re-running

The records are in [`quantization/results/kormedmcqa_midm/qat_control/`](../quantization/results/kormedmcqa_midm/qat_control/).

| File | Contents |
|---|---|
| `run_log.txt` | Start time and duration of each stage, accuracy and correct counts, training records (seeds 42-44) |
| `paired_printout.txt` | Paired comparisons for seeds 42-44 (questions only one model got right, McNemar p), comparison of answers between P and the talk's PTQ |
| `eval_printouts.txt` | Output of `steps/4_evaluate.py` for each model (including accuracy per profession) |
| `train_printouts.txt` | Loss every 20 steps, and the record at the end of training |
| `summary.json` | Summary and verdicts built from the four files above and the talk's A and PTQ scoring files |

`summary.json` is generated by [`scripts/qat_control_summary.py`](../scripts/qat_control_summary.py). The script also checks that the records agree with each other: that the difference in correct counts equals the difference in the counts of questions only one model got right, and that each printed p equals a recomputed p. CI runs it with `--check` to confirm that `summary.json` matches the records.

To run it again, put the starting model in `outputs/0_start` and run the following for each seed (about 2.9 hours per seed on an H100 NVL).

```bash
hf download HyeonSang/Midm-2.0-KorMedMCQA --local-dir outputs/0_start
python steps/2_int4_ptq.py --src outputs/0_start --out outputs/qc_p && python steps/4_evaluate.py outputs/qc_p
python steps/3_int4_qat.py --no-fake-quant --seed 42 --src outputs/0_start --out outputs/qc_x42
python steps/4_evaluate.py outputs/qc_x42_bf16 && python steps/4_evaluate.py outputs/qc_x42
python steps/3_int4_qat.py --seed 42 --src outputs/0_start --out outputs/qc_q42 && python steps/4_evaluate.py outputs/qc_q42
```

After each stage, move the scoring files (`outputs/4_eval_*_test.json`) to a disk that is not wiped. This run kept these files on a disk that gets wiped, and lost them.
