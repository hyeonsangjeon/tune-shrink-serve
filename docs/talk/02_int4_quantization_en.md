# 2. INT4 quantization — TorchAO PTQ and QAT

[한국어](02_int4_quantization.md) | **English**

Shared conditions are in [0. Experiment setup](00_setup_en.md). The starting point is the BF16 LoRA model trained with transformers+PEFT in [1. Fine-tuning](01_finetune_unsloth_en.md).

## What was changed

The same BF16 LoRA model was shrunk to INT4 in two ways. Both use the same storage format.

| | INT4 PTQ (post-training quantization) | INT4 QAT (quantization-aware training) |
|---|---|---|
| What it does | Converts only the weights to INT4, with no training | Trains further with fake quantization inserted, then converts to INT4 |
| TorchAO settings | `Int4WeightOnlyConfig(group_size=128)`, format `TILE_PACKED_TO_4D` | Training: `QATConfig(step="prepare")` → train → `QATConfig(step="convert")`. Saved with the same settings as PTQ |
| Extra training | None | 600 steps, learning rate 1e-5, batch 4 × accumulation 4, 8-bit AdamW, gradient checkpointing |
| Weights trained | — | All linear layers except the output layer (lm_head), 10,469M weights |
| Time taken | 17 s (load, convert, save) | 1 h 42 min of training (6,111 s) |

- **Only the weights are INT4.** Computation is done in BF16 (weight-only). There is one scale for every 128 weights.
- **`TILE_PACKED_TO_4D` uses the tinygemm INT4 kernel that ships with PyTorch.** No separate kernel library needs to be installed. This format determines the speed results in ③.
- **The two INT4 models are exactly the same size (7.20 GiB).** The QAT code explicitly excludes the output layer. Because PTQ gives the same size, we take it that the same part is excluded from quantization (we did not compare this directly).
- **Before training, we checked that QAT's fake quantization works the same way as the INT4 actually used in serving** ([`qat_selftest.json`](../../quantization/results/kormedmcqa_midm/qat_selftest.json)).
- **On top of being quantization-aware, QAT trains 600 more steps on the same training data** (about 2.8 epochs). This difference is also part of any comparison with PTQ.

> **Measurement conditions**
>
> - Run: 2026-11-10 to 11-11, H100 NVL 94GB, 1 GPU, seed 42, 1 run per model
> - Evaluation: 3,009 questions from 2022-2024, only once per model. vLLM 0.23; INT4 served with `quantization="torchao"`
> - The INT4 models were built on the same GPU and the same software as the evaluation, because an INT4 file is tied to the torch and torchao builds that made it (pickle format)
> - Result files: `eval_vllm_*_seed42_h100.json`, [`compare_final_seed42_h100.json`](../../quantization/results/kormedmcqa_midm/compare_final_seed42_h100.json), [`artifact_sizes.json`](../../quantization/results/kormedmcqa_midm/artifact_sizes.json)

## Results

![Accuracy on 3,009 questions from 2022-2024: untuned, BF16 LoRA from both engines, INT4 PTQ, INT4 QAT](figures/accuracy_test_en.svg)

| Model | Accuracy (95% confidence interval) | Size |
|---|---|---|
| Untuned Mi:dm 2.0 Base | 63.3% (61.5-65.0) | 21.51 GiB |
| BF16 LoRA | **72.3%** (70.7-73.9) | 21.51 GiB |
| INT4 PTQ | 69.5% (67.8-71.1) | **7.20 GiB** |
| INT4 QAT | 69.7% (68.0-71.3) | **7.20 GiB** |

| Comparison (paired by question) | Accuracy difference | Questions only one side got right | McNemar p |
|---|---|---|---|
| BF16 LoRA vs INT4 PTQ | −2.9 percentage points (pt) | 198 vs 112 | 0.000001 |
| BF16 LoRA vs INT4 QAT | −2.6 pt | 205 vs 126 | 0.000017 |
| INT4 PTQ vs INT4 QAT | +0.2 pt | 97 vs 104 | 0.67 |

"Questions only one side got right" gives the number that only the first-listed model got right vs the number that only the second-listed model got right. In the first row, 198 questions were right only for BF16 LoRA and 112 were right only for INT4 PTQ.

Accuracy differences were computed from unrounded values. So subtracting the rounded accuracies in the results table may give a value that differs by 0.1 pt (e.g. 72.3 − 69.5 = 2.8, while the unrounded difference is 2.86).

### Accuracy by profession

3,009 questions from 2022-2024, in %. Differences (pt) in the text were computed from unrounded values.

| Profession (questions) | Untuned | BF16 LoRA | INT4 PTQ | INT4 QAT |
|---|---|---|---|---|
| Doctor (435) | 56.3 | 72.2 | 68.0 | 69.7 |
| Nurse (878) | 73.2 | 81.8 | 78.8 | 78.8 |
| Pharmacist (885) | 67.7 | 74.7 | 70.6 | 71.1 |
| Dentist (811) | 51.4 | 59.6 | 58.8 | 58.3 |

### Accuracy by exam year

| Exam year | Untuned | BF16 LoRA | INT4 PTQ | INT4 QAT |
|---|---|---|---|---|
| 2022 | 63.9 | 72.1 | 70.7 | 70.2 |
| 2023 | 64.4 | 74.5 | 70.8 | 71.2 |
| 2024 | 61.4 | 70.2 | 66.8 | 67.5 |

**Observed.**
- Converting to INT4 makes the size 1/3 (21.51 → 7.20 GiB).
- In exchange, accuracy dropped by 2.6-2.9 pt. Even when paired question by question, this difference is hard to attribute to chance.
- In this run, QAT did not improve on PTQ (+0.2 pt, p=0.67). When training was redone with the same code, the result was different (the first item under "What this result does not show" below).
- Both INT4 models are more than 6 pt above the untuned model and above the 60% reference pass line.
- How much accuracy dropped differs by profession. Compared with BF16 LoRA, the doctor, nurse and pharmacist questions dropped by 2.5-4.1 pt, and the dentist questions dropped by only 0.7-1.2 pt.

**Plausible reason.** Weight-only INT4 rounds each group of 128 weights with one scale. The simplest explanation is that this rounding error changed some of the answer choices. There may be several reasons why QAT did not make up for this error in this run.

- A learning rate of 1e-5 with 600 steps may not have been enough for recovery.
- For a model already fitted with LoRA, further training may have had little left to gain.

This measurement cannot tell which of these it is.

**What this result does not show.**
- **We cannot decide whether QAT is better than PTQ.** When training was redone from LoRA onward with the [`steps/`](../../steps/README_en.md) code on 2026-11-16 to 11-17, QAT was 1.0 pt above PTQ (questions only one side got right: 70 vs 101, McNemar p=0.02). When QAT on the same BF16 LoRA model was stopped mid-training and then resumed, the difference was +0.6 pt (p=0.18). McNemar holds the two models fixed and looks only at the variation that comes from the questions. It does not include the variation from retraining, so we can say neither "the same" from this run's p=0.67 nor "better" from the retrained run's p=0.02. In the [control experiment](../qat_control_en.md) run separately on 2026-11-17 after the slides were made (same BF16 LoRA model, 3 seeds), the mean QAT − PTQ was +0.42 pt, and we still could not decide. However, a model trained for the same extra 600 steps without fake quantization was almost the same as PTQ after conversion to INT4 (mean −0.13 pt). The amount by which QAT came out above PTQ is not explained by the extra training.
- **We cannot generalize to "QAT has no effect".** We did not vary the learning rate, the number of steps, or the part of the model QAT is applied to.
- **No causal comparison with the earlier experiment.** In the earlier experiment (Qwen3-8B, KorQuAD), QAT recovered more F1 than PTQ did. Between then and now, the model and the data changed together, so the different result cannot be attributed to either one.
- **Only one INT4 method was measured.** It is weight-only, group 128, tile-packed format. GPTQ, AWQ and methods that also shrink activations (W4A8 etc.) were not measured.
- **A single seed, one GPU type.**

## How to rerun

Use the environment from [step-by-step guide 2](../guide/02_gpu_setup_en.md) (the checks for each stage are in [step-by-step guide 4](../guide/04_int4_en.md)), and run the following once the BF16 LoRA model exists.

```bash
CFG=quantization/configs/kormedmcqa_midm.yaml
python -m quantization.v2_run selftest --config $CFG            # check that QAT fake quantization works the same way as serving INT4
python -m quantization.v2_run b --config $CFG --seed 42         # INT4 PTQ
python -m quantization.v2_run c --config $CFG --seed 42         # INT4 QAT (saved as INT4 after training)
python -m quantization.eval_vllm --config $CFG --method B_int4_ptq   # 2022-2024 questions, only once
python -m quantization.eval_vllm --config $CFG --method C_int4_qat
python -m quantization.kormedmcqa compare <model 1 result JSON> <model 2 result JSON>   # McNemar, paired by question
```

To evaluate on a different GPU, do not move the INT4 files; build them again on that GPU. There are two ways.

- **INT4 PTQ:** download the BF16 LoRA model and rebuild it with `v2_run b`.
- **INT4 QAT:** download the BF16 weights from finished QAT training ([`HyeonSang/Midm-2.0-KorMedMCQA-QAT-bf16`](https://huggingface.co/HyeonSang/Midm-2.0-KorMedMCQA-QAT-bf16)) and build it with `v2_run c --export-only`.
