# Code for each step: training → INT4 → scoring → serving
[한국어](README.md) | **English**

This folder puts the talk's training, INT4 quantization, scoring and vLLM serving into **one Python file per step**. Each file has its settings at the top, and below them the functions in the order they run. The `main()` at the bottom only calls those functions in turn. Each function has comments on what it does and why.

| Step | File | What it does | Key API | Time on H100 NVL |
|---|---|---|---|---|
| 1 | [`1_lora_train.py`](1_lora_train.py) | LoRA training, then merging into one BF16 model | PEFT `LoraConfig` and `get_peft_model`, Unsloth `FastLanguageModel`, `Trainer` | 24 min (Unsloth 16 min) |
| 2 | [`2_int4_ptq.py`](2_int4_ptq.py) | Converting to INT4 without training (PTQ) | TorchAO `Int4WeightOnlyConfig` + transformers `TorchAoConfig` | 17 s |
| 3 | [`3_int4_qat.py`](3_int4_qat.py) | Training further while mimicking INT4, then converting to INT4 (QAT) | TorchAO `quantize_` + `QATConfig(step="prepare" / "convert")` | 1 h 42 min |
| 4 | [`4_evaluate.py`](4_evaluate.py) | Measuring accuracy on the 3,009 questions from 2022-2024 | vLLM `LLM.generate` | 1-6 min per model |
| 5 | [`5_serve.sh`](5_serve.sh), [`5_ask.py`](5_ask.py) | Launching an OpenAI-compatible server and sending questions | `vllm.entrypoints.openai.api_server` | — |

Shared pieces (model name, folders, data, prompt, INT4 format) are in [`common.py`](common.py).

## Running

Run these on a GPU VM (1 GPU with 80GB or more), from the repo's top-level folder. At least 200GB of disk must be free. One model is 22GB, and each checkpoint that step 3 leaves while it trains is 42GB. While a checkpoint is being saved, three checkpoints exist at once, so the training folder grows to 123GB (the checkpoints are deleted when the step finishes). The virtual environment is created by `bash scripts/gpu/bootstrap.sh` in [step-by-step guide 2](../docs/guide/02_gpu_setup_en.md). The model and the data are public, so they download without a Hugging Face login.

```bash
source ~/venv-q/bin/activate

python steps/1_lora_train.py                  # → outputs/1_lora
python steps/2_int4_ptq.py                    # → outputs/2_int4_ptq
python steps/3_int4_qat.py                    # → outputs/3_int4_qat

python steps/4_evaluate.py                    # score the LoRA model
python steps/4_evaluate.py outputs/2_int4_ptq
python steps/4_evaluate.py outputs/3_int4_qat
python steps/4_evaluate.py base               # untuned model

bash steps/5_serve.sh outputs/2_int4_ptq      # server (stop with Ctrl-C)
python steps/5_ask.py                         # 3 questions, from another terminal
```

Each step's default input folder is the previous step's output folder, so running the steps in order without arguments chains them together. Models are written to `outputs/` and are not committed to git.

**A short run first:** train for only 5 steps and score only 50 questions, to see that the whole pipeline runs through without errors. Downloading the model (about 23GB), loading it and saving still take their full time, and disk use still peaks at about 120GB. This is because even 5 steps of training save one checkpoint when training ends.

```bash
python steps/1_lora_train.py --max-steps 5
python steps/2_int4_ptq.py
python steps/3_int4_qat.py --max-steps 5
python steps/4_evaluate.py outputs/2_int4_ptq --split dev --limit 50
```

**Training with Unsloth:** Unsloth is in a different virtual environment. Its output goes separately to `outputs/1_lora_unsloth`.

```bash
source ~/venv-unsloth/bin/activate
python steps/1_lora_train.py --engine unsloth
```

## Checking against the talk's numbers

Step 4 gives the accuracy. On 2026-11-16 to 17, on the same H100 NVL and in the same virtual environment as for the talk, we retrained from step 1 with the code in this folder and scored on the 3,009 questions from 2022-2024 (seed 42, once per model). The table below shows the talk's values next to these values. "Same answer" is the number of questions where the two models chose the same option.

| Model | Talk accuracy (correct) | Retrained with this folder's code (correct) | Same answer | Training time (talk → rerun) |
|---|---|---|---|---|
| Untuned model (`base`) | 63.3% (1,904) | 63.3% (1,904) | 3,009 | no training |
| Step 1 BF16 LoRA (transformers+PEFT) | 72.3% (2,176) | 72.3% (2,177) | 2,686 | 24.3 min → 24.7 min |
| Step 1 BF16 LoRA (Unsloth) | 72.2% (2,173) | 72.5% (2,182) | 2,732 | 15.7 min → 16.5 min |
| Step 2 INT4 PTQ | 69.5% (2,090) | 69.1% (2,078) | 2,596 | no training |
| Step 3 INT4 QAT | 69.7% (2,097) | 70.1% (2,109) | 2,612 | 101.9 min → 102.9 min |

The untuned model has no training, so on 2026-11-16 it was only scored again. INT4 PTQ and INT4 QAT were made from the retrained transformers+PEFT model. The talk's results are in [`eval_vllm_*_seed42_h100.json`](../quantization/results/kormedmcqa_midm/), and the retrained results are in [`steps_rerun/`](../quantization/results/kormedmcqa_midm/steps_rerun/).

- **Retraining from scratch changed the answer on 9-14% of the questions. Accuracy moved within 0.4 percentage points (pt).** In correct counts, that is +1, +9, −12 and +12. The McNemar p from comparing the same questions was 0.5 or higher in every case ([`paired_compare.json`](../quantization/results/kormedmcqa_midm/steps_rerun/paired_compare.json)). The 95% confidence interval for one model is about ±1.6 pt, so we treat differences of this size as the same result.
- **We see the changed answers as variation between runs, not as a code difference.** The first step's loss matched the talk (PEFT 17.3403, Unsloth 17.3405), but after that it drifted slightly at every step (at most 0.049 from the talk). Two trainings on the same day with this folder's code also differed in loss by up to 0.019 over the first 110 steps. The trained model files also differed from the talk's files byte for byte. The most plausible reason is that the order of GPU operations is not the same every time, but we did not measure this separately.
- **The drop from BF16 to INT4 was similar to the talk.** Against the retrained BF16 LoRA model, INT4 PTQ is −3.3 pt and INT4 QAT is −2.3 pt (talk: −2.9 pt, −2.6 pt). The McNemar p from comparing the same questions was below 0.001 for both.
- **The difference between INT4 QAT and INT4 PTQ varied from run to run.**

  | Comparison (same questions) | Accuracy difference | Questions only one side got right | McNemar p |
  |---|---|---|---|
  | Talk: INT4 PTQ vs INT4 QAT | +0.2 pt | 97 vs 104 | 0.67 |
  | Retrained: INT4 PTQ vs INT4 QAT | +1.0 pt | 70 vs 101 | 0.02 |
  | Retrained: INT4 PTQ vs INT4 QAT that was stopped and resumed | +0.6 pt | 79 vs 98 | 0.18 |

  QAT was higher all three times, but the difference ranged from 0.2 to 1.0 pt, and p was below 0.05 only once. The bottom two rows use the same BF16 LoRA model and the same INT4 PTQ model. McNemar holds the two models fixed and looks only at variation that comes from the questions. It does not include the variation from training again. So a single p=0.02 does not let us say "QAT is better than PTQ", and the talk's p=0.67 does not let us say "they are the same" either.
- **Scoring the same INT4 file again changes the answers on a few questions.** On 2026-11-16, we converted the talk's BF16 LoRA model to INT4 again with step 2, and the file was byte-identical to the talk's INT4 file. Scoring that file again changed the answers on 5 questions and lowered the correct count by 2 (69.458% → 69.392%). The untuned model and the BF16 LoRA model gave the same answers on all 3,009 questions when scored again. We did not separately measure why answers change only with INT4.
- **The untuned model's score should be almost the same.** This is because there is no training, and the model revision, the prompt and temperature 0 are the same. A different GPU or vLLM version may change a few questions. If it differs by more than 1 pt, check the prompt or the library versions first.
- **Trained models may differ as much as shown above.** In the talk's experiments, changing only the seed moved the score on the 2021 questions by 0.9-1.5 pt ([5. Raising the score](../docs/talk/05_recipe_sweep_en.md)). We retrained only this once (twice if you count the training that was stopped and resumed), so this result cannot set the range of run-to-run variation.
- The versions installed for the talk are in [`env_h100.json`](../quantization/results/kormedmcqa_midm/env_h100.json). The versions for the retraining were the same (`versions` in [`compare.json`](../quantization/results/kormedmcqa_midm/steps_rerun/compare.json)).

## Stopping and resuming training (`--resume`)

On 2026-11-16 to 17, we force-killed step 1 (transformers+PEFT) and step 3 in the middle of training (SIGKILL, the same as `kill -9`), then added `--resume` to the same command and ran them to the end. Values in parentheses are from the training that ran to the end with the same code without being stopped ("Retrained with this folder's code" in the table above).

| | Step 1 BF16 LoRA | Step 3 INT4 QAT |
|---|---|---|
| Stopped at | step 110 (425 s after start) | step 160 (1,685 s after start) |
| Checkpoint left | `checkpoint-100` | `checkpoint-150` |
| Resumed span and time taken | step 100 → 426, 18.8 min | step 150 → 600, 77.3 min |
| First loss logged after resuming | 1.6023 (1.6004), step 110 | 1.3998 (1.3825), step 160 |
| Logged learning-rate values that matched the uninterrupted training | 32 of 32 | 23 of 23 |
| Final loss | 1.4373 (1.4348) | 1.3145 (1.3151) |
| Accuracy | 72.6% (72.3%) | 69.7% (70.1%) |
| Same answer | 2,715 / 3,009 | 2,907 / 3,009 |
| McNemar p on the same questions | 0.62 | 0.17 |

- **Training resumed from the checkpoint.** The learning rate matched the uninterrupted training at every logged step. The first loss logged after resuming also differed from the uninterrupted training by only 0.002 (step 1) and 0.017 (step 3). If training had started over from the beginning, the step 1 loss would have restarted near 17.3, the first-step value.
- **The gap between the resumed and the uninterrupted training is about the same size as the gap between two trainings.** Loss differed by at most 0.058 (step 1) and 0.017 (step 3). Two uninterrupted trainings also differed in step 1 loss by up to 0.049 (this folder vs the talk). The accuracy differences are also in the same range as the retraining differences in the table above (within 0.4 pt).
- In step 3, both trainings start from the same BF16 LoRA model. We think this is why 2,907 questions have the same answer, more than in step 1 (2,715).
- Both times, the checkpoint folder (`outputs/*_run`) was deleted after training finished.
- **Step 3 checkpoint disk use:** during the resumed step 3 training, we measured the training folder size every 30 s. One checkpoint was 42GB, and the training folder peaked at 123GB. This is because training keeps at most two checkpoints (`save_total_limit=2`), but three exist at once while a new checkpoint is being saved.

## How this folder relates to the code behind the talk's numbers

The talk's numbers were produced by [`quantization/v2_run.py`](../quantization/v2_run.py), [`v2_pipeline.py`](../quantization/v2_pipeline.py) and [`eval_vllm.py`](../quantization/eval_vllm.py), called by [`scripts/gpu/stage.sh`](../scripts/gpu/stage.sh). That code also carries the options for the score-raising experiments, the dev checks and the naming rules for result files, so it is long and hard to read. This folder rewrites the same work so that it is easy to read.

- **What is the same:** data splitting, the prompt, building the training data (loss only on the answer token) and scoring call that code's functions directly ([`common.py`](common.py)). CI checks that settings such as the learning rate, LoRA rank, batch and QAT steps match the YAML config ([`test_steps.py`](../quantization/tests/test_steps.py)).
- **What was checked on GPU (H100 NVL, the same virtual environment as for the talk):**
  - 2026-11-16, short runs: step 1 for 5 training steps each with transformers+PEFT and with Unsloth, and step 3 for 5 training steps from the talk's BF16 LoRA model. The first step's loss matched the talk's training (PEFT 17.3403, Unsloth 17.3405, QAT 1.3766). This means the training questions, their order, the LoRA initial values and the loss computation are the same. Step 2 converted the talk's BF16 LoRA model to INT4 in 16.6 s (21.52 → 7.20 GiB), and the file was byte-identical to the talk's INT4 file.
  - 2026-11-16 to 17, start to finish: step 1 (426 training steps, both engines) and step 3 (600 training steps) were trained to the end, followed by steps 2 and 4. The results are in the table above. `--resume` for steps 1 and 3 was also checked (section above).
  - The step 5 server was launched with the retrained INT4 QAT model and sent 3 questions. The server's answers (2, 5, 1) matched the answers to the same questions in step 4 scoring. This was true both for the server with prefix caching on and for the server with it off (`NO_PREFIX_CACHE=1`). The 3 questions are only there to see that the server works, so the correct count (1/3) should not be read as an accuracy.
- **What was checked on CPU:** steps 1 and 3 were run with a small dummy model (Mi:dm tokenizer, 2 layers). We also checked that a training stopped at training step 2 continues to training step 4 with `--resume`.
- **Retraining records ([`steps_rerun/`](../quantization/results/kormedmcqa_midm/steps_rerun/)):** files with `r1_` or `r3_` in their names are from the trainings that were stopped and resumed.
  - `1_lora_train*.json`, `2_int4_ptq.json`, `3_int4_qat.json`, `r1_lora_train.json`, `r3_qat.json`: the record each step left (training time, final loss, size)
  - `4_eval_*_test.json`: the correct answer and the model's answer for each question (no question text)
  - `loss_logs.json`: loss and learning rate logged during training
  - `compare.json`: values checked against the talk's records, the comparison with the stopped-and-resumed trainings, checkpoint disk use, versions
  - `paired_compare.json`: comparison on the same questions (can be rebuilt with `python -m quantization.kormedmcqa compare <result 1> <result 2>`)

## When it does not work

- **Unsloth cannot read the Mi:dm tokenizer** (`'AddedToken' object has no attribute 'keys'`): `load_with_unsloth()` in `1_lora_train.py` already works around this. If it still happens, check that the Unsloth version matches [`requirements/unsloth.txt`](../requirements/unsloth.txt).
- **Disk full** (`No space left on device`): check the free space with `df -h`. Step 3 needs at least 150GB free when it starts. After you free up space, add `--resume` to the same command, and step 1 or step 3 training continues from the last checkpoint.
- **Training stopped partway:** add `--resume` to the same command. Checkpoints are saved every 100 training steps in step 1 and every 150 training steps in step 3. Training continues from the step after the last checkpoint, so any training done after that checkpoint is done again (see [Stopping and resuming training](#stopping-and-resuming-training---resume) above).
- **GPU out of memory in step 3:** run with `--batch-size 2 --grad-accum 8`. Each training step still covers 16 questions.
- **Error when loading an INT4 folder:** INT4 models are saved with pickle, so the torch and torchao versions must be the same when saving and when loading. Create and load them in the same virtual environment.
- **Step 5 server does not start:** use `nvidia-smi` to check whether another vLLM server is using the same GPU or port (8000). Change the port with `PORT=8001 bash steps/5_serve.sh`.
- Other known pitfalls are in [step-by-step guide 8](../docs/guide/08_troubleshooting_en.md).
