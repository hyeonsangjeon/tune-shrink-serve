# quantization/ — code

[한국어](README.md) | **English**

All the code for training, INT4 conversion, scoring, serving measurement and API comparison is in this package. Run commands from the repo root as `python -m quantization.<module>`. How to run them in order is in the [step-by-step guide](../docs/guide/README_en.md).

## Folders

| Folder | What |
|---|---|
| [`configs/`](configs/) | Configs. The single file [`kormedmcqa_midm.yaml`](configs/kormedmcqa_midm.yaml) covers this whole experiment (model, data split, prompt, training values, INT4 method, results folder). `kormedmcqa_midm_unsloth.yaml` is for Unsloth training, and [`sweep/`](configs/sweep/) is for the experiments to raise the score |
| [`results/kormedmcqa_midm/`](results/kormedmcqa_midm/) | Result JSON. Every number in the docs comes from here |
| [`artifacts/`](artifacts/) | Where trained weights are created. Not committed |
| [`serving/`](serving/) | vLLM server launch script, online benchmark, and a client that sends one question |
| [`tests/`](tests/) | CPU tests (`python -m pytest -q`) |

## Modules

| Module | What it does | Step-by-step guide |
|---|---|---|
| `kormedmcqa.py` | Loading KorMedMCQA, splitting by year (train 2012-2020, 2021, 2022-2024), prompts, reading the answer number, same-question comparison (McNemar) and confidence intervals | 3 |
| `v2_run.py` | Runs GPU stages: `selftest`, `a` (LoRA training), `b` (INT4 PTQ), `c` (INT4 QAT), `merge` | 3, 4 |
| `v2_pipeline.py` | The actual implementation of training and conversion: chat template, loss only on answer tokens, transformers+PEFT and Unsloth, TorchAO INT4 | 3, 4 |
| `eval_vllm.py` | Scoring with vLLM (answer with the number only, option probabilities, write an explanation and then answer), adapter scoring | 3, 4, 7 |
| `v2_bench.py` | Offline throughput and single-request latency | 5 |
| `serving/v2_bench_serve.py` | Online server benchmark (concurrent requests, prefix caching on/off, `--fresh-per-level`) | 5 |
| `eval_managed.py` | Has managed API models answer the same questions (Entra ID, no key) | 6 |
| `managed_compare.py` | Compares API results with Mi:dm results on the same questions; summary table | 6 |
| `self_explain.py` | Has the untuned Mi:dm write explanations for the training questions itself, then filters them | 7 |
| `sweep_report.py` | Summarizes the experiments to raise the score on the 2021 questions: checkpoint selection, pass criteria | 7 |
| `publish_hf.py` | Uploads trained weights to Hugging Face with a model card | 4 |
| `data_korquad.py` | Reading the config (`load_config`, used by every stage) and loading KorQuAD for the earlier experiment | — |

**How to read the names.** `v2_` is a leftover name that means the second version of this code. `a`, `b`, `c` in stage and file names, and `A_bf16`, `B_int4_ptq`, `C_int4_qat`, are the BF16 LoRA model, INT4 PTQ and INT4 QAT, in that order.

## Modules used only for the earlier experiment (Qwen3-8B · KorQuAD)

`train_lora.py`, `eval_qa.py`, `bench_baseselect.py`, `splits.py`, `v2_report.py`. Their configs and results are in [`archive/qwen3_korquad/`](../archive/qwen3_korquad/), and the code's default config is `config.yaml` in that folder. For this experiment, always add `--config quantization/configs/kormedmcqa_midm.yaml`.
