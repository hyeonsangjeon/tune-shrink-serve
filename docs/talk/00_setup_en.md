# 0. Experiment setup — model, data, prompt, scoring, hardware

[한국어](00_setup.md) | **English**

This document lists the conditions shared by all results ① to ④. Each result document lists separately the conditions that apply only to that experiment.

## Summary in one paragraph

We trained KT's open model Mi:dm 2.0 Base (11.5B) with LoRA on past questions from the Korean national licensing exams for health professions (KorMedMCQA). It was trained on the 2012-2020 questions, and the 2021 questions were used only for a dev check. Final accuracy was measured on 3,009 questions from 2022-2024. The same model was shrunk to INT4 with TorchAO and served with vLLM, and speed and cost were also measured on the same single H100.

## Model

| Item | Value |
|---|---|
| Model | [`K-intelligence/Midm-2.0-Base-Instruct`](https://huggingface.co/K-intelligence/Midm-2.0-Base-Instruct) |
| Pinned commit | `35479c5fc9a18a5db7cc6dbadcf1db68db7beab0` (2025-10-29) |
| Size | 11.5B parameters, BF16 weights 21.51 GiB |
| Architecture | Llama family (`LlamaForCausalLM`), 48 layers, 8 KV heads, vocabulary of 131,384 |
| License | MIT |

Size is the sum of the safetensors file sizes reported by the Hugging Face Hub ([`artifact_sizes.json`](../../quantization/results/kormedmcqa_midm/artifact_sizes.json)). All sizes in this document are in GiB (1024³ bytes).

## Data

[`sean0042/KorMedMCQA`](https://huggingface.co/datasets/sean0042/KorMedMCQA) contains past 5-option multiple-choice questions from the national licensing exams for doctors, nurses, pharmacists and dentists (commit `79efd6f9`, CC BY-NC 2.0). The dataset is not in the repo; it is downloaded at run time. To show the format, one training question is reproduced below with its source.

| Use | HF split | Exam year | Questions | Config name in this repo |
|---|---|---|---|---|
| train | `train` | 2012-2020 | 3,401 | `train` |
| dev | `dev` | 2021 | 1,059 | `selection_dev` |
| test | `test` | 2022-2024 | 3,009 | `final_holdout` |

- The three sets share no exam year. The loader stops immediately if a question ID or a year overlaps (`kormedmcqa.assert_disjoint`).
- The 2022-2024 questions were measured **only once per model**. If the same result file already exists, the evaluation script does not measure again (`eval_vllm.py`).
- Accuracy is read against two reference points: guessing gives 20%, and the reference pass line for the national exams is 60%.

### What one question looks like

This is question 1 of session 1 of the 2019 nurse licensing exam (the training `train` split). Source: KorMedMCQA (arXiv 2403.01469), CC BY-NC 2.0.

| Field | Value |
|---|---|
| `subject` · `year` · `period` · `q_number` | `nurse` · 2019 · 1 · 1 |
| `question` | Which hypersensitivity reaction occurs when IgE antibodies to an antigen are produced and then mediators such as histamine are released? (translated from Korean) |
| `A` ~ `E` | Asthma / Contact dermatitis / Systemic lupus erythematosus / Organ transplant rejection / ABO-incompatible transfusion reaction (translated from Korean) |
| `answer` | `1` (option numbers 1-5) |

The dataset also has a `cot` field with worked solutions, but this training and evaluation did not use it. Training used 3,401 questions like this one.

## Prompt and answer format

Mi:dm's chat template is used as is. The template puts KT's fixed system text (about 450 tokens) and today's date in front of every request. Because a change of date changes the prompt, the date was fixed to `21 Nov 2026` (`template_kwargs.date_string` in the config file).

```text
[system]  당신은 한국 보건의료 국가시험 문제를 푸는 어시스턴트입니다. 보기 중 정답의 번호 하나만 숫자로 출력하세요. 설명은 쓰지 마세요.
[user]    다음은 간호사 국가시험 문제입니다. 보기 중 정답의 번호 하나만 숫자로 답하세요.

          항원에 대한 IgE 항체가 생성된 후 히스타민과 같은 매개물을 분비하여 발생하는 과민반응은?
          1. 천식
          2. 접촉피부염
          3. 전신홍반루푸스
          4. 장기이식 거부반응
          5. ABO 부적합 수혈반응
[model]   1
```

This block shows the example question from "What one question looks like" turned into a prompt. In the actual request, the fixed system text and the date from the Mi:dm template come before it.

- On the 2022-2024 questions, prompts average 654 tokens (counted by vLLM). The maximum training length was set to 1,024 tokens, and the training log shows no question truncated for exceeding this length.
- During training, loss is applied only to the answer number and the turn-closing token `<|eot_id|>`. For the example question, the part the model learns is just the two tokens `1<|eot_id|>`; no loss is applied to the prompt.
- Generation is greedy. The model's default sampling settings (temperature 0.8, top_p 0.7) are turned off, and generation stops at `<|eot_id|>` and `<|end_of_text|>`.

## Scoring

- The first of ①-⑤, 1-5 or A-E that appears in the model's answer is taken as the chosen option. If none can be found, the answer counts as wrong. In this measurement, 0 answers from the self-hosted GPU models had an option number that could not be read. Managed API models are covered separately in [4. Cost](04_cost_en.md#-comparison-with-managed-apis--slm-vs-api).
- The range next to an accuracy is the 95% confidence interval (Wilson). For 3,009 questions it is about ±1.7 percentage points (pt).
- To compare two models, we **pair them question by question** and use the exact McNemar test. It counts the questions only one of the two models got right and checks whether the difference could be chance. It catches smaller differences than just checking whether confidence intervals overlap.

## Hardware and software

| Item | Value |
|---|---|
| GPU | NVIDIA H100 NVL 94GB, 1 GPU (Azure `Standard_NC40ads_H100_v5`, on-demand) |
| Driver / CUDA | 580.178.04 / 13.0 |
| Training, quantization and evaluation environment | Python 3.10, torch 2.11.0+cu130, transformers 4.57.6, peft 0.20.0, trl 0.24.0, torchao 0.17.0, bitsandbytes 0.50.0, vLLM 0.23.0 |
| Unsloth training environment | A separate virtual environment. unsloth 2026.11.3, unsloth_zoo 2026.11.4, torch 2.14.1, transformers 4.57.6, peft 0.20.0, trl 0.24.0 |

The versions actually installed are recorded in [`env_h100.json`](../../quantization/results/kormedmcqa_midm/env_h100.json). All settings are in one file: [`quantization/configs/kormedmcqa_midm.yaml`](../../quantization/configs/kormedmcqa_midm.yaml).

## What this experiment as a whole does not show

- **There is a single seed, 42.** Every number on the 2022-2024 questions comes from models trained once with seed 42. Later, only the Unsloth training was run once more with seed 43; measured on the 2021 questions, it differed by 0.9-1.5 pt ([5. Raising the score](05_recipe_sweep_en.md)). To know properly how much the score changes with the seed, more runs are needed.
- **One GPU type (H100).** On other GPUs, INT4 speed in particular may differ.
- **The past exam questions are public.** We cannot check whether the untuned model saw these questions during pretraining. The untuned accuracy already being 63% may also be an effect of that.
- **This is not a model for medical advice.** Only the ability to pick answers to exam questions was measured.
