# Making a small model smaller and faster
[한국어](README.md) | **English**

**A record of measuring Unsloth LoRA · TorchAO INT4 · vLLM serving end to end on one model** — repo for a PyTorch Day Korea 2026 Serve Track talk

To tune a small model and use it in a service, you want to know three things: how much tuning improves it, how much it loses when you shrink it, and how fast and cheap it is to serve. This repo measures all three with the same model, the same questions and the same GPU. Last, it gives the same questions to 11 managed API models and puts them side by side with the small model (SLM) on a self-hosted GPU.

KT's open model **Mi:dm 2.0 Base (11.5B)** was trained on **past national licensing exam questions for doctors, nurses, pharmacists and dentists (KorMedMCQA)**. It was then shrunk to INT4 and served with vLLM. Every number below comes from the **3,009 questions of the 2022-2024 exams**. The self-hosted GPU numbers come from **one H100 NVL and a single seed**. The API numbers were measured once per model on the same questions through Azure AI Foundry.

- **LoRA tuning:** accuracy 63.3% → **72.3%**. The same training with Unsloth gave the same accuracy and took 35% less training time.
- **INT4 quantization (TorchAO):** size 21.51 GiB → **7.20 GiB**. Accuracy went down by 2.6-2.9 percentage points (pt). The difference between QAT and PTQ was +0.2 pt (p=0.67) in the talk run and +1.0 pt (p=0.02) in a run retrained with the same code, so we could not decide whether QAT is better than PTQ. Whether that difference comes from fake quantization or from the 600 extra training steps was measured separately in a [control experiment](docs/qat_control_en.md).
- **vLLM serving:** with 1-4 requests, INT4 is 20% faster. With many requests, BF16 is up to 9 times faster. With prefix caching off, INT4 is stuck at 2 questions per second.
- **SLM vs API:** on the same 3,009 questions, 10 of the 11 managed API models scored higher than the tuned Mi:dm (72.3%) (highest 98.4%). When all 3,009 questions are run at once, Mi:dm on a self-hosted H100 has the lowest cost per 1,000 questions, $0.0111 (computed at list price). On a server that takes one request at a time (concurrency 1, prefix caching off), it becomes $0.137, more than gpt-6-luna ($0.0836), which has 95.9% accuracy.
- **Extra, raising the score:** without any other model's outputs, we changed the training method (epochs, answer-option order, learning rate, profession mix, explanations written by Mi:dm itself) and compared the results on the 2021 questions. No method went beyond the difference seen when only the seed changed (0.9-1.5 pt), so the 2022-2024 questions were not measured again.

![Accuracy on the 3,009 questions from 2022-2024: untuned model, BF16 LoRA (transformers+PEFT, Unsloth), INT4 PTQ, INT4 QAT](docs/talk/figures/accuracy_test_en.svg)

## Conclusions in talk order

| Agenda | Conclusion | What this result does not show | Details |
|---|---|---|---|
| ① Unsloth LoRA | The same LoRA training went from 24.3 min → 15.7 min. GPU memory during training (`nvidia-smi`) went from 82.3 → 25.5 GiB. No difference in accuracy (p=0.89) | Unsloth could not read the Mi:dm tokenizer, so a workaround was needed. Memory is a single reading at one point in time | [01](docs/talk/01_finetune_unsloth_en.md) |
| ② TorchAO INT4 | Size cut to 1/3; accuracy −2.9 pt (PTQ), −2.6 pt (QAT). Both differences are significant in a same-question comparison. QAT is +0.2 pt over PTQ (p=0.67), and +1.0 pt (p=0.02) when retrained with the same code | Only a single QAT setting and a single INT4 scheme were tested. Whether QAT is better than PTQ could not be decided, because the two runs gave different results | [02](docs/talk/02_int4_quantization_en.md) |
| ③ vLLM serving | Prefix caching, which reuses only the prefix shared by every request, cut INT4's time to first token (TTFT) from 542 → 110 ms. Even so, when requests pile up, BF16 is ahead. Online, it is about 3 times ahead with prefix caching on for both; offline, tile-packed INT4 stops in the 800s of tokens per second while BF16 goes up to 7,600 tokens | Prefix caching was measured only for reuse of the shared prefix; repeated identical questions were not measured | [03](docs/talk/03_vllm_serving_en.md) |
| ④ Cost (SLM vs API) | When all 3,009 questions are run at once, cost per 1,000 questions is $0.011 for BF16 and $0.172 for INT4 (computed at the H100 on-demand list price). On the same questions, 10 of the 11 managed API models were more accurate than Mi:dm, and some API models are also cheaper than the cost of a server that takes one request at a time | Costs are computed at list price and on different bases (GPU time vs tokens). The managed models either have undisclosed sizes or are much larger than Mi:dm, and their training data is unknown | [04](docs/talk/04_cost_en.md) |
| Extra: raising the score | Of the experiments that changed the training method without other models' outputs, none passed the bar (+2.0 pt, p<0.05) on the 2021 questions. The highest value (72.0%) came from a run that kept the recipe and changed only the seed | Each experiment was run once, and the pick was made after looking at the 2021 score 31 times. Methods that use other models' outputs were not measured | [05](docs/talk/05_recipe_sweep_en.md) |

The shared conditions (model, data, prompt, scoring, hardware) are in [00_setup.md](docs/talk/00_setup_en.md). Every number comes from the result JSONs in [`quantization/results/kormedmcqa_midm/`](quantization/results/kormedmcqa_midm/). If a number in the docs and the JSON disagree, CI fails ([evidence index](docs/EVIDENCE.md)). This check reads the Korean docs; the `_en.md` files are translations of them and are not checked by CI.

## Where to go

| What you want to do | Where to go |
|---|---|
| See the results and evidence in detail | [`docs/talk/`](docs/talk/) 00-05 |
| See the talk slides | [PDF, Korean](docs/talk/slides/pytorchday2026_slides_ko.pdf) (as presented on 2026-11-21) · [PDF, English](docs/talk/slides/pytorchday2026_slides_en.pdf) (translation) |
| Check that this repo works, without a GPU | “1-minute check” just below |
| Read the training, INT4 and serving code and run it one step at a time | [`steps/`](steps/) 1-5 (one file per step, an explanation for each function) |
| Rerun from scratch on a GPU | [step-by-step guide](docs/guide/README_en.md) 1-8 (for each step: commands, time taken, what to check, what to do when it fails) |
| See what is in which folder | “Repo layout” below |
| Only use the trained models | [`HyeonSang/Midm-2.0-KorMedMCQA`](https://huggingface.co/HyeonSang/Midm-2.0-KorMedMCQA), [`…-QAT-bf16`](https://huggingface.co/HyeonSang/Midm-2.0-KorMedMCQA-QAT-bf16) |

## 1-minute check (CPU only, no accounts)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/cpu.txt
python -m pytest -q
python scripts/build_evidence_index.py --check
python scripts/make_figures.py --check
python -m quantization.managed_compare --check
```

**What this 1 minute checks:**
- Whether the data split, prompt, scoring and serving benchmark code pass the tests
- Whether the numbers in the docs match the result JSONs
- Whether the charts are redrawn identically from the result JSONs
- Whether the managed API comparison table is rebuilt identically from the result JSONs

**What it does not check:** it does not redo training, quantization or serving on a GPU. It does not call the managed APIs. It does not download the model or the data either. The training, quantization and serving code is in [`steps/`](steps/), and how to run it on a GPU is just below.

## Rerunning on a GPU

One H100 (or an 80GB-class GPU) is enough. There are two ways.

**One step at a time, reading the code:** [`steps/`](steps/) puts training, INT4, scoring and serving into one Python file per step. Settings are at the top of each file, and each function has comments on what it does and why.

```bash
source ~/venv-q/bin/activate                  # virtual environment created by bootstrap.sh below
python steps/1_lora_train.py                  # LoRA training (transformers+PEFT; --engine unsloth also works)
python steps/2_int4_ptq.py                    # INT4 PTQ (TorchAO)
python steps/3_int4_qat.py                    # INT4 QAT (TorchAO)
python steps/4_evaluate.py outputs/2_int4_ptq # accuracy on the 3,009 questions from 2022-2024 (vLLM)
bash steps/5_serve.sh outputs/2_int4_ptq      # vLLM server. In another terminal: python steps/5_ask.py
```

On 2026-11-16~17, this code was used on an H100 NVL to retrain from step 1 and score again. Accuracy matched the talk to within 0.4 pt (BF16 LoRA 72.3%, Unsloth 72.5%, INT4 PTQ 69.1%, INT4 QAT 70.1%), and training times differed by less than 1 minute. We also checked that training in steps 1 and 3 can be killed partway and continued with `--resume`. Details are in [`steps/README.md`](steps/README_en.md).

**In the same order that produced the talk numbers:** the `stage.sh` stages below also do the dev check, resume interrupted training, measure serving speed and save the result JSONs. For each stage, the commands, what to check and what to do when it fails are in the [step-by-step guide](docs/guide/README_en.md). The overall flow, and the time each stage took in this run:

```bash
bash scripts/gpu/bootstrap.sh                 # install 2 virtual environments, check driver, GPU and tests (about 6 min)
bash scripts/gpu/stage.sh selftest            # check that QAT's fake quantization matches the served INT4
bash scripts/gpu/stage.sh base_dev            # untuned model, 2021 questions
BACKEND=hf bash scripts/gpu/stage.sh a        # LoRA training, transformers+PEFT (24 min)
CONFIG=quantization/configs/kormedmcqa_midm_unsloth.yaml bash scripts/gpu/stage.sh a   # same training, Unsloth (16 min)
bash scripts/gpu/stage.sh a_dev               # dev check on the 2021 questions (continues only if at least 2 pt above the untuned model)
bash scripts/gpu/stage.sh b                   # INT4 PTQ (17 s)
bash scripts/gpu/stage.sh c                   # INT4 QAT (1 h 42 min)
bash scripts/gpu/stage.sh final_A             # 2022-2024 questions, once per model (also final_base, final_B, final_C)
bash scripts/gpu/stage.sh bench_A             # offline throughput (also bench_B, bench_C)
FRESH=1 bash scripts/gpu/stage.sh online_A    # online serving, prefix caching on/off, new questions for each concurrency level (also online_B)
```

Any stage runs in the background if you add `--detach`. Check progress with `bash scripts/gpu/status.sh`.

## Repo layout

```
README.md                  this file. Summary of the results and the starting point
steps/                     code by step: LoRA training → INT4 PTQ → INT4 QAT → accuracy → vLLM serving. One step per file
docs/
  talk/                    result docs 00-05 (in talk order), charts (figures/), talk slides as PDF (slides/)
  guide/                   step-by-step guide 1-8: check without a GPU → GPU VM → training → INT4 → serving → API comparison → raising the score → when it fails
  EVIDENCE.md              evidence index: for each number in the docs, which value in which result JSON (generated from evidence_ledger.yaml)
  DATA_AND_LICENSES.md     licenses for the model, data and results, and what was not put in the repo
quantization/              code (folder guide in quantization/README.md)
  configs/                 settings. The single file kormedmcqa_midm.yaml covers this whole experiment; sweep/ holds the score-raising experiments
  results/kormedmcqa_midm/ result JSONs (every number in the docs comes from here)
  serving/                 vLLM server launch scripts and the online benchmark
  tests/                   CPU tests
requirements/              install files: cpu · gpu · unsloth · slides (requirements/README.md)
scripts/                   GPU VM scripts (gpu/), building the charts, slides and evidence index (scripts/README.md)
archive/qwen3_korquad/     earlier experiment (Qwen3-8B · KorQuAD): results, notebooks, settings. Not compared with the main experiment
```

## Earlier experiment (Qwen3-8B · KorQuAD)

This repo first ran the same comparison (BF16 LoRA, INT4 PTQ, INT4 QAT) with Qwen3-8B and KorQuAD (A100, 3 seeds). Those results are kept as they were in [`archive/qwen3_korquad/README.md`](archive/qwen3_korquad/README.md). The two experiments differ in both the model and the data, so they are not compared causally with each other.

## Data and licenses

- **Code and result JSONs:** [MIT](LICENSE).
- **Model and data:** not in the repo; they are downloaded at run time. To show the format, the docs include only one training question, with its source ([example](docs/talk/00_setup_en.md#what-one-question-looks-like)). Mi:dm 2.0 Base is MIT, KorMedMCQA is CC BY-NC 2.0, and KorQuAD 1.0 is CC BY-ND 2.0 KR.
- **Weights trained on KorMedMCQA:** published on Hugging Face. Under the data license, they may be used for non-commercial purposes only. The model is not for medical advice.
- **Weights trained on KorQuAD:** not published.

Details are in [`docs/DATA_AND_LICENSES.md`](docs/DATA_AND_LICENSES.md).

The code in this repo started from `pdf_qa_extraction/quantization/` (commit `760a92c`) in [`hyeonsangjeon/PDF2LLM-Tuning-Studio`](https://github.com/hyeonsangjeon/PDF2LLM-Tuning-Studio).
