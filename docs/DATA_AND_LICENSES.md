# Data & Licenses

Provenance and license notes for every third‑party asset this repository uses.

> **Scope of the root [`LICENSE`](../LICENSE) (MIT).** MIT covers **this
> repository's own source code and its committed result JSON only**. It does
> **not** relicense the external dataset or model weights listed below, and it
> does **not** automatically cover fine‑tuned model artifacts you generate under
> `quantization/artifacts/` (those inherit their base model and training‑data
> terms).

> **Enforced coverage.** A machine‑checkable mirror of this file lives in
> [`docs/asset_ledger.yaml`](asset_ledger.yaml).
> `python scripts/check_asset_ledger.py --check` (a CI gate) fails if any
> committed bundled asset (PDF/font/image) lacks an entry. The only committed
> images are the generated talk charts in `docs/talk/figures/` (MIT, regenerated
> from the result JSON by `scripts/make_figures.py`). The only committed PDFs are
> the talk slides in `docs/talk/slides/` (MIT; slide 4 quotes the same KorMedMCQA item
> as `docs/talk/00_setup.md`, with attribution; the English PDF shows it translated).

## 1. Datasets (downloaded at runtime — never committed)

| Asset | Where | License | Redistribution note |
|---|---|---|---|
| **KorQuAD 1.0** (`KorQuAD/squad_kor_v1`) | Hugging Face Hub, fetched at run time by `quantization/data_korquad.py` | **CC BY‑ND 2.0 KR** (Attribution‑**NoDerivatives**) | The dataset is **not** copied into this repo. The **ND** clause governs redistributing *modified copies of the dataset*; it is a **known‑unknown** for redistributing models fine‑tuned on it — see §4. |
| **KorMedMCQA** (`sean0042/KorMedMCQA`) | Hugging Face Hub, fetched at run time by `quantization/kormedmcqa.py` | **CC BY‑NC 2.0** per the HF dataset card (the paper's datasheet says MIT; the stricter card is followed) | Not copied into this repo, except one training item (nurse, 2019, session 1, Q1) quoted with attribution in `docs/talk/00_setup.md` (and on slide 4 of the talk slides in `docs/talk/slides/`) to show the format. Committed eval JSON holds only item ids, gold/pred option numbers and the first 40 chars of model output — no question text. Weights fine‑tuned on it are published **for non‑commercial use**, with attribution (arXiv 2403.01469) — see §3. |

## 2. Base models (downloaded at runtime — never committed)

| Model | License | Access |
|---|---|---|
| `Qwen/Qwen3-8B` (default) | Apache‑2.0 | ungated |
| `Qwen/Qwen2.5-7B-Instruct` | Apache‑2.0 | ungated |
| `01-ai/Yi-1.5-9B-Chat` | Apache‑2.0 | ungated (automatic fallback) |
| `meta-llama/Llama-3.1-8B-Instruct` | **Llama 3.1 Community License** | gated; used **only** if an HF token is present, otherwise skipped |
| `K-intelligence/Midm-2.0-Base-Instruct` (KorMedMCQA run) | **MIT**, Copyright (c) 2025 KT Corporation | ungated; the MIT notice must travel with derived weights |

Pin a specific revision (commit SHA) rather than a mutable tag when you need
reproducibility.

## 3. Derived model artifacts (regenerable — **not committed**)

`quantization/artifacts/` is **gitignored**: only its `.gitkeep`, `.gitignore`
and `README.md` scaffolding are tracked. Model weights are regenerated locally
by the pipeline.

| Artifact (local, regenerable) | Derived from | Weight terms | Data terms |
|---|---|---|---|
| BF16 LoRA adapter / merged + INT4 weights | **Qwen3‑8B** (Apache‑2.0) fine‑tuned on **KorQuAD 1.0** | Follow the **Apache‑2.0** base | Trained on KorQuAD (**CC BY‑ND 2.0 KR**) → redistribution is a **known‑unknown**, see §4. **Not published.** |
| A merge and C `_bf16adapt` (KorMedMCQA run) | **Mi:dm 2.0 Base** (MIT) fine‑tuned on **KorMedMCQA** | MIT notice reproduced in the HF repo's `LICENSE` | CC BY‑NC 2.0 → **published on the HF Hub for non‑commercial use**, with attribution and a "not medical advice" notice, by `quantization/publish_hf.py`. Only BF16 safetensors are published; pickled INT4 dirs are rebuilt locally. |

## 3b. Managed-API evaluation outputs (committed result JSON)

`quantization/results/kormedmcqa_midm/managed/` holds one result per Azure AI Foundry deployment scored on the
KorMedMCQA test split by `quantization/eval_managed.py`. Each item keeps only the item id, role, year, gold and
predicted option numbers, the first 40 characters of the model output (usually a single digit),
per-request token usage, latency, finish reason and a short error reason. The three files written first
(gpt-oss-120b, llama-4-maverick, mistral-medium-3-5) predate the short format and keep the first 120 characters of
the raw HTTP 400 body instead, which includes the provider's opaque response id. No question text, no prompts beyond the
fixed system/user templates, no endpoint address, subscription, tenant or account identifier. The KorMedMCQA
questions were sent to the managed endpoints for evaluation only; no model was trained on those outputs.

## 4. Known unknowns (do not overstate)

1. **KorQuAD ND clause vs. fine‑tuned artifacts.** Whether weights trained on a
   NoDerivatives dataset may be redistributed is unsettled and depends on
   jurisdiction/interpretation. This repo does **not** claim it is settled, and
   does not publish KorQuAD‑trained weights.
3. **KorMedMCQA source exams.** The questions come from Korean national
   licensing exams; the dataset's upstream rights are not recorded beyond its HF
   card. Published weights stay non‑commercial and point to the dataset card.
4. The coverage gate is **not** an SBOM, a signature, or a build attestation.
