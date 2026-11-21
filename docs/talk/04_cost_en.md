# 4. Cost — a small model (SLM) on a self-hosted GPU vs. managed APIs
[한국어](04_cost.md) | **English**

Shared conditions are in [0. Experiment setup](00_setup_en.md). Every cost in this document is a **calculated value**, not a value taken from a bill.
- **Self-hosted GPU (①, ②):** the time and throughput measured in [2](02_int4_quantization_en.md) and [3](03_vllm_serving_en.md), multiplied by the GPU's hourly list price.
- **Managed API (③):** the token usage the API returned, multiplied by the per-token list price.

## How we calculated

```text
cost per 1,000 questions = (hourly price ÷ 3,600) × seconds taken ÷ number of questions × 1,000
```

Tokenizers differ between models, so cost per token is hard to compare. That is why we divided by the number of questions.

| Price | Value | Source |
|---|---|---|
| On-demand | $6.98 per hour | Azure Retail Prices API, `Standard_NC40ads_H100_v5`, West US 2, Linux |
| Spot | $3.537 per hour | Same API, as of 2026-11-01. Spot prices change often, and a Spot VM can be reclaimed while in use |

Both are public list prices. Contract discounts are not included. Using only the Spot price would make the self-hosted GPU look cheaper than it really is, so we give both values.

## ① Submitting all 3,009 exam questions at once (calculated from measured time)

This is the time from when `eval_vllm.py` puts all 3,009 questions into vLLM at once until all the answers are out. Model loading time is excluded.

![GPU cost per 1,000 questions when all 3,009 exam questions are submitted at once](figures/cost_per_1k.svg)

| Model | Time for 3,009 questions | Per 1,000 questions, on-demand | Per 1,000 questions, Spot |
|---|---|---|---|
| BF16 LoRA | 17.27 s | **$0.0111** | $0.0056 |
| INT4 PTQ | 266.21 s | $0.1715 | $0.0869 |
| INT4 QAT | 266.27 s | $0.1716 | $0.0869 |

For work that processes many questions at once, BF16 is 15x cheaper than INT4. As seen in [3. vLLM serving](03_vllm_serving_en.md), this is because the INT4 format used here is slow at large batch sizes.

## ② Taking requests through a server (calculated from measured throughput)

We used the requests per second measured in the online serving part of [3. vLLM serving](03_vllm_serving_en.md). These values assume the server keeps receiving requests without idling, so they come out lower than in practice.

| Condition | BF16: requests/s → on-demand per 1,000 questions | INT4: requests/s → on-demand per 1,000 questions |
|---|---|---|
| Concurrency 1, prefix caching off | 14.1 → $0.137 | 1.8 → $1.086 |
| Concurrency 1, prefix caching on | 21.2 → $0.091 | 7.6 → $0.255 |
| Concurrency 64, prefix caching off | 25.9 → $0.075 | 1.9 → $1.023 |
| Concurrency 64, prefix caching on | 32.8 → **$0.059** | 11.1 → $0.175 |

"On" means new questions were sent at each concurrency level, so only the prefix that is the same in every request (530 tokens) was reused from the cache (2026-11-14 measurement).

With Spot, each value is about half (× 3.537 / 6.98).

**What we saw.**
- Cost depends more on **how the model is served** than on the model.
- Even for the same INT4 model at concurrency 64, it is $1.023 per 1,000 questions with prefix caching off and $0.175 with it on. That is a gap of about 6x.
- With the same way of serving, BF16 is cheaper than INT4: $0.059 vs. $0.175 at concurrency 64 with prefix caching on.
- In this H100 measurement, BF16 LoRA did not go above $0.14 per 1,000 questions on-demand under any condition.

## ③ Comparison with managed APIs — SLM vs. API

We also had 11 managed API models on Azure AI Foundry answer the same 3,009 questions from 2022-2024. The questions, system text, question wording and scoring are the same as for Mi:dm. There are two differences.
- **Prompt:** the KT system block (about 450 tokens) and the date line that the Mi:dm chat template adds on its own are not in the API requests.
- **Generation settings:** Mi:dm generates up to 16 tokens with greedy decoding. The API models generate up to 2,048 tokens with each model's default settings.

> **Measurement conditions**
> - Run: 2026-11-12, Azure AI Foundry, deployment type Global Standard, resource in East US 2, 1 run per model
> - Requests: Chat Completions, `max_completion_tokens` 2,048. `temperature` and `reasoning_effort` were not sent, so the model defaults were used. Concurrency was 12-32, set to fit each model's request limit
> - Authentication: Microsoft Entra ID (`az` login). No API keys were used, and the endpoint address was not kept in the result files
> - Prices: public list prices read from the Azure Retail Prices API (Foundry Models, eastus2, Global Standard) on 2026-11-12. For models whose price depends on context length, we used the short-context price. Contract discounts are not included
> - Order: first we checked token counts and errors on 30 questions from 2021 (pilot, not in the repo). Then we measured the 2022-2024 questions once per model

![SLM vs. API: accuracy on the 3,009 questions from 2022-2024 and cost per 1,000 questions](figures/slm_vs_api.svg)

The last column of the table below is a paired comparison with the BF16 LoRA model on the same questions. "Mi:dm only vs. API only" is the number of questions only the BF16 LoRA model got right vs. the number of questions only that API model got right.

| Model | Accuracy (95% confidence interval) | Tokens per question: input / output (of which reasoning) | Cost per 1,000 questions | Blocked questions | vs. BF16 LoRA: Mi:dm only vs. API only, McNemar p |
|---|---|---|---|---|---|
| gpt-6-astra | **98.4%** (97.9-98.8) | 205 / 21 (15) | $3.0851 | 8 | 26 vs. 810, below 0.000001 |
| gpt-6.1-sol | 98.3% (97.8-98.7) | 205 / 25 (19) | $0.6622 | 8 | 24 vs. 806, below 0.000001 |
| gpt-5.6-sol | 97.5% (96.9-98.0) | 205 / 40 (35) | $1.6257 | 8 | 35 vs. 792, below 0.000001 |
| gpt-6-luna | 95.9% (95.2-96.6) | 205 / 126 (116) | $0.0836 | 8 | 50 vs. 761, below 0.000001 |
| gpt-5.4 | 92.0% (91.0-92.9) | 205 / 4 (0) | $0.5712 | 8 | 80 vs. 673, below 0.000001 |
| deepseek-v4-1-flash | 85.0% (83.7-86.3) | 220 / 2 (0) | $0.0685 | 7 | 160 vs. 543, below 0.000001 |
| gpt-5.4-mini | 84.4% (83.1-85.7) | 205 / 4 (0) | $0.1718 | 8 | 162 vs. 526, below 0.000001 |
| gpt-oss-120b | 84.2% (82.9-85.5) | 268 / 296 (not counted separately) | $0.2180 | 10 | 205 vs. 564, below 0.000001 |
| llama-4-maverick | 81.5% (80.1-82.8) | 193 / 3 (0) | $0.0509 | 7 | 206 vs. 482, below 0.000001 |
| mistral-medium-3-5 | 76.0% (74.4-77.5) | 208 / 2 (0) | $0.3276 | 7 | 283 vs. 393, 0.000027 |
| **Mi:dm 2.0 Base BF16 LoRA (self-hosted H100)** | 72.3% (70.7-73.9) | 654 / 2 | **$0.0111** (all at once) to $0.137 (server, concurrency 1, prefix caching off) | 0 | — |
| Mi:dm 2.0 Base INT4 PTQ (self-hosted H100) | 69.5% (67.8-71.1) | 654 / 2 | $0.1715 (all at once) | 0 | — |
| gpt-5.4-nano | 64.9% (63.2-66.6) | 205 / 4 (0) | $0.0459 | 8 | 514 vs. 291, below 0.000001 |

Token counts are rounded means, and blocked requests were counted as 0 tokens. Tokenizers differ between models, so token counts are not compared directly with each other. "All at once" for the self-hosted GPU is the value from ① above. "Server, concurrency 1" is the prefix-caching-off value from ②, calculated from the throughput measured on 256 questions from 2021. All numbers come from [`compare_final_managed.json`](../../quantization/results/kormedmcqa_midm/managed/compare_final_managed.json) (the server value is `server_c1_cacheoff` in that file).

**What we saw.**
- 10 of the 11 managed models had higher accuracy than the tuned Mi:dm BF16 LoRA (72.3%). The top three models (gpt-6-astra, gpt-6.1-sol, gpt-5.6-sol) are at 97.5-98.4%.
- Only gpt-5.4-nano (64.9%), the lowest tier of the gpt-5.4 family, was below Mi:dm. In the same-question comparison too, 514 questions were answered correctly only by Mi:dm and 291 only by nano.
- When all 3,009 questions are solved at once, BF16 LoRA on the self-hosted H100 is the cheapest, at $0.0111 per 1,000 questions. Next come gpt-5.4-nano at $0.0459 and llama-4-maverick at $0.0509.
- When a server takes one request at a time (concurrency 1, prefix caching off), BF16 LoRA on the self-hosted H100 costs $0.137. Under this condition, gpt-6-luna (95.9%, $0.0836) is ahead on both accuracy and cost. deepseek-v4-1-flash and llama-4-maverick are also ahead on both.
- For each managed model, 7-10 questions were blocked by the Azure content filter. An HTTP 400 error came back, so no answer was received, and these were counted as wrong. 11 questions were blocked in all, and 7 of them were blocked on all 11 models. Adding up the blocks for every model gives 67 for doctor, 19 for nurse and 1 for dentist. On the self-hosted GPU, no questions were blocked.
- Questions with no answer because the output hit the 2,048-token limit: 13 for gpt-6-luna, 13 for gpt-oss-120b, 2 for gpt-5.6-sol. These questions were also counted as wrong.
- Questions marked wrong because the model answered with a number that is not an option number: 6 for mistral-medium-3-5, 3 for deepseek-v4-1-flash, 3 for gpt-5.4-mini, 2 for gpt-5.4-nano (e.g. "350", "1.8"). For the self-hosted GPU models it was 0.
- The prompt is 654 tokens for Mi:dm and 193-268 tokens for the API requests. Tokenizers differ, so the two cannot be compared directly, but most of the difference is the fixed system block of the Mi:dm template (about 450 tokens with the Mi:dm tokenizer).

**Plausible reasons.**
- It is natural to attribute the accuracy gap to differences in model size and amount of training. The top four models (95.9-98.4%) used 15-116 reasoning tokens per question before answering. gpt-5.4, on the other hand, reached 92.0% with no reasoning tokens. The Mi:dm LoRA learned only the answer number and answers right away. This experiment did not measure these explanations separately.
- The self-hosted GPU came out cheapest because the calculation assumes the GPU is used without idle time. On a server that takes one request at a time, the same GPU cost is split over far fewer questions.

**What this result does not show.**
- **This is not a comparison between models of the same size.** The managed models either have undisclosed sizes or are much larger than Mi:dm (11.5B). Turning this into "small models are useless" or "tuning has no effect" would be wrong.
- **We cannot check whether the managed models saw these past exam questions during training.** KorMedMCQA is public data. The untuned Mi:dm has the same limitation.
- **The cost bases differ.** The self-hosted GPU is GPU time × list price, and idle time, operations staff and disk are not included. The managed API is tokens used × list price. Both are public list prices, and contract discounts are not included.
- **Each model was measured once.** A managed deployment can have its model version change automatically even under the same name. For the 7 GPT-family models, the versions in this run are in `models_reported` in the result file; the other 4 returned only the model name, so their versions could not be recorded. The content filter results were not the same every time either. When only the blocked questions were sent again, some questions gave a different result (the records of that resend are not in the repo).
- **Model settings were left at their defaults.** Changing settings such as `reasoning_effort` may change accuracy and cost.
- **Latency is not compared.** Request limits and concurrency differed between models.
- **We looked only at accuracy and cost.** Whether data leaves the company, whether the model version can be pinned and whether the content filter can be adjusted are not in this table. Each place that uses these models has to weigh such conditions separately.

**Running it again.** No GPU is needed, but you must be able to access the Foundry resource with `az login`.

```bash
export MANAGED_BASE_URL=https://<resource name>.services.ai.azure.com/openai/v1
export MANAGED_AZ_SUBSCRIPTION=<subscription name or ID>      # selects the tenant to get the token from
# Pilot: 30 questions from 2021 (does not touch the 2022-2024 result files)
python -m quantization.eval_managed --config quantization/configs/kormedmcqa_midm.yaml \
    --deployment gpt-5.4-nano --split selection_dev --eval-size 30
# Main measurement: 3,009 questions from 2022-2024, once per model (prices are USD per 1M tokens; record the source and date with them)
python -m quantization.eval_managed --config quantization/configs/kormedmcqa_midm.yaml \
    --deployment gpt-5.4-nano --concurrency 32 --region eastus2 \
    --price-in 0.2 --price-out 1.25 --price-cached-in 0.02 --price-source "Azure Retail Prices API ..., read <date>"
python -m quantization.managed_compare            # paired comparison and summary table
```

The data loader (`datasets` 4.3.0) breaks on Python 3.14, so we ran it with Python 3.12.

## What this result does not show (self-hosted GPU side)

- **Only GPU time in use was counted.** Server idle time, model loading time, disk, network and the time of the people who run it are not included. The real cost is higher than this.
- **② is a calculation that rests on an assumption.** It assumes the server is busy without a break. The prefix-caching-on values are the case where only the prefix that is the same in every request (530 tokens) was reused from the cache. Repeated questions could make it cheaper, but we did not measure that ([3. vLLM serving](03_vllm_serving_en.md#exam-answer-requests-answers-of-1-2-tokens)).
- **One GPU type (H100 NVL), based on public list prices.** Results differ for other GPUs, other regions and contract prices.
- **INT4 coming out expensive is a result of the INT4 format used here (TorchAO tile-packed).** Turning it into a conclusion about INT4 in general would be wrong.
