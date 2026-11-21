# 3. vLLM serving — batches, concurrent requests, prefix caching
[한국어](03_vllm_serving.md) | **English**

Shared conditions are in [0. Experiment setup](00_setup_en.md). The models compared are the BF16 LoRA model and the INT4 model ([2. INT4 quantization](02_int4_quantization_en.md)). INT4 PTQ and INT4 QAT have the same storage format, so their speed is also the same. In the offline measurement we measured both, and the throughput difference at each batch size was within 1%. So the online measurement used INT4 PTQ only.

## We measured two things

| | Offline throughput | Online serving |
|---|---|---|
| What it shows | How fast the GPU is, depending on how many requests it processes at once | How long a response takes when several users ask at the same time |
| Method | Put 1-256 requests into vLLM `LLM.generate` at once | Send HTTP streaming requests to the vLLM OpenAI-compatible server (`vllm.entrypoints.openai.api_server`) at concurrency 1, 8, 32 and 64 |
| Prompts | First 256 questions from 2022-2024 | ① Exam answer: 2021 questions, shuffled; for each concurrency level, 256 not sent before ② 128-token generation: first 256 questions from 2021 |
| Generation length | Exactly 128 tokens per request (`ignore_eos`) | ① Exam answer: up to 16 tokens, stops at `<|eot_id|>` (1-2 tokens in practice) ② 128-token generation (`ignore_eos`) |
| Code | [`quantization/v2_bench.py`](../../quantization/v2_bench.py) | [`quantization/serving/v2_bench_serve.py`](../../quantization/serving/v2_bench_serve.py), [`serve_vllm.sh`](../../quantization/serving/serve_vllm.sh) |

> **Measurement conditions**
>
> - Runs: 2026-11-11 (offline, 128-token generation requests), 2026-11-14 (exam-answer requests), 1× H100 NVL 94GB, vLLM 0.23.0, 1 run per condition
> - Shared server settings: BF16 compute, `gpu_memory_utilization=0.85`, `max_model_len=2048`, CUDA graphs on, greedy, the model's default sampling settings off (`--generation-config vllm`)
> - INT4: `--quantization torchao`, tile-packed format
> - Online: 256 requests per concurrency level. The client and the server were on the same VM, so network latency is not included
> - Questions for the exam-answer requests: the 1,059 questions from 2021 were shuffled with seed 0 and split into four sets of 256 (1,024 questions), and one set was sent to each of concurrency 1, 8, 32 and 64. Prefix caching on and off received the same sets in the same order (`FRESH=1 bash scripts/gpu/stage.sh online_A`, `online_B`)
> - Prefix caching: the vLLM default is on. Only the online measurement compared on and off (`--no-enable-prefix-caching`)
> - Result files: `bench_*_seed42_h100.json` (offline), `online_*_seed42_h100.json` (online), [`prompt_shared_prefix.json`](../../quantization/results/kormedmcqa_midm/prompt_shared_prefix.json) (the token count of the prefix that all prompts share)

## Offline throughput

![Generated tokens per second by number of requests submitted at once, BF16 and INT4](figures/offline_throughput.svg)

| Requests submitted at once | 1 | 4 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|
| BF16 (tokens/s) | 116 | 451 | 1,652 | 2,892 | 4,703 | 5,955 | 7,647 |
| INT4 (tokens/s) | **139** | **551** | 668 | 759 | 834 | 729 | 844 |

With a single request (median of 24 repeats):

| | Time to first token (TTFT) | Until all 128 tokens are received |
|---|---|---|
| BF16 | 14 ms | 1.10 s |
| INT4 | 15 ms | **0.70 s** |

**What we saw.**
- With 1-4 requests, INT4 is about 20% faster.
- With 16 or more requests, BF16 is ahead, and at 256 the gap is 9x.
- INT4 does not go past the 800s of tokens per second.

**Plausible reason.** With few requests, most of the time goes to reading weights from memory, so INT4, whose weights are 1/3 the size, has the advantage. With more requests, the amount of computation grows. BF16 then uses tensor-core matrix multiplication as is, while tile-packed INT4 computes through the tinygemm kernel, unpacking the weights as it goes. This kernel is tuned for processing few rows at a time. We did not do kernel-level profiling, though.

## Online serving

### Exam-answer requests (answers of 1-2 tokens)

This is what the actual service looks like. The prompt is about 650 tokens and the answer is a single number, so almost all of the time goes to prompt processing (prefill).

![Requests processed per second by concurrency, BF16 and INT4, prefix caching on and off](figures/online_mcqa_requests.svg)

**Read this first.** For each concurrency level we sent 256 questions that had not been sent before. So the only thing the server with prefix caching on reused from its cache was the prefix that is the same in every request. This prefix is the KT system text, the date and the task instruction: 530 tokens, a little over 80% of the 653-token average prompt. On and off received the same questions in the same order, so the two can be compared under the same conditions at every concurrency level.

BF16 (requests per second / time to first token, median · 99th percentile):

| Concurrency | Prefix caching on | Prefix caching off |
|---|---|---|
| 1 | 21.2 / 37 ms · 48 ms | 14.1 / 62 ms · 73 ms |
| 8 | 39.3 / 198 ms · 298 ms | 26.3 / 270 ms · 469 ms |
| 32 | 33.2 / 950 ms · 1,092 ms | 26.3 / 1,040 ms · 1,814 ms |
| 64 | 32.8 / 1,818 ms · 2,541 ms | 25.9 / 2,070 ms · 3,578 ms |

INT4:

| Concurrency | Prefix caching on | Prefix caching off |
|---|---|---|
| 1 | 7.6 / 110 ms · 302 ms | 1.8 / 542 ms · 735 ms |
| 8 | 11.0 / 724 ms · 986 ms | 1.9 / 4,151 ms · 4,485 ms |
| 32 | 10.4 / 2,805 ms · 4,328 ms | 1.9 / 14,349 ms · 18,342 ms |
| 64 | 11.1 / 5,542 ms · 7,052 ms | 1.9 / 26,174 ms · 34,840 ms |

**What we saw.**
- All requests succeeded.
- Reusing only the shared prefix already has an effect at every concurrency level.
  - BF16: 1.3-1.5x the requests per second. At concurrency 1, TTFT goes from 62 → 37 ms
  - INT4: 4-6x the requests per second. At concurrency 1, TTFT goes from 542 → 110 ms
- At concurrency 8 and above, BF16 handles more, whether prefix caching is on or off. Comparing the runs with it on, BF16 handles 33-39 requests per second and INT4 10-11, a gap of about 3x.
- INT4 with prefix caching off does not go past 2 requests per second. At concurrency 64, the wait for the first token is 26 s.
- **Difference from the 2026-11-11 measurement.** That time, the same 256 questions were sent again at each concurrency level, and with prefix caching on, INT4 appeared to handle more than BF16. When new questions were sent, that reversal did not appear.

**Plausible reason.** The INT4 model is especially slow at prompt processing. The tinygemm INT4 kernel is weak at processing the 650 tokens of one request all at once (the same reason as in the "Offline throughput" section of this document). Prefix caching cuts that work by reusing the KV cache of the prefix that was already processed. That is why the difference is much larger for INT4. Even with the prefix skipped, though, the remaining ~120 tokens that differ for each question still have to be processed, and when requests pile up, the limit of the INT4 kernel shows again here. INT4 looking ahead in the earlier measurement seems to be because the whole prompt was in the cache, so even this processing was skipped. We did not measure this broken down at the kernel level.

### 128-token generation requests (prefix caching on)

This is the case of long answers, as in a chat service. It is the 2026-11-11 measurement, and the prompts are the same questions as the exam-answer requests sent just before, so they are in the cache. Please read it as a table that compares generation speed.

| Concurrency | BF16: tokens/s / full response, median · 99th | INT4: tokens/s / full response, median · 99th |
|---|---|---|
| 1 | 116 / 1.10 s · 1.11 s | **180** / **0.71 s** · 0.72 s |
| 8 | **854** / 1.18 s · 1.36 s | 723 / 1.40 s · 1.54 s |
| 32 | **2,432** / 1.47 s · 2.19 s | 974 / 4.08 s · 4.66 s |
| 64 | **3,541** / 2.02 s · 3.41 s | 1,025 / 7.64 s · 9.07 s |

This follows the same pattern as the offline results. With one request INT4 is faster, and at concurrency 8 and above BF16 is faster.

## What this result does not show

- **For prefix caching, we measured only the case where the prefix that is the same in every request is reused.** A service where the same questions come in repeatedly (e.g. frequently asked questions) could reuse more, but that case is not in these tables.
- **The exam-answer requests and the 128-token generation requests were measured on different days.** The 128-token generation table is the 2026-11-11 measurement as it was, and it did not send new questions.
- **The offline measurement also used the vLLM default (prefix caching on), and the same questions went in again at each batch size.** Since it is mostly 128-token generation, we expect the effect to be small, but we did not measure it separately.
- **We measured only one INT4 method (TorchAO tile-packed, tinygemm kernel).** Methods that use other INT4 kernels (e.g. Marlin) may give different results at large batch sizes.
- **One H100, one vLLM version (0.23).** We did not repeat the measurements.
- **Network latency is not included.** For real users, network round-trip time is added to these values.

## To run it again

```bash
CFG=quantization/configs/kormedmcqa_midm.yaml

# Offline throughput
python quantization/v2_bench.py --config $CFG --model-dir quantization/artifacts/kormedmcqa_midm/A_bf16_seed42 \
  --method A_bf16 --precision bf16 --max-model-len 2048 --out bench_A.json

# Online serving: start the server (for INT4, the third argument is torchao), then send requests with real exam questions
quantization/serving/serve_vllm.sh quantization/artifacts/kormedmcqa_midm/B_int4_ptq_seed42 8000 torchao
python -m quantization.serving.v2_bench_serve --model quantization/artifacts/kormedmcqa_midm/B_int4_ptq_seed42 \
  --concurrency 1,8,32,64 --num-prompts 256 --prompts-config $CFG --prompts-split selection_dev \
  --stop-token-ids 131301,2 --max-tokens 16 --fresh-per-level --shuffle-seed 0 --output online_B.json

# To compare with prefix caching off, start the server like this
EXTRA_ARGS=--no-enable-prefix-caching quantization/serving/serve_vllm.sh <model folder> 8000 [torchao]
```

`--fresh-per-level` sends questions that have not been sent before at each concurrency level. If you give on and off the same `--shuffle-seed`, the same questions go out in the same order.

On the GPU VM, `FRESH=1 bash scripts/gpu/stage.sh online_A` and `online_B` run the steps above twice each, once with prefix caching on and once off. Without `FRESH=1`, they run the 2026-11-11 way (the same 256 questions are sent again, and the 128-token generation requests are included).
