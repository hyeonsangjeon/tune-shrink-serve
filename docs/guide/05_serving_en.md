# 5. Measure serving speed

[한국어](05_serving.md) | **English**

We serve the BF16 LoRA model and the INT4 model with vLLM and measure two things.
- **Offline throughput:** generated tokens per second when 1 to 256 requests are sent at once
- **Online serving:** requests completed per second and time to first token (TTFT) when 1, 8, 32 or 64 concurrent requests are sent to an OpenAI-compatible server. We compare prefix caching on and off

| What you need | Time taken (H100 NVL) | Result doc |
|---|---|---|
| The BF16 LoRA model and the INT4 model from [step 4](04_int4_en.md) | Online, both models: about 30 min (including server start-up) | [3. vLLM serving](../talk/03_vllm_serving_en.md) |

The shortest code that starts the server and sends one question is in [`steps/5_serve.sh`](../../steps/5_serve.sh) and [`steps/5_ask.py`](../../steps/5_ask.py). The steps below show how to measure speed.

## Steps

```bash
bash scripts/gpu/stage.sh bench_A          # offline, BF16 LoRA
bash scripts/gpu/stage.sh bench_B          # offline, INT4 PTQ
bash scripts/gpu/stage.sh bench_C          # offline, INT4 QAT (same format as PTQ, so this checks that the speed is the same)

FRESH=1 bash scripts/gpu/stage.sh online_A # online, BF16 LoRA, prefix caching on → off
FRESH=1 bash scripts/gpu/stage.sh online_B # online, INT4 PTQ, prefix caching on → off
```

- **`FRESH=1`:** At each concurrency level, sends 256 questions from 2021 that have not been sent yet, and sends the same questions in the same order with caching on and with caching off. The numbers in the talk come from this mode. Without it, the same 256 questions are sent again at every level (the whole prompt stays in the cache), and 128-token generation requests are measured as well.
- [`quantization/serving/serve_vllm.sh`](../../quantization/serving/serve_vllm.sh) starts and stops the server. The client and the server are on the same VM, so network latency is not included.

## Verify

| Stage | File (`quantization/results/kormedmcqa_midm/`) | What to look for |
|---|---|---|
| `bench_*` | `bench_<model>_seed42_h100.json` | `throughput_tok_s` for each batch |
| `online_*` | `online_<model>_mcqa_cache{on,off}_seed42_h100.json` | At every level: `failed` 0, `request_throughput_rps`, `ttft_s`. With `FRESH=1`: `config.fresh_per_level: true`, and `prompt_slice` does not overlap between levels |

If you changed the online results, the server cost changes too. Redo the "Taking requests through a server" table in [results doc 4](../talk/04_cost_en.md) and re-run `python -m quantization.managed_compare` ([doc 8](08_troubleshooting_en.md#after-changing-the-results)).

## When it fails

- **The INT4 server is slow to start:** The INT4 model takes longer to load than BF16 (in this run, model loading during scoring took 106 s vs 22 s). `HEALTH_TRIES` is set to 900 (about 15 min).
- **INT4 with prefix caching off takes tens of seconds to the first token:** This is expected. Tile-packed INT4 is slow at prompt processing, so the queue gets long when concurrent requests pile up ([results doc 3](../talk/03_vllm_serving_en.md)).
- **The next server does not start (out of GPU memory):** The previous server has not released the GPU yet. Check with `nvidia-smi` and stop the leftover process. After stopping a server, the stage waits until GPU memory drops below 2GB.

Next: [6. Compare with API models](06_managed_api_en.md)
