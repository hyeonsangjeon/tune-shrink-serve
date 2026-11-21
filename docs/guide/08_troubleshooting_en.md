# 8. When it fails, and what to commit

[한국어](08_troubleshooting.md) | **English**

## Known pitfalls

- **The Mi:dm chat template always prepends a KT system message (about 450 tokens) and today's date.**
  - The prompt changes when the date changes, so `template_kwargs.date_string` in the config fixes the date.
  - Online serving puts the same value into each request as well (`chat_template_kwargs`).
- **End-of-turn and stop tokens:** End of turn is `<|eot_id|>` (131301), and eos is `<|end_of_text|>` (2). The training target is `[number, 131301]`, and generation stops at `[131301, 2]`.
- **The model's default generation config is sampling** (temperature 0.8, top_p 0.7, repetition_penalty 1.05). Scoring and the server ignore these values with `generation_config="vllm"` and run greedy.
- **Unsloth 2026.11.3 and the Mi:dm tokenizer:** Tokenizer loading breaks on transformers 5.17, and also on 4.57.6 when a GPU is visible. `train.unsloth_external_tokenizer: true` loads only the tokenizer separately and passes it in ([results doc 1](../talk/01_finetune_unsloth_en.md#what-it-took-to-use-unsloth)).
- **Trial-training checkpoints:** If you ran a 5-step trial training by hand, clear the `*_run/` folder before the main training. If you leave it, `--resume` continues from the 5-step checkpoint. `stage.sh a_smoke` and `c_smoke` do this for you.
- **Do not overwrite a script file while it is running.** bash reads a script while it runs it. If you change the file while it runs, it breaks near the end (in this run, it actually stopped this way after QAT training finished). Copy long-running queues out of the repo and run the copy.
- **INT4 speed:** Tile-packed INT4 (tinygemm kernel) is fast with few requests and slow with many. Prompt processing is also slow, so it depends heavily on whether prefix caching is on ([results doc 3](../talk/03_vllm_serving_en.md)).
- **Questions in the online benchmark:** Without `FRESH=1`, the same 256 questions are sent again at every concurrency level. With prefix caching on, the whole prompt is then in the cache, so it looks faster than it really is.
- **Python 3.14:** `datasets` 4.3.0 breaks with a pickle error. Use Python 3.12 or earlier.
- **zsh:** Unquoted variables are not split into words. If you put the options in one variable and pass it, it goes in as a single argument and argparse stops.
- **VM cost:** If you only stop the VM, you keep being billed. When you are not using it, deallocate it (`az vm deallocate`).

## What to commit

- **Committed:** The result JSON files in `quantization/results/kormedmcqa_midm/`. The managed results (`managed/`) contain the question id, the correct answer, the predicted number, the first 40 characters of the answer, tokens used, latency, finish reason, and a short error reason. The three files run first (gpt-oss-120b, llama-4-maverick, mistral-medium-3-5) predate the short error-reason format, so they keep the first 120 characters of the error response body as is (including the provider's response id). Pilot results are not committed.
- **Not committed:** Weights (`quantization/artifacts/`) are gitignored. The dataset text and the explanation texts written by Mi:dm are not included either (KorMedMCQA is CC BY-NC 2.0, [data and licenses](../DATA_AND_LICENSES.md)).
- **Secrets:** Pass endpoint addresses, subscriptions and tokens only as environment variables. Before committing, run `python scripts/scan_secrets.py quantization scripts docs README.md`.

## After changing the results

If you produced new result JSON, update the docs and charts to match. CI catches mismatches in the Korean docs (the `_en.md` translations are not checked, so update them by hand).

```bash
python scripts/make_figures.py                     # redraw the charts (docs/talk/figures/)
python -m quantization.managed_compare             # rebuild the SLM vs API comparison table (also when you changed the online results)
# after fixing the numbers in the docs, update the expected values in docs/evidence_ledger.yaml, then
python scripts/build_evidence_index.py --emit      # rebuild docs/EVIDENCE.md
python scripts/build_evidence_index.py --check     # doc numbers = result JSON values?
python -m pytest -q
```

The [evidence index](../EVIDENCE.md) is a table that records, for each number in the docs, which value in which result JSON it came from. If a number differs from the result JSON, `--check` fails.
