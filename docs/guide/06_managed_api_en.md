# 6. Compare with API models (no GPU needed)

[한국어](06_managed_api.md) | **English**

Managed API models answer the same 3,009 questions from 2022-2024, and we put their accuracy and cost side by side with Mi:dm on a self-hosted GPU. Any OpenAI-compatible Chat Completions endpoint works; this time we used `/openai/v1` on Azure AI Foundry.

| What you need | Time taken (this run) | Cost | Result doc |
|---|---|---|---|
| An Azure AI Foundry resource and model deployments, the `az` CLI, Python 3.12 | 2.6-28.7 min per model | Tokens used × the model's list price. The cost per 1,000 questions is in the table in the results doc | [4. Cost](../talk/04_cost_en.md#-comparison-with-managed-apis--slm-vs-api) |

## Steps

1. **Sign in:** Use `az login` to sign in to an account that can access the Foundry resource. Authentication uses an Entra ID token; no API key is used.
2. **Pass the endpoint only as an environment variable.** It is not stored in the repo or in the result files.
   ```bash
   export MANAGED_BASE_URL=https://<resource name>.services.ai.azure.com/openai/v1
   export MANAGED_AZ_SUBSCRIPTION=<subscription name or ID>      # picks the tenant to get the token from (optional)
   ```
3. **Pilot:** Run 30 of the 2021 questions to check token counts, answer format and errors. This does not touch the 2022-2024 result files.
   ```bash
   python -m quantization.eval_managed --config quantization/configs/kormedmcqa_midm.yaml \
       --deployment <deployment name> --split selection_dev --eval-size 30
   ```
4. **Prices:** Read the model's input, output and cached-input prices (USD per 1M tokens) from the Azure Retail Prices API.
5. **Full run:** Measure each deployment once. If the result file already exists, the script stops (use `--force` only when that run went wrong).
   ```bash
   python -m quantization.eval_managed --config quantization/configs/kormedmcqa_midm.yaml \
       --deployment <deployment name> --concurrency 32 --region eastus2 \
       --price-in <input> --price-out <output> --price-cached-in <cached input> --price-source "Azure Retail Prices API, read <date>"
   ```
6. **Comparison table:** Build the paired comparison and the summary table (`managed/compare_final_managed.json`).
   ```bash
   python -m quantization.managed_compare
   ```

## Verify

| File (`quantization/results/kormedmcqa_midm/managed/`) | What to look for |
|---|---|
| `eval_managed_<deployment>.json` | `n_eval` 3,009. `n_errors` is the number of questions blocked by the content filter. `length` in `finish_reasons` is the number of questions that hit the output limit (2,048 tokens) |
| `compare_final_managed.json` | For each model: accuracy, cost per 1,000 questions, and a same-question comparison with Mi:dm |

The result files contain the question id, the correct answer, the predicted number, the first 40 characters of the answer, tokens used, latency, finish reason, and a short error reason. They do not contain the question text or the endpoint address.

## When it fails

- **HTTP 429:** The deployment's request limit was hit. The script follows `Retry-After` and sends the request again. For a deployment limited to 250 requests per minute, set about `--concurrency 12`.
- **HTTP 400 (content filter):** The Azure content filter blocks some medical questions (7-10 questions per model in this run). They are counted as wrong and recorded in `n_errors`.
- **`datasets` breaks with a pickle error:** You are on Python 3.14. Use a Python 3.12 virtual environment.
- **In zsh, argparse stops when you pass options in a variable:** zsh does not split unquoted variables into words. Write the command out in full, or run it with bash.

Next: [7. (Optional) Score-raising experiments](07_recipe_sweep_en.md)
