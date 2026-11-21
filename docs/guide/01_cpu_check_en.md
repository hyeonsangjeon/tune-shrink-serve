# 1. Check without a GPU (about 1 min)
[한국어](01_cpu_check.md) | **English**

Check that this repo works without a GPU, a cloud account or a model download. A laptop is enough.

| What you need | Time taken | Cost |
|---|---|---|
| Python 3.10 or later (CI uses 3.10 and 3.11; also checked on 3.14), git | about 1 min | none |

## Steps

```bash
git clone https://github.com/hyeonsangjeon/tune-shrink-serve.git && cd tune-shrink-serve
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements/cpu.txt

python -m pytest -q                                   # tests
python scripts/build_evidence_index.py --check        # numbers in the docs = values in the result JSONs?
python scripts/make_figures.py --check                # are the charts redrawn identically from the result JSONs?
python -m quantization.managed_compare --check        # does the SLM vs API table come out identically from the result JSONs?
```

## Verify

| Command | Line you see when it passes |
|---|---|
| `pytest` | `passed` (a few tests skipped because there is no GPU show as `skipped`) |
| `build_evidence_index.py --check` | `[evidence] OK — 72 claims verified` |
| `make_figures.py --check` | `[figures] OK — 6 figures match the result JSONs` |
| `managed_compare --check` | `[managed_compare] OK` |

**What this 1 minute checks:**
- Whether the data split, prompt, scoring and serving benchmark code pass the tests
- Whether the numbers written in the docs match the values in the result JSONs ([evidence index](../EVIDENCE.md))
- Whether the charts and the comparison table are rebuilt identically from the result JSONs

**What it does not check:** it does not redo training, quantization or serving. Whether the result JSONs themselves are correct can only be known by rerunning on a GPU (from step 2).

## When it fails

- **`ModuleNotFoundError`:** the virtual environment is not activated, or `requirements/cpu.txt` is not installed.
- **`[evidence] ... not found in README.md`:** a number in the docs was changed by hand. Fix the result JSON and [`evidence_ledger.yaml`](../evidence_ledger.yaml) together ([doc 8](08_troubleshooting_en.md#after-changing-the-results)).
- **`make_figures.py --check` fails:** the result JSONs changed but the charts were not redrawn. Redraw them with `python scripts/make_figures.py`.

The same checks run in GitHub Actions ([`.github/workflows/test.yml`](../../.github/workflows/test.yml)) on every push and PR.

Next: [2. Prepare the GPU VM](02_gpu_setup_en.md)
