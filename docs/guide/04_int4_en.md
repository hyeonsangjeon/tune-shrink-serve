# 4. Shrink to INT4 and score
[한국어](04_int4.md) | **English**

Shrink the BF16 LoRA model from step 3 to TorchAO INT4 in two ways (PTQ, QAT). Then score the four models (untuned, BF16 LoRA, INT4 PTQ, INT4 QAT) on the 3,009 questions from 2022-2024, **only once per model**, and compare them on the same questions.

| What you need | Time taken (H100 NVL) | Result doc |
|---|---|---|
| The BF16 LoRA model from [step 3](03_finetune_en.md) | QAT training 102 min (600 steps). Scoring: BF16 about 40 s, INT4 about 6 min (including model loading) | [2. INT4 quantization](../talk/02_int4_quantization_en.md) |

To read the TorchAO code, see [`steps/2_int4_ptq.py`](../../steps/2_int4_ptq.py) (PTQ), [`steps/3_int4_qat.py`](../../steps/3_int4_qat.py) (QAT) and [`steps/4_evaluate.py`](../../steps/4_evaluate.py) (scoring). Each one can also be run directly with `python steps/<file>`.

## Steps

```bash
bash scripts/gpu/stage.sh b          # INT4 PTQ: convert the trained BF16 model to INT4 as is (about 7.2 GiB)
bash scripts/gpu/stage.sh c_smoke    # INT4 QAT: 5-step trial training
bash scripts/gpu/stage.sh c          # INT4 QAT: main training, 600 steps, then save as INT4

bash scripts/gpu/stage.sh final_base # 2022-2024 questions, once per model
BACKEND=hf bash scripts/gpu/stage.sh final_A
bash scripts/gpu/stage.sh final_B
bash scripts/gpu/stage.sh final_C
bash scripts/gpu/stage.sh compare    # McNemar comparison on the same questions, 4 pairs
```

- There is a single INT4 scheme: weight-only, group 128, tile-packed format (`quant:` in the config).
- `final_*` stops without measuring again if the same result file already exists. This is because the 2022-2024 questions are meant to be looked at only once, at the end.

### QAT memory

| GPU | Setting | Notes |
|---|---|---|
| H100 NVL 94GB | `--qat-batch 4 --qat-accum 4` (default) | 73.3 GiB during training (`nvidia-smi`, a single reading around step 85) |
| 80GB class | `--qat-batch 2 --qat-accum 8` | Still 16 questions per step. This setting was not measured |

## Verify

| Stage | File (`quantization/results/kormedmcqa_midm/`) | What to look for |
|---|---|---|
| `b` | `B_build_seed42.json` | `device` is that GPU. Folder size about 7.2 GiB (`du -sh quantization/artifacts/kormedmcqa_midm/B_int4_ptq_seed42`) |
| `c` | `C_train_seed42.json` | `global_step` 600, `train_seconds` |
| `final_*` | `eval_vllm_<model>_seed42_h100.json` | `n_eval` 3,009, `n_unparsed` close to 0, `accuracy` |
| `compare` | `compare_final_seed42_h100.json` | For each pair, the number of questions that only one side got right, and the McNemar p |

This run's values are in the tables of [results doc 2](../talk/02_int4_quantization_en.md). We did not check whether measuring again under the same conditions gives the same values.

## Scoring on a different GPU

The INT4 files are torchao tensors saved with pickle (`.bin`), so they are tied to the torch and torchao builds that made them. So for a different GPU, move only the two BF16 safetensors and rebuild INT4 on that GPU.

1. Put the merged BF16 LoRA model in `quantization/artifacts/kormedmcqa_midm/A_bf16_seed42/`.
2. Put the BF16 weights from the finished QAT training in `quantization/artifacts/kormedmcqa_midm/C_int4_qat_seed42_bf16adapt/`. To download the published models, use `bash scripts/gpu/stage.sh fetch_hf`.
3. Rebuild INT4 with `stage.sh b` and `stage.sh c_export`. Check that `device` in `B_build_seed42.json` and `C_export_seed42.json` is that GPU.
4. Run `final_*`, `compare` and [step 5](05_serving_en.md). The result file names carry the GPU type (`_h100`, `_a100`), so the tables are built separately for each GPU.

## Publishing the trained weights (optional)

```bash
python -m quantization.publish_hf --kind A_bf16 --dir <merged BF16 LoRA folder> --repo <account>/<repo> --train-json <training result JSON> --dry-run
```

- **First check the model card with `--dry-run`.** The card includes the data source, the non-commercial condition, the Mi:dm MIT notice and "not medical advice".
- **Files uploaded:** only safetensors. INT4 pickles and training state files are not uploaded.
- **HF token:** use a fine-grained token with write access to that repo only, and revoke it when the upload is done. Enter the token yourself with `hf auth login`; do not write it in a script.

## When it fails

- **Out of memory in QAT:** lower it to the 80GB setting in the “QAT memory” table above.
- **`final_*` stops right away:** a 2022-2024 result file for that model already exists. Move the file and rerun only if the measurement really was wrong.
- **The script breaks after QAT training finishes:** the script file was overwritten while it was running ([doc 8](08_troubleshooting_en.md)). Redo only the save with `stage.sh c_export`.

Next: [5. Measure serving speed](05_serving_en.md)
