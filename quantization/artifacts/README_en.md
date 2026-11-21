# artifacts/ — trained weights (not committed)

[한국어](README.md) | **English**

Models made by the GPU stages are created here. They are large and can be rebuilt, so git ignores them. For this experiment (config `kormedmcqa_midm.yaml`) they are created under `kormedmcqa_midm/`.

| Folder | What | Stage that makes it |
|---|---|---|
| `A_bf16_seed42/` | BF16 LoRA model (safetensors with the LoRA merged in, about 22GB). Input to INT4 conversion and serving | `stage.sh a` |
| `A_bf16_seed42_run/` | Checkpoints during training (adapters) | `stage.sh a` |
| `B_int4_ptq_seed42/` | INT4 PTQ (about 7.2 GiB; it is a torchao pickle, so it is tied to the torch and torchao builds that made it) | `stage.sh b` |
| `C_int4_qat_seed42_bf16adapt/` | BF16 weights after QAT training (this is what you move to another GPU) | `stage.sh c` |
| `C_int4_qat_seed42/` | INT4 QAT | `stage.sh c`, `c_export` |
| `A_bf16_unsloth_seed42/` | BF16 LoRA model trained with Unsloth | `stage.sh a` with the Unsloth config |
| `sweep/`, `self_explain/` | Checkpoints from the experiments to raise the score, and the explanations written by Mi:dm (raw text, not committed) | [Step-by-step guide 7](../../docs/guide/07_recipe_sweep_en.md) |

To download and use the published weights, run `bash scripts/gpu/stage.sh fetch_hf` ([step-by-step guide 4](../../docs/guide/04_int4_en.md#scoring-on-a-different-gpu)).
