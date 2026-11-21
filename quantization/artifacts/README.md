# artifacts/ — 학습한 가중치 (커밋하지 않음)

**한국어** | [English](README_en.md)

GPU 단계가 만든 모델이 여기에 생깁니다. 크고 다시 만들 수 있어서 git이 무시합니다. 이번 실험(설정 `kormedmcqa_midm.yaml`)은 `kormedmcqa_midm/` 아래에 생깁니다.

| 폴더 | 무엇 | 만드는 단계 |
|---|---|---|
| `A_bf16_seed42/` | BF16 LoRA 모델(LoRA를 합친 safetensors, 약 22GB). INT4 변환과 서빙의 입력 | `stage.sh a` |
| `A_bf16_seed42_run/` | 학습 중 체크포인트(어댑터) | `stage.sh a` |
| `B_int4_ptq_seed42/` | INT4 PTQ(약 7.2 GiB, torchao pickle이라 만든 torch·torchao 빌드에 묶임) | `stage.sh b` |
| `C_int4_qat_seed42_bf16adapt/` | QAT 학습이 끝난 BF16 가중치(다른 GPU로 옮길 때 이것을 옮김) | `stage.sh c` |
| `C_int4_qat_seed42/` | INT4 QAT | `stage.sh c`, `c_export` |
| `A_bf16_unsloth_seed42/` | Unsloth로 학습한 BF16 LoRA 모델 | Unsloth 설정으로 `stage.sh a` |
| `sweep/`, `self_explain/` | 점수 올리기 실험의 체크포인트와 Mi:dm이 쓴 해설(원문, 커밋하지 않음) | [따라하기 7번](../../docs/guide/07_recipe_sweep.md) |

공개한 가중치를 받아 쓰려면 `bash scripts/gpu/stage.sh fetch_hf`를 씁니다([따라하기 4번](../../docs/guide/04_int4.md#다른-gpu에서-채점하기)).
