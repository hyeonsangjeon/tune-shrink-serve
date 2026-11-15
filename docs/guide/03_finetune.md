# 3. LoRA 학습

튜닝 전 모델의 점수를 재고, Mi:dm 2.0 Base에 LoRA를 학습해 BF16 모델 하나로 합칩니다. 같은 학습을 두 엔진(transformers+PEFT, Unsloth)으로 한 번씩 합니다. 뒤 단계(INT4, 서빙)는 transformers+PEFT로 학습한 모델을 씁니다.

| 필요한 것 | 걸린 시간 (H100 NVL) | 결과 문서 |
|---|---|---|
| [2번](02_gpu_setup.md)을 마친 VM | 학습: transformers+PEFT 24.3분, Unsloth 15.7분 | [1. 파인튜닝](../talk/01_finetune_unsloth.md) |

## 순서

```bash
bash scripts/gpu/stage.sh selftest           # 0. QAT 가짜 양자화가 서빙 INT4와 같은 방식인지 작은 행렬로 확인
bash scripts/gpu/stage.sh base_dev           # 1. 튜닝 전 모델, 2021년 1,059문항
BACKEND=hf bash scripts/gpu/stage.sh a_smoke # 2. 5 step 시험 학습 (끝나면 결과를 *_smoke로 옮겨 둠)
BACKEND=hf bash scripts/gpu/stage.sh a       # 3. 본학습 426 step (2 epoch)
BACKEND=hf bash scripts/gpu/stage.sh a_dev   # 4. 학습한 모델, 2021년 문제
bash scripts/gpu/stage.sh gate               # 5. 튜닝 전보다 2%p 이상 올랐는지 (못 넘으면 멈춤)
```

Unsloth로 같은 학습을 하려면 설정만 바꿉니다. 결과는 `unsloth_A/` 폴더로 따로 가서 위 결과를 덮어쓰지 않습니다.

```bash
CONFIG=quantization/configs/kormedmcqa_midm_unsloth.yaml bash scripts/gpu/stage.sh a_smoke
CONFIG=quantization/configs/kormedmcqa_midm_unsloth.yaml bash scripts/gpu/stage.sh a
```

- `BACKEND=hf`는 transformers+PEFT로 `~/venv-q`에서 학습합니다. 빼면 Unsloth로 `~/venv-unsloth`에서 학습합니다(설정의 기본값).
- 학습값(LoRA rank 16·alpha 32, 학습률 2e-4, 한 step에 16문항)은 [`kormedmcqa_midm.yaml`](../../quantization/configs/kormedmcqa_midm.yaml)의 `train:`에 있습니다.

## 확인

결과 파일은 `quantization/results/kormedmcqa_midm/`에 남습니다.

| 단계 | 파일 | 볼 것 |
|---|---|---|
| `selftest` | `qat_selftest.json` | `ok: true` |
| `base_dev` | `eval_vllm_base_seed42_selection_dev.json` | `n_unparsed`(번호를 못 읽은 답)가 거의 0. 많으면 프롬프트부터 고칩니다. 이번 실행의 정답률은 같은 파일에 있습니다 |
| `a_smoke` | 로그 `~/work/logs/a_smoke.log` | 오류 없이 병합 폴더가 생기고 `vocab_ok: true`, `prompt_ids_same: true` |
| `a` | `A_train_seed42.json` | `global_step` 426, `train_seconds`. 로그에 `[tokenize] WARNING`(길이 초과)이 없어야 합니다 |
| `a_dev` | `eval_vllm_A_bf16_seed42_selection_dev.json` | `n_eval` 1,059, `n_unparsed` 0에 가까움 |
| `gate` | `gate_dev_seed42.json` | `pass: true`. 두 결과가 같은 GPU에서 나왔는지도 확인합니다 |

학습한 모델은 `quantization/artifacts/kormedmcqa_midm/A_bf16_seed42/`(BF16 safetensors, 약 22GB)에 생깁니다. 가중치는 커밋하지 않습니다.

## 안 될 때

- **Unsloth가 Mi:dm 토크나이저를 못 읽음:** 설정의 `train.unsloth_external_tokenizer: true`로 토크나이저를 따로 불러 넘깁니다(Unsloth 설정에 이미 들어 있습니다). 자세한 것은 [1번 결과 문서](../talk/01_finetune_unsloth.md#unsloth를-쓰려면-필요했던-것).
- **본학습이 5 step에서 이어 감:** 손으로 시험 학습을 돌리고 `*_run/` 폴더를 남긴 경우입니다. 폴더를 치우고 다시 돌립니다. `stage.sh a_smoke`는 이 일을 대신 합니다.
- **Spot VM이 회수됨:** VM을 다시 켠 뒤 `RESUME=1 BACKEND=hf bash scripts/gpu/stage.sh a`로 마지막 체크포인트(100 step마다)에서 이어 갑니다.
- **`gate`가 멈춤(종료 코드 3):** 학습이 2021년 문제에서 2%p 이상 오르지 않은 것입니다. 학습 로그와 `n_unparsed`를 먼저 봅니다.

다음: [4. INT4로 줄이고 채점](04_int4.md)
