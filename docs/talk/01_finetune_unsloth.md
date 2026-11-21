# 1. 파인튜닝 — 같은 LoRA를 transformers+PEFT와 Unsloth로

**한국어** | [English](01_finetune_unsloth_en.md)

공통 조건(모델, 데이터, 프롬프트, 채점, 장비)은 [0. 실험 설정](00_setup.md)에 있습니다.

## 무엇을 바꿨나

**바꾼 것은 학습 엔진 하나입니다.**

- transformers `Trainer` + PEFT LoRA
- Unsloth `FastLanguageModel` + `get_peft_model`

데이터, 프롬프트, LoRA 설정, 학습 일정, seed, GPU는 같게 두었습니다. 학습이 끝나면 두 엔진 모두 LoRA를 원래 가중치에 합쳐(merge) BF16 모델 하나로 저장합니다. ②의 INT4 변환과 ③의 서빙은 이 모델을 그대로 받아 씁니다. ②·③에는 transformers+PEFT로 학습한 모델을 썼습니다.

## 학습 설정

| 항목 | 값 |
|---|---|
| 학습 데이터 | 2012~2020년 기출 3,401문항, 2 epoch ([예시 문항](00_setup.md#한-문항은-이렇게-생겼습니다)) |
| optimizer step | 426 (배치 8 × gradient accumulation 2 = 한 step에 16문항) |
| LoRA | rank 16, alpha 32, dropout 0, 대상 `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |
| 학습률 | 2e-4, warmup 3%, weight decay 0.01, gradient clipping 1.0 |
| 정밀도 | BF16 (4비트로 불러오지 않음) |
| 최대 길이 | 1,024토큰 |
| gradient checkpointing | 켬. transformers+PEFT는 기본 방식, Unsloth는 자체 방식(`use_gradient_checkpointing="unsloth"`) |
| 손실 | 정답 번호와 `<|eot_id|>`에만 걸림 |

> **측정 조건**
>
> - 실행: 2026-11-10 ~ 11-11, Azure H100 NVL 94GB 1장, seed 42, 각 엔진 1회
> - 학습 시간: 학습 반복(`trainer.train`)만 잰 값. 모델 불러오기와 병합·저장은 빠짐
> - GPU 메모리: 학습 도중 `nvidia-smi`로 직접 읽은 값. 최댓값이 아니고 읽은 시점의 값입니다. PyTorch가 잡아 둔 여유 메모리까지 들어갑니다
> - 정답률: vLLM 0.23으로 서빙한 상태에서 greedy로 잼. 2021년 1,059문항, 2022~2024년 3,009문항
> - 결과 파일: [`A_train_seed42.json`](../../quantization/results/kormedmcqa_midm/A_train_seed42.json), [`unsloth_A/A_train_seed42.json`](../../quantization/results/kormedmcqa_midm/unsloth_A/A_train_seed42.json), [`train_gpu_memory_observed_h100.json`](../../quantization/results/kormedmcqa_midm/train_gpu_memory_observed_h100.json)

## 결과

![transformers+PEFT와 Unsloth의 학습 시간과 GPU 메모리](figures/finetune_engines.svg)

| | transformers+PEFT | Unsloth |
|---|---|---|
| 학습 시간 (426 step) | 24.3분 (1,457.6초) | **15.7분** (943.9초) |
| 학습 중 GPU 메모리 (`nvidia-smi`) | 82.3 GiB | **25.5 GiB** |
| 평균 학습 손실 | 1.879 | 1.874 |
| 정답률, 2021년 1,059문항 | 70.4% (67.6~73.1) | 70.7% (67.9~73.4) |
| 정답률, 2022~2024년 3,009문항 | 72.3% (70.7~73.9) | 72.2% (70.6~73.8) |

괄호는 95% 신뢰구간입니다. 튜닝 전 모델은 같은 3,009문항에서 63.3%였습니다.

같은 문항끼리 짝지어 비교하면 두 엔진의 차이는 우연으로 설명됩니다.

| 비교 (같은 문항끼리) | 한쪽만 맞힌 문항 | McNemar p |
|---|---|---|
| transformers+PEFT 대 Unsloth, 2021년 | 36 대 39 | 0.82 |
| transformers+PEFT 대 Unsloth, 2022~2024년 | 113 대 110 | 0.89 |
| 튜닝 전 대 transformers+PEFT LoRA, 2022~2024년 | 164 대 436 | 0.000001 미만 |

"한쪽만 맞힌 문항"은 앞에 쓴 쪽만 맞힌 수 대 뒤에 쓴 쪽만 맞힌 수입니다.

**본 것.**
- 학습 시간이 35% 줄었습니다.
- `nvidia-smi`로 본 메모리는 3분의 1 아래로 내려갔습니다.
- 정답률과 학습 손실은 두 엔진이 같았습니다.
- LoRA 학습만으로 튜닝 전보다 9.0%p 올랐고 네 직군 모두 올랐습니다(직군별 수치는 [2. INT4 양자화](02_int4_quantization.md#직군별-정답률)).

**그럴 법한 이유.** Unsloth는 LoRA 계산을 묶은 자체 커널과, 활성값을 CPU로 내려 두는 자체 gradient checkpointing을 씁니다. 속도와 메모리 차이는 이 두 가지에서 왔을 가능성이 큽니다. 다만 프로파일링으로 나눠 재지는 않았습니다.

**이 결과로 말할 수 없는 것.**
- **메모리 숫자는 단위를 맞춰 읽어야 합니다.** 두 값 모두 `nvidia-smi`로 한 번씩 읽은 값입니다. transformers+PEFT 쪽은 PyTorch 기준 최대 할당량을 기록하지 않았습니다. Unsloth의 PyTorch 기준 최대 할당은 24.5 GiB입니다.
- **gradient checkpointing 방식이 다릅니다.** 차이 일부는 엔진이 아니라 이 설정에서 왔을 수 있고 따로 떼어 재지 않았습니다.
- **소프트웨어 환경이 조금 다릅니다.** Unsloth 환경은 torch 2.14.1, 나머지는 2.11.0입니다.
- **seed 하나, GPU 한 종류입니다.** 2022~2024년 문제에서 학습을 다시 했을 때 점수가 얼마나 달라지는지는 모릅니다. Unsloth 학습을 seed 43으로 한 번 더 돌려 2021년 문제로 재 보니 0.9~1.5%p 달랐습니다([5. 점수 올리기](05_recipe_sweep.md)). 위 표에서 두 엔진의 정답률 차이(2021년 문제 0.3%p, 2022~2024년 문제 0.1%p)는 이보다 작습니다.

## Unsloth를 쓰려면 필요했던 것

Mi:dm 2.0을 Unsloth 2026.11.3으로 그대로 불러오면 토크나이저에서 멈춥니다. 다시 해 보려는 분은 아래 두 가지를 알아 두면 시간이 덜 듭니다.

1. **transformers를 4.57.6으로 고정합니다.** Unsloth를 설치하면 transformers 5.17이 같이 깔리는데, 5.17은 Mi:dm 토크나이저를 읽지 못합니다(`TypeError: argument of type 'tokenizers.AddedToken' is not iterable`).
2. **토크나이저만 따로 불러 넘깁니다.** 4.57.6으로 맞춰도 GPU가 보이는 상태에서 `import unsloth`를 하면 토크나이저 로딩이 다시 깨집니다(`AttributeError: 'tokenizers.AddedToken' object has no attribute 'keys'`). GPU가 없는 환경에서는 재현되지 않았고 원인이 되는 패치는 찾지 못했습니다.
   - 우회: [`kormedmcqa_midm_unsloth.yaml`](../../quantization/configs/kormedmcqa_midm_unsloth.yaml)의 `train.unsloth_external_tokenizer: true`
   - 이 옵션을 켜면 Unsloth를 불러오지 않은 별도 프로세스에서 일반 transformers로 토크나이저를 읽어 넘깁니다.
   - 모델 로딩과 학습은 Unsloth가 합니다.

우회한 뒤에는 두 가지를 확인했습니다. 병합된 모델의 어휘가 131,384개 그대로인지, 그리고 그 모델의 토크나이저가 튜닝 전 모델과 같은 프롬프트 토큰과 정지 토큰을 내는지입니다(`scripts/gpu/stage.sh a_smoke`).

## 다시 돌리려면

GPU 환경 준비와 단계별 확인 기준은 [따라하기 2·3번](../guide/02_gpu_setup.md)에 있습니다. 학습 명령은 이렇습니다.

```bash
# transformers + PEFT
python -m quantization.v2_run a --config quantization/configs/kormedmcqa_midm.yaml --seed 42 --backend hf

# Unsloth (따로 만든 가상환경에서, 결과는 unsloth_A/ 폴더로 감)
python -m quantization.v2_run a --config quantization/configs/kormedmcqa_midm_unsloth.yaml --seed 42

# 2021년 문제로 중간 점검
python -m quantization.eval_vllm --config quantization/configs/kormedmcqa_midm.yaml --method A_bf16 --split selection_dev
```

학습된 모델은 Hugging Face에 공개했습니다: [`HyeonSang/Midm-2.0-KorMedMCQA`](https://huggingface.co/HyeonSang/Midm-2.0-KorMedMCQA)(transformers+PEFT로 학습한 BF16 LoRA 병합본).
