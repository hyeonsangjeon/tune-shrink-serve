# 2. INT4 양자화 — TorchAO PTQ와 QAT

공통 조건은 [0. 실험 설정](00_setup.md)에 있습니다. 출발점은 [1. 파인튜닝](01_finetune_unsloth.md)에서 transformers+PEFT로 학습한 BF16 LoRA 모델입니다.

## 무엇을 바꿨나

같은 BF16 LoRA 모델을 두 가지 방법으로 INT4로 줄였습니다. 저장 형식은 둘이 같습니다.

| | INT4 PTQ (학습 후 양자화) | INT4 QAT (양자화를 고려한 학습) |
|---|---|---|
| 하는 일 | 학습 없이 가중치만 INT4로 바꿈 | 가짜 양자화(fake quantization)를 끼운 채로 더 학습한 뒤 INT4로 바꿈 |
| TorchAO 설정 | `Int4WeightOnlyConfig(group_size=128)`, 형식 `TILE_PACKED_TO_4D` | 학습: `QATConfig(step="prepare")` → 학습 → `QATConfig(step="convert")`. 저장은 PTQ와 같은 설정 |
| 추가 학습 | 없음 | 600 step, 학습률 1e-5, 배치 4 × accumulation 4, 8비트 AdamW, gradient checkpointing |
| 학습한 가중치 | — | 출력층(lm_head)을 뺀 모든 linear, 10,469M개 |
| 걸린 시간 | 17초 (불러오기·변환·저장) | 학습 1시간 42분 (6,111초) |

- **가중치만 INT4입니다.** 계산은 BF16으로 합니다(weight-only). 128개 가중치마다 배율(scale) 하나를 둡니다.
- **`TILE_PACKED_TO_4D`는 PyTorch에 들어 있는 tinygemm INT4 커널을 씁니다.** 따로 커널 라이브러리를 깔지 않아도 됩니다. 이 형식이 ③의 속도 결과를 좌우합니다.
- **두 INT4 모델은 크기가 똑같습니다(7.20 GiB).** QAT는 코드에서 출력층을 명시적으로 뺍니다. PTQ도 크기가 같으므로 양자화에서 빼는 범위가 같다고 봅니다(직접 비교하지는 않았습니다).
- **QAT의 가짜 양자화가 실제 서빙 INT4와 같은 방식인지 학습 전에 확인했습니다**([`qat_selftest.json`](../../quantization/results/kormedmcqa_midm/qat_selftest.json)).
- **QAT는 양자화를 고려하는 것에 더해 같은 학습 데이터로 600 step을 더 학습합니다**(약 2.8 epoch). PTQ와 비교할 때 이 차이도 함께 들어갑니다.

> **측정 조건**
>
> - 실행: 2026-11-10 ~ 11-11, H100 NVL 94GB 1장, seed 42, 모델마다 1회
> - 평가: 2022~2024년 3,009문항, 모델마다 한 번만. vLLM 0.23, INT4는 `quantization="torchao"`로 서빙
> - INT4 모델은 평가와 같은 GPU·같은 소프트웨어에서 만들었습니다. INT4 파일은 만든 torch·torchao 빌드에 묶이기 때문입니다(pickle 저장)
> - 결과 파일: `eval_vllm_*_seed42_h100.json`, [`compare_final_seed42_h100.json`](../../quantization/results/kormedmcqa_midm/compare_final_seed42_h100.json), [`artifact_sizes.json`](../../quantization/results/kormedmcqa_midm/artifact_sizes.json)

## 결과

![2022~2024년 3,009문항 정답률: 튜닝 전, BF16 LoRA 두 엔진, INT4 PTQ, INT4 QAT](figures/accuracy_test.svg)

| 모델 | 정답률 (95% 신뢰구간) | 크기 |
|---|---|---|
| 튜닝 전 Mi:dm 2.0 Base | 63.3% (61.5~65.0) | 21.51 GiB |
| BF16 LoRA | **72.3%** (70.7~73.9) | 21.51 GiB |
| INT4 PTQ | 69.5% (67.8~71.1) | **7.20 GiB** |
| INT4 QAT | 69.7% (68.0~71.3) | **7.20 GiB** |

| 비교 (같은 문항끼리) | 정답률 차이 | 한쪽만 맞힌 문항 | McNemar p |
|---|---|---|---|
| BF16 LoRA 대 INT4 PTQ | −2.9%p | 198 대 112 | 0.000001 |
| BF16 LoRA 대 INT4 QAT | −2.6%p | 205 대 126 | 0.000017 |
| INT4 PTQ 대 INT4 QAT | +0.2%p | 97 대 104 | 0.67 |

"한쪽만 맞힌 문항"은 앞에 쓴 모델만 맞힌 수 대 뒤에 쓴 모델만 맞힌 수입니다. 첫 줄이면 BF16 LoRA만 맞힌 문항이 198개, INT4 PTQ만 맞힌 문항이 112개입니다.

정답률 차이는 반올림 전 값으로 계산했습니다. 그래서 결과 표의 반올림된 정답률끼리 빼면 0.1%p 다를 수 있습니다(예: 72.3 − 69.5 = 2.8, 반올림 전 차이는 2.86).

### 직군별 정답률

2022~2024년 3,009문항, 단위 %. 본문의 차이(%p)는 반올림 전 값으로 계산했습니다.

| 직군 (문항 수) | 튜닝 전 | BF16 LoRA | INT4 PTQ | INT4 QAT |
|---|---|---|---|---|
| 의사 (435) | 56.3 | 72.2 | 68.0 | 69.7 |
| 간호사 (878) | 73.2 | 81.8 | 78.8 | 78.8 |
| 약사 (885) | 67.7 | 74.7 | 70.6 | 71.1 |
| 치과의사 (811) | 51.4 | 59.6 | 58.8 | 58.3 |

### 연도별 정답률

| 시험 연도 | 튜닝 전 | BF16 LoRA | INT4 PTQ | INT4 QAT |
|---|---|---|---|---|
| 2022 | 63.9 | 72.1 | 70.7 | 70.2 |
| 2023 | 64.4 | 74.5 | 70.8 | 71.2 |
| 2024 | 61.4 | 70.2 | 66.8 | 67.5 |

**본 것.**
- INT4로 바꾸면 크기가 3분의 1이 됩니다(21.51 → 7.20 GiB).
- 그 대신 정답률이 2.6~2.9%p 내려갔습니다. 같은 문항끼리 비교해도 우연으로 보기 어려운 차이입니다.
- 이 실행에서 QAT는 PTQ보다 나아지지 않았습니다(+0.2%p, p=0.67). 같은 코드로 다시 학습했을 때는 결과가 달랐습니다(아래 「이 결과로 말할 수 없는 것」 첫 항목).
- 두 INT4 모델 모두 튜닝 전보다는 6%p 넘게 높고 합격 참고선 60%를 넘습니다.
- 정답률이 내려간 정도는 직군마다 다릅니다. BF16 LoRA와 비교하면 의사·간호사·약사는 2.5~4.1%p 내려갔고 치과의사는 0.7~1.2%p만 내려갔습니다.

**그럴 법한 이유.** weight-only INT4는 128개 가중치를 배율 하나로 반올림합니다. 그 반올림 오차가 정답 선택을 일부 바꿨다고 보는 게 가장 단순한 설명입니다. 이 실행에서 QAT가 이 오차를 메우지 못한 데는 몇 가지 이유가 있을 수 있습니다.

- 학습률 1e-5 · 600 step이 회복에 부족했을 수 있습니다.
- 이미 LoRA로 맞춘 모델에서는 더 학습해도 얻을 것이 적었을 수 있습니다.

이번 측정으로는 이 중 무엇인지 가릴 수 없습니다.

**이 결과로 말할 수 없는 것.**
- **QAT가 PTQ보다 나은지 정할 수 없습니다.** 2026-11-16~17에 [`steps/`](../../steps/README.md) 코드로 LoRA부터 다시 학습했을 때는 QAT가 PTQ보다 1.0%p 높았습니다(한쪽만 맞힌 문항 70 대 101, McNemar p=0.02). 같은 BF16 LoRA 모델에서 QAT를 학습 도중 끊었다가 이어 간 경우는 +0.6%p(p=0.18)였습니다. McNemar는 두 모델을 고정해 두고 문항에서 오는 흔들림만 봅니다. 학습을 다시 할 때의 흔들림은 넣지 않으므로, 이 실행의 p=0.67로 "같다"고도, 다시 학습한 실행의 p=0.02로 "낫다"고도 할 수 없습니다. 발표 자료를 만든 뒤 2026-11-17에 따로 돌린 [대조 실험](../qat_control.md)(같은 BF16 LoRA 모델, seed 3개)에서도 QAT − PTQ 평균은 +0.42%p로 정하지 못했습니다. 다만 가짜 양자화 없이 같은 600 step만 더 학습한 모델은 INT4로 바꿨을 때 PTQ와 거의 같았습니다(평균 −0.13%p). QAT가 PTQ보다 높게 나온 만큼은 추가 학습으로 설명되지 않습니다.
- **"QAT는 효과가 없다"고 일반화할 수 없습니다.** 학습률, step 수, QAT를 거는 범위를 바꿔 보지 않았습니다.
- **이전 실험과 인과로 비교하지 않습니다.** 이전 실험(Qwen3-8B, KorQuAD)에서는 QAT가 PTQ보다 F1을 회복했습니다. 그때와 지금은 모델과 데이터가 함께 바뀌었으므로 결과가 다른 이유를 어느 한쪽으로 돌릴 수 없습니다.
- **INT4 방식 하나만 쟀습니다.** weight-only, group 128, tile-packed 형식입니다. GPTQ, AWQ, 활성값까지 줄이는 방식(W4A8 등)은 재지 않았습니다.
- **seed 하나, GPU 한 종류입니다.**

## 다시 돌리려면

[따라하기 2번](../guide/02_gpu_setup.md)의 환경을 쓴 뒤(단계별 확인 기준은 [따라하기 4번](../guide/04_int4.md)), BF16 LoRA 모델이 있는 상태에서 실행합니다.

```bash
CFG=quantization/configs/kormedmcqa_midm.yaml
python -m quantization.v2_run selftest --config $CFG            # QAT 가짜 양자화가 서빙 INT4와 같은 방식인지 확인
python -m quantization.v2_run b --config $CFG --seed 42         # INT4 PTQ
python -m quantization.v2_run c --config $CFG --seed 42         # INT4 QAT (학습 후 INT4로 저장)
python -m quantization.eval_vllm --config $CFG --method B_int4_ptq   # 2022~2024년 문제, 한 번만
python -m quantization.eval_vllm --config $CFG --method C_int4_qat
python -m quantization.kormedmcqa compare <모델 1 결과 JSON> <모델 2 결과 JSON>   # 같은 문항끼리 McNemar
```

다른 GPU에서 평가하려면 INT4 파일은 옮기지 말고 그 GPU에서 다시 만듭니다. 방법은 둘입니다.

- **INT4 PTQ:** BF16 LoRA 모델을 받아 `v2_run b`로 다시 만듭니다.
- **INT4 QAT:** QAT 학습이 끝난 BF16 가중치([`HyeonSang/Midm-2.0-KorMedMCQA-QAT-bf16`](https://huggingface.co/HyeonSang/Midm-2.0-KorMedMCQA-QAT-bf16))를 받아 `v2_run c --export-only`로 만듭니다.
