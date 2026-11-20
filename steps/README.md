# 단계별 코드: 학습 → INT4 → 채점 → 서빙

발표의 학습, INT4 양자화, 채점, vLLM 서빙을 **단계마다 Python 파일 하나**로 정리했습니다. 파일마다 맨 위에 설정값이 있고, 그 아래 함수가 실행 순서대로 있습니다. 맨 아래 `main()`은 그 함수들을 차례로 부르기만 합니다. 함수마다 무엇을 왜 하는지 주석으로 적었습니다.

| 단계 | 파일 | 하는 일 | 핵심 API | H100 NVL에서 걸린 시간 |
|---|---|---|---|---|
| 1 | [`1_lora_train.py`](1_lora_train.py) | LoRA 학습 후 BF16 모델 하나로 합치기 | PEFT `LoraConfig`·`get_peft_model`, Unsloth `FastLanguageModel`, `Trainer` | 24분 (Unsloth 16분) |
| 2 | [`2_int4_ptq.py`](2_int4_ptq.py) | 학습 없이 INT4로 바꾸기 (PTQ) | TorchAO `Int4WeightOnlyConfig` + transformers `TorchAoConfig` | 17초 |
| 3 | [`3_int4_qat.py`](3_int4_qat.py) | INT4를 흉내 내며 더 학습한 뒤 INT4로 (QAT) | TorchAO `quantize_` + `QATConfig(step="prepare" / "convert")` | 1시간 42분 |
| 4 | [`4_evaluate.py`](4_evaluate.py) | 2022~2024년 3,009문항으로 정답률 재기 | vLLM `LLM.generate` | 모델마다 1~6분 |
| 5 | [`5_serve.sh`](5_serve.sh), [`5_ask.py`](5_ask.py) | OpenAI 호환 서버로 띄우고 질문 보내기 | `vllm.entrypoints.openai.api_server` | — |

공통으로 쓰는 것(모델 이름, 폴더, 데이터, 프롬프트, INT4 형식)은 [`common.py`](common.py)에 있습니다.

## 실행

GPU VM(80GB 이상 1장)에서 레포 최상위 폴더에서 실행합니다. 디스크는 200GB 이상 비어 있어야 합니다. 모델 하나가 22GB이고, 3단계가 학습하는 동안 남기는 체크포인트가 하나에 42GB입니다. 저장하는 동안에는 체크포인트 세 개가 같이 있어 학습 폴더가 123GB까지 커집니다(체크포인트는 단계가 끝나면 지웁니다). 가상환경은 [따라하기 2번](../docs/guide/02_gpu_setup.md)의 `bash scripts/gpu/bootstrap.sh`가 만듭니다. 모델과 데이터는 공개되어 있어 Hugging Face 로그인 없이 받습니다.

```bash
source ~/venv-q/bin/activate

python steps/1_lora_train.py                  # → outputs/1_lora
python steps/2_int4_ptq.py                    # → outputs/2_int4_ptq
python steps/3_int4_qat.py                    # → outputs/3_int4_qat

python steps/4_evaluate.py                    # LoRA 모델 채점
python steps/4_evaluate.py outputs/2_int4_ptq
python steps/4_evaluate.py outputs/3_int4_qat
python steps/4_evaluate.py base               # 튜닝 전 모델

bash steps/5_serve.sh outputs/2_int4_ptq      # 서버 (Ctrl-C로 끔)
python steps/5_ask.py                         # 다른 터미널에서 질문 3개
```

각 단계의 기본 입력 폴더가 앞 단계의 출력 폴더라서, 인자 없이 차례로 돌리면 이어집니다. 모델은 `outputs/`에 생기고 git에는 올라가지 않습니다.

**먼저 짧게 돌려 보기:** 학습을 5 step만 하고 50문항만 채점해, 전체 과정이 에러 없이 이어지는지 봅니다. 모델을 받고(약 23GB) 올리고 저장하는 시간은 그대로 걸리고, 디스크도 가장 많을 때 120GB쯤 씁니다. 5 step만 해도 학습이 끝날 때 체크포인트를 하나 저장하기 때문입니다.

```bash
python steps/1_lora_train.py --max-steps 5
python steps/2_int4_ptq.py
python steps/3_int4_qat.py --max-steps 5
python steps/4_evaluate.py outputs/2_int4_ptq --split dev --limit 50
```

**Unsloth로 학습하기:** Unsloth는 다른 가상환경에 있습니다. 결과는 `outputs/1_lora_unsloth`로 따로 갑니다.

```bash
source ~/venv-unsloth/bin/activate
python steps/1_lora_train.py --engine unsloth
```

## 발표 숫자와 맞춰 보기

4단계를 돌리면 정답률이 나옵니다. 2026-11-16~17에 발표 때와 같은 H100 NVL, 같은 가상환경에서 이 폴더의 코드로 1단계부터 다시 학습하고, 2022~2024년 3,009문항으로 채점했습니다(seed 42, 모델마다 한 번). 아래 표는 발표 때 값과 그 값입니다. "답이 같은 문항"은 두 모델이 같은 보기를 고른 문항 수입니다.

| 모델 | 발표 정답률 (맞힌 수) | 이 폴더로 다시 학습한 값 (맞힌 수) | 답이 같은 문항 | 학습 시간 (발표 → 다시) |
|---|---|---|---|---|
| 튜닝 전 모델 (`base`) | 63.3% (1,904) | 63.3% (1,904) | 3,009 | 학습 없음 |
| 1단계 BF16 LoRA (transformers+PEFT) | 72.3% (2,176) | 72.3% (2,177) | 2,686 | 24.3분 → 24.7분 |
| 1단계 BF16 LoRA (Unsloth) | 72.2% (2,173) | 72.5% (2,182) | 2,732 | 15.7분 → 16.5분 |
| 2단계 INT4 PTQ | 69.5% (2,090) | 69.1% (2,078) | 2,596 | 학습 없음 |
| 3단계 INT4 QAT | 69.7% (2,097) | 70.1% (2,109) | 2,612 | 101.9분 → 102.9분 |

튜닝 전 모델은 학습이 없어서 2026-11-16에 다시 채점만 했습니다. INT4 PTQ와 INT4 QAT는 다시 학습한 transformers+PEFT 모델에서 만들었습니다. 발표 때 결과는 [`eval_vllm_*_seed42_h100.json`](../quantization/results/kormedmcqa_midm/)에, 다시 학습한 결과는 [`steps_rerun/`](../quantization/results/kormedmcqa_midm/steps_rerun/)에 있습니다.

- **처음부터 다시 학습하면 문항의 9~14%에서 답이 바뀌었습니다. 정답률은 0.4%p 안에서 움직였습니다.** 맞힌 수로는 +1, +9, −12, +12개입니다. 같은 문항끼리 비교한 McNemar p는 모두 0.5 이상이었습니다([`paired_compare.json`](../quantization/results/kormedmcqa_midm/steps_rerun/paired_compare.json)). 한 모델의 95% 신뢰구간이 ±1.6%p쯤이므로, 이 정도 차이는 같은 결과로 봅니다.
- **답이 바뀌는 것은 코드 차이가 아니라 실행마다 생기는 흔들림으로 봅니다.** 첫 step의 loss는 발표 때와 같았지만(PEFT 17.3403, Unsloth 17.3405), 그 뒤로는 step마다 조금씩 달라졌습니다(발표 때와 가장 클 때 0.049 차이). 이 폴더의 코드로 같은 날 두 번 학습한 것끼리도 첫 110 step에서 loss가 가장 클 때 0.019 달랐습니다. 학습한 모델 파일도 발표 때 파일과 바이트 단위로 달랐습니다. GPU 연산 순서가 매번 같지 않은 것이 가장 그럴 법한 이유지만 따로 재지는 않았습니다.
- **BF16에서 INT4로 바꿀 때 내려가는 폭은 발표 때와 비슷했습니다.** 다시 학습한 BF16 LoRA 모델 대비 INT4 PTQ −3.3%p, INT4 QAT −2.3%p입니다(발표 때 −2.9%p, −2.6%p). 같은 문항끼리 비교한 McNemar p는 둘 다 0.001보다 작았습니다.
- **INT4 QAT와 INT4 PTQ의 차이는 실행마다 달랐습니다.**

  | 비교 (같은 문항끼리) | 정답률 차이 | 한쪽만 맞힌 문항 | McNemar p |
  |---|---|---|---|
  | 발표 때: INT4 PTQ 대 INT4 QAT | +0.2%p | 97 대 104 | 0.67 |
  | 다시 학습: INT4 PTQ 대 INT4 QAT | +1.0%p | 70 대 101 | 0.02 |
  | 다시 학습: INT4 PTQ 대 끊었다 이어 간 INT4 QAT | +0.6%p | 79 대 98 | 0.18 |

  세 번 모두 QAT가 높았지만 차이는 0.2~1.0%p로 갈렸고, p가 0.05보다 작은 것은 한 번뿐입니다. 아래 두 줄은 같은 BF16 LoRA 모델과 같은 INT4 PTQ 모델을 씁니다. McNemar는 두 모델을 고정해 두고 문항에서 오는 흔들림만 봅니다. 학습을 다시 할 때의 흔들림은 넣지 않습니다. 그래서 p=0.02 한 번으로 "QAT가 PTQ보다 낫다"고 할 수 없고, 발표 때의 p=0.67로 "같다"고 할 수도 없습니다.
- **같은 INT4 파일을 다시 채점해도 몇 문항 답이 바뀝니다.** 2026-11-16에 발표 때 BF16 LoRA 모델을 2단계로 다시 INT4로 바꿨더니 발표 때 INT4 파일과 바이트까지 같았습니다. 그 파일을 다시 채점하자 5문항의 답이 바뀌었고 맞힌 수는 2개 줄었습니다(69.458% → 69.392%). 튜닝 전 모델과 BF16 LoRA 모델은 다시 채점해도 3,009문항의 답이 모두 같았습니다. INT4에서만 답이 바뀌는 이유는 따로 재지 않았습니다.
- **튜닝 전 모델 점수는 거의 같아야 합니다.** 학습이 없고, 같은 모델 revision, 같은 프롬프트, temperature 0이기 때문입니다. GPU나 vLLM 버전이 다르면 몇 문항 달라질 수 있습니다. 1%p 넘게 다르면 프롬프트나 라이브러리 버전부터 봅니다.
- **학습한 모델은 위만큼 다를 수 있습니다.** 발표 실험에서 seed만 바꿨을 때는 2021년 문제 점수가 0.9~1.5%p 움직였습니다([5. 점수 올리기](../docs/talk/05_recipe_sweep.md)). 다시 학습은 이번 한 번(끊었다 이어 간 학습까지 치면 두 번)만 했으므로, 실행마다 얼마나 흔들리는지 범위를 이 결과로 정할 수는 없습니다.
- 발표 때 설치한 버전은 [`env_h100.json`](../quantization/results/kormedmcqa_midm/env_h100.json)에 있습니다. 다시 학습할 때의 버전도 같았습니다([`compare.json`](../quantization/results/kormedmcqa_midm/steps_rerun/compare.json)의 `versions`).

## 학습을 끊었다가 이어 가기 (`--resume`)

2026-11-16~17에 1단계(transformers+PEFT)와 3단계를 학습 도중 강제로 끄고(SIGKILL, `kill -9`와 같습니다), 같은 명령에 `--resume`을 붙여 끝까지 이어 갔습니다. 괄호 안은 같은 코드로 끊지 않고 끝까지 간 학습(위 표의 "이 폴더로 다시 학습한 값")의 값입니다.

| | 1단계 BF16 LoRA | 3단계 INT4 QAT |
|---|---|---|
| 끈 때 | 110 step (시작하고 425초) | 160 step (시작하고 1,685초) |
| 남아 있던 체크포인트 | `checkpoint-100` | `checkpoint-150` |
| 이어 간 구간과 걸린 시간 | 100 → 426 step, 18.8분 | 150 → 600 step, 77.3분 |
| 이어 간 뒤 처음 찍힌 loss | 1.6023 (1.6004), 110 step | 1.3998 (1.3825), 160 step |
| learning rate가 끊지 않은 학습과 같았던 기록 | 32번 중 32번 | 23번 중 23번 |
| 마지막 loss | 1.4373 (1.4348) | 1.3145 (1.3151) |
| 정답률 | 72.6% (72.3%) | 69.7% (70.1%) |
| 답이 같은 문항 | 2,715 / 3,009 | 2,907 / 3,009 |
| 같은 문항끼리 McNemar p | 0.62 | 0.17 |

- **체크포인트 위치에서 이어졌습니다.** learning rate는 기록된 모든 step에서 끊지 않은 학습과 같았습니다. 이어 간 뒤 처음 찍힌 loss도 끊지 않은 학습과 0.002(1단계), 0.017(3단계) 차이였습니다. 처음부터 다시 시작했다면 1단계 loss는 첫 step 값인 17.3 근처에서 다시 시작했을 것입니다.
- **이어 간 학습과 끊지 않은 학습의 차이는 두 번 학습한 차이와 비슷한 크기입니다.** loss는 가장 클 때 0.058(1단계), 0.017(3단계) 달랐습니다. 끊지 않고 두 번 학습한 것끼리도 1단계 loss가 가장 클 때 0.049 달랐습니다(이 폴더 대 발표). 정답률 차이도 위 표의 다시 학습 차이(0.4%p 안)와 같은 범위입니다.
- 3단계는 두 학습이 같은 BF16 LoRA 모델에서 시작합니다. 답이 같은 문항이 2,907개로 1단계(2,715개)보다 많은 것은 그 때문으로 봅니다.
- 두 번 다 학습이 끝난 뒤 체크포인트 폴더(`outputs/*_run`)가 지워졌습니다.
- **3단계 체크포인트 디스크:** 이어 간 3단계 학습에서 30초마다 학습 폴더 크기를 쟀습니다. 체크포인트 하나가 42GB였고, 학습 폴더가 가장 컸을 때는 123GB였습니다. 체크포인트를 두 개까지만 남기도록 했지만(`save_total_limit=2`) 새 체크포인트를 저장하는 동안에는 세 개가 같이 있기 때문입니다.

## 발표 숫자를 만든 코드와의 관계

발표 숫자는 [`scripts/gpu/stage.sh`](../scripts/gpu/stage.sh)가 부르는 [`quantization/v2_run.py`](../quantization/v2_run.py)·[`v2_pipeline.py`](../quantization/v2_pipeline.py)·[`eval_vllm.py`](../quantization/eval_vllm.py)로 만들었습니다. 그 코드는 점수 올리기 실험의 옵션, 중간 점검, 결과 파일 이름 규칙까지 들어 있어 길고 읽기 어렵습니다. 이 폴더는 같은 일을 읽기 쉽게 다시 쓴 것입니다.

- **같은 것:** 데이터 나누기, 프롬프트, 학습 데이터 만들기(정답 토큰에만 손실), 채점은 그 코드의 함수를 그대로 부릅니다([`common.py`](common.py)). 학습률, LoRA rank, 배치, QAT step 같은 설정값이 YAML 설정과 같은지는 CI가 확인합니다([`test_steps.py`](../quantization/tests/test_steps.py)).
- **GPU에서 확인한 것 (H100 NVL, 발표 때와 같은 가상환경):**
  - 2026-11-16, 짧게: 1단계를 transformers+PEFT와 Unsloth로 5 step씩, 3단계를 발표 때 BF16 LoRA 모델에서 5 step 돌렸습니다. 첫 step의 loss가 발표 때 학습과 같았습니다(PEFT 17.3403, Unsloth 17.3405, QAT 1.3766). 학습 문항, 순서, LoRA 초깃값, 손실 계산이 같다는 뜻입니다. 2단계는 발표 때 BF16 LoRA 모델을 16.6초에 INT4로 바꿨고(21.52 → 7.20 GiB), 파일이 발표 때 INT4 파일과 바이트까지 같았습니다.
  - 2026-11-16~17, 처음부터 끝까지: 1단계(426 step, 두 엔진)와 3단계(600 step)를 끝까지 학습하고 2·4단계를 이어 돌렸습니다. 결과는 위 표에 있습니다. 1·3단계의 `--resume`도 확인했습니다(위 절).
  - 5단계 서버를 다시 학습한 INT4 QAT 모델로 띄우고 질문 3개를 보냈습니다. 서버의 답(2, 5, 1)이 4단계 채점 때 같은 문항의 답과 같았습니다. prefix caching을 켠 서버와 끈 서버(`NO_PREFIX_CACHE=1`) 둘 다 같았습니다. 3문항은 서버가 제대로 도는지 보는 용도라, 맞힌 수(1/3)는 정답률로 읽지 않습니다.
- **CPU에서 확인한 것:** 작은 가짜 모델(Mi:dm 토크나이저, 층 2개)로 1단계와 3단계를 돌렸습니다. 2 step에서 멈춘 학습을 `--resume`으로 4 step까지 이어 가는 것도 확인했습니다.
- **다시 학습한 기록 ([`steps_rerun/`](../quantization/results/kormedmcqa_midm/steps_rerun/)):** 이름에 `r1_`, `r3_`가 들어간 파일이 끊었다 이어 간 학습입니다.
  - `1_lora_train*.json`, `2_int4_ptq.json`, `3_int4_qat.json`, `r1_lora_train.json`, `r3_qat.json`: 각 단계가 남긴 기록(학습 시간, 마지막 loss, 크기)
  - `4_eval_*_test.json`: 문항마다 정답과 모델 답(문제 본문은 없습니다)
  - `loss_logs.json`: 학습 중 찍힌 loss와 learning rate
  - `compare.json`: 발표 기록과 맞춘 값, 끊었다 이어 간 학습과의 비교, 체크포인트 디스크, 버전
  - `paired_compare.json`: 같은 문항끼리 비교(`python -m quantization.kormedmcqa compare <결과 1> <결과 2>`로 다시 만들 수 있습니다)

## 안 될 때

- **Unsloth가 Mi:dm 토크나이저를 못 읽음** (`'AddedToken' object has no attribute 'keys'`): `1_lora_train.py`의 `load_with_unsloth()`가 이미 우회합니다. 그래도 나면 Unsloth 버전이 [`requirements/unsloth.txt`](../requirements/unsloth.txt)와 같은지 봅니다.
- **디스크가 가득 참** (`No space left on device`): `df -h`로 남은 공간을 봅니다. 3단계는 시작할 때 150GB 이상 비어 있어야 합니다. 공간을 비운 뒤 같은 명령에 `--resume`을 붙이면, 1·3단계 학습이 마지막 체크포인트부터 이어집니다.
- **학습이 중간에 끊김:** 같은 명령에 `--resume`을 붙입니다. 체크포인트는 1단계 100 step, 3단계 150 step마다 남습니다. 마지막 체크포인트 다음 step부터 이어지므로, 그 뒤에 한 학습은 다시 합니다(위 [학습을 끊었다가 이어 가기](#학습을-끊었다가-이어-가기---resume)).
- **3단계에서 GPU 메모리 부족:** `--batch-size 2 --grad-accum 8`로 돌립니다. 한 step에 16문항인 것은 같습니다.
- **INT4 폴더를 읽다가 에러:** INT4 모델은 pickle로 저장되어, 저장할 때와 읽을 때 torch·torchao 버전이 같아야 합니다. 같은 가상환경에서 만들고 읽습니다.
- **5단계 서버가 안 뜸:** 다른 vLLM 서버가 같은 GPU나 포트(8000)를 쓰고 있는지 `nvidia-smi`로 봅니다. 포트는 `PORT=8001 bash steps/5_serve.sh`로 바꿉니다.
- 그 밖의 알려진 함정은 [따라하기 8번](../docs/guide/08_troubleshooting.md)에 있습니다.
