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

GPU VM(80GB 이상 1장)에서 레포 최상위 폴더에서 실행합니다. 가상환경은 [따라하기 2번](../docs/guide/02_gpu_setup.md)의 `bash scripts/gpu/bootstrap.sh`가 만듭니다. 모델과 데이터는 공개되어 있어 Hugging Face 로그인 없이 받습니다.

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

**먼저 짧게 돌려 보기:** 학습을 5 step만 하고 50문항만 채점해, 전체 과정이 에러 없이 이어지는지 봅니다. 모델을 받고(약 23GB) 올리고 저장하는 시간은 그대로 걸립니다.

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

4단계를 돌리면 정답률이 나옵니다. 발표 때 잰 값은 이렇습니다(2022~2024년 3,009문항, H100 NVL, seed 42).

| 모델 | 발표 정답률 | 결과 JSON |
|---|---|---|
| 튜닝 전 (`base`) | 63.3% | [`eval_vllm_base_seed42_h100.json`](../quantization/results/kormedmcqa_midm/eval_vllm_base_seed42_h100.json) |
| 1단계 LoRA (transformers+PEFT) | 72.3% | [`eval_vllm_A_bf16_seed42_h100.json`](../quantization/results/kormedmcqa_midm/eval_vllm_A_bf16_seed42_h100.json) |
| 1단계 LoRA (Unsloth) | 72.2% | [`unsloth_A/eval_vllm_A_bf16_seed42_h100.json`](../quantization/results/kormedmcqa_midm/unsloth_A/eval_vllm_A_bf16_seed42_h100.json) |
| 2단계 INT4 PTQ | 69.5% | [`eval_vllm_B_int4_ptq_seed42_h100.json`](../quantization/results/kormedmcqa_midm/eval_vllm_B_int4_ptq_seed42_h100.json) |
| 3단계 INT4 QAT | 69.7% | [`eval_vllm_C_int4_qat_seed42_h100.json`](../quantization/results/kormedmcqa_midm/eval_vllm_C_int4_qat_seed42_h100.json) |

- **튜닝 전 모델 점수는 거의 같아야 합니다.** 학습이 없고, 같은 모델 revision, 같은 프롬프트, temperature 0이기 때문입니다. GPU나 vLLM 버전이 다르면 몇 문항 달라질 수 있습니다. 1%p 넘게 다르면 프롬프트나 라이브러리 버전부터 봅니다.
- **학습한 모델은 조금 다를 수 있습니다.** GPU 연산 순서에 따라 학습 결과가 조금씩 달라집니다. 발표 실험에서 seed만 바꿨을 때 2021년 문제 점수가 0.9~1.5%p 움직였습니다([5. 점수 올리기](../docs/talk/05_recipe_sweep.md)). 그 정도 차이는 같은 결과로 봅니다.
- 발표 때 설치한 버전은 [`env_h100.json`](../quantization/results/kormedmcqa_midm/env_h100.json)에 있습니다.

## 발표 숫자를 만든 코드와의 관계

발표 숫자는 [`scripts/gpu/stage.sh`](../scripts/gpu/stage.sh)가 부르는 [`quantization/v2_run.py`](../quantization/v2_run.py)·[`v2_pipeline.py`](../quantization/v2_pipeline.py)·[`eval_vllm.py`](../quantization/eval_vllm.py)로 만들었습니다. 그 코드는 점수 올리기 실험의 옵션, 끊긴 학습 이어 가기, 중간 점검, 결과 파일 이름 규칙까지 들어 있어 길고 읽기 어렵습니다. 이 폴더는 같은 일을 읽기 쉽게 다시 쓴 것입니다.

- **같은 것:** 데이터 나누기, 프롬프트, 학습 데이터 만들기(정답 토큰에만 손실), 채점은 그 코드의 함수를 그대로 부릅니다([`common.py`](common.py)). 학습률, LoRA rank, 배치, QAT step 같은 설정값이 YAML 설정과 같은지는 CI가 확인합니다([`test_steps.py`](../quantization/tests/test_steps.py)).
- **확인한 것:** 작은 가짜 모델(Mi:dm 토크나이저, 층 2개)로 CPU에서 1단계(transformers+PEFT)와 3단계(QAT)를 2 step씩 끝까지 돌렸습니다. 4·5단계는 프롬프트, 멈춤 토큰, 채점, 요청 형식까지 확인했습니다. 라이브러리 버전은 발표 때와 같습니다.
- **아직 안 한 것:** 이 폴더의 코드로 GPU에서 처음부터 다시 돌려 위 표와 맞춰 본 적은 아직 없습니다. Unsloth 학습, INT4 저장, vLLM 실행은 GPU가 있어야 해서 확인하지 못했습니다.

## 안 될 때

- **Unsloth가 Mi:dm 토크나이저를 못 읽음** (`'AddedToken' object has no attribute 'keys'`): `1_lora_train.py`의 `load_with_unsloth()`가 이미 우회합니다. 그래도 나면 Unsloth 버전이 [`requirements/unsloth.txt`](../requirements/unsloth.txt)와 같은지 봅니다.
- **3단계에서 GPU 메모리 부족:** `--batch-size 2 --grad-accum 8`로 돌립니다. 한 step에 16문항인 것은 같습니다.
- **INT4 폴더를 읽다가 에러:** INT4 모델은 pickle로 저장되어, 저장할 때와 읽을 때 torch·torchao 버전이 같아야 합니다. 같은 가상환경에서 만들고 읽습니다.
- **5단계 서버가 안 뜸:** 다른 vLLM 서버가 같은 GPU나 포트(8000)를 쓰고 있는지 `nvidia-smi`로 봅니다. 포트는 `PORT=8001 bash steps/5_serve.sh`로 바꿉니다.
- 그 밖의 알려진 함정은 [따라하기 8번](../docs/guide/08_troubleshooting.md)에 있습니다.
