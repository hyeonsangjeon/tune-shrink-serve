# 작은 모델을 더 작고 빠르게

**한국어** | [English](README_en.md)

**Unsloth LoRA · TorchAO INT4 · vLLM 서빙을 모델 하나로 끝까지 재 본 기록** — PyTorch Day Korea 2026 Serve Track 발표 레포

작은 모델을 튜닝해서 서비스에 쓰려면 세 가지가 궁금합니다. 튜닝하면 얼마나 좋아지는지, 줄이면 얼마나 잃는지, 서빙하면 얼마나 빠르고 싼지입니다. 이 레포는 세 가지를 같은 모델, 같은 문항, 같은 GPU로 잽니다. 마지막으로 같은 문항을 매니지드 API 모델 11개에도 풀게 해, 자체 GPU의 작은 모델(SLM)과 나란히 놓습니다.

KT의 공개 모델 **Mi:dm 2.0 Base(11.5B)** 에 의사·간호사·약사·치과의사 **국가시험 기출문제(KorMedMCQA)** 를 학습시켰습니다. 그다음 INT4로 줄이고 vLLM으로 서빙했습니다. 아래 숫자는 모두 **2022~2024년 시험 3,009문항**에서 나온 값입니다. 자체 GPU 숫자는 **H100 NVL 한 장, seed 하나**에서 나왔고, API 숫자는 같은 문항을 Azure AI Foundry에서 모델마다 한 번 잰 값입니다.

- **LoRA 튜닝:** 정답률 63.3% → **72.3%**. 같은 학습을 Unsloth로 하면 정답률은 같고 학습 시간은 35% 줄었습니다.
- **INT4 양자화(TorchAO):** 크기 21.51 GiB → **7.20 GiB**. 정답률은 2.6~2.9%p 내려갔습니다. QAT와 PTQ의 차이는 발표 실행에서 +0.2%p(p=0.67), 같은 코드로 다시 학습한 실행에서 +1.0%p(p=0.02)여서, QAT가 PTQ보다 나은지는 정하지 못했습니다. 그 차이가 가짜 양자화 덕인지 600 step 추가 학습 덕인지는 [대조 실험](docs/qat_control.md)으로 따로 쟀습니다.
- **vLLM 서빙:** 요청이 1~4개면 INT4가 20% 빠릅니다. 요청이 많으면 BF16이 최대 9배 빠릅니다. prefix caching을 끄면 INT4는 초당 2문항에서 막힙니다.
- **SLM 대 API:** 같은 3,009문항에서 매니지드 API 모델 11개 중 10개가 튜닝한 Mi:dm(72.3%)보다 높았습니다(최고 98.4%). 3,009문항을 한꺼번에 풀면 1,000문항당 비용은 자체 H100의 Mi:dm이 $0.0111로 가장 쌉니다(정가로 계산). 한 요청씩 받는 서버(동시 1, prefix caching 끔)에서는 $0.137이 되어, 정답률 95.9%인 gpt-6-luna($0.0836)보다 비쌉니다.
- **덧붙임, 점수 올리기:** 다른 모델의 출력 없이 학습 방법(epoch, 보기 순서, 학습률, 직군 비중, Mi:dm이 직접 쓴 해설)을 바꿔 2021년 문제로 비교했습니다. seed만 바꿨을 때의 차이(0.9~1.5%p)를 넘은 방법이 없어 2022~2024년 문제는 다시 재지 않았습니다.

![2022~2024년 3,009문항 정답률: 튜닝 전, BF16 LoRA(transformers+PEFT, Unsloth), INT4 PTQ, INT4 QAT](docs/talk/figures/accuracy_test.svg)

## 발표 순서대로 본 결론

| 아젠다 | 결론 | 이 결과로 말할 수 없는 것 | 자세히 |
|---|---|---|---|
| ① Unsloth LoRA | 같은 LoRA 학습이 24.3분 → 15.7분. 학습 중 GPU 메모리(`nvidia-smi`)는 82.3 → 25.5 GiB. 정답률 차이는 없습니다(p=0.89) | Mi:dm 토크나이저를 Unsloth가 읽지 못해 우회가 필요했습니다. 메모리는 한 시점에 읽은 값입니다 | [01](docs/talk/01_finetune_unsloth.md) |
| ② TorchAO INT4 | 크기 3분의 1, 정답률 −2.9%p(PTQ)·−2.6%p(QAT). 두 차이 모두 같은 문항 비교에서 유의합니다. QAT는 PTQ보다 +0.2%p(p=0.67), 같은 코드로 다시 학습했을 때는 +1.0%p(p=0.02) | QAT 설정 하나, INT4 방식 하나만 시험했습니다. QAT가 PTQ보다 나은지는 두 실행의 결과가 갈려 정하지 못했습니다 | [02](docs/talk/02_int4_quantization.md) |
| ③ vLLM 서빙 | 모든 요청에 같은 앞부분만 다시 쓰는 prefix caching으로 INT4의 첫 토큰이 542 → 110 ms로 줄었습니다. 그래도 요청이 몰리면 BF16이 앞섭니다. 온라인에서는 켬끼리 약 3배, 오프라인에서는 tile-packed INT4가 초당 800토큰대에서 멈추고 BF16은 7,600토큰까지 갑니다 | prefix caching은 같은 앞부분만 다시 쓴 경우를 쟀고, 같은 질문이 반복되는 경우는 재지 않았습니다 | [03](docs/talk/03_vllm_serving.md) |
| ④ 비용 (SLM 대 API) | 3,009문항을 한꺼번에 풀면 1,000문항당 BF16 $0.011, INT4 $0.172입니다(H100 온디맨드 정가로 계산). 같은 문항에서 매니지드 API 모델 11개 중 10개가 Mi:dm보다 정확했고, 한 요청씩 받는 서버 비용으로는 더 싼 API 모델도 있습니다 | 비용은 정가로 계산한 값이고 기준이 다릅니다(GPU 시간 대 토큰). 매니지드 모델은 크기가 공개되지 않았거나 Mi:dm보다 훨씬 크고, 학습 데이터는 알 수 없습니다 | [04](docs/talk/04_cost.md) |
| 덧붙임: 점수 올리기 | 다른 모델의 출력 없이 학습 방법을 바꾼 실험 중 2021년 문제에서 통과 기준(+2.0%p, p<0.05)을 넘은 것이 없습니다. 가장 높은 값(72.0%)은 레시피는 그대로 두고 seed만 바꾼 학습에서 나왔습니다 | 실험마다 한 번씩 돌렸고, 2021년 점수를 31번 보고 골랐습니다. 다른 모델의 출력을 쓰는 방법은 재지 않았습니다 | [05](docs/talk/05_recipe_sweep.md) |

공통 조건(모델, 데이터, 프롬프트, 채점, 장비)은 [00_setup.md](docs/talk/00_setup.md)에 있습니다. 숫자는 모두 [`quantization/results/kormedmcqa_midm/`](quantization/results/kormedmcqa_midm/)의 결과 JSON에서 나옵니다. 문서의 숫자와 JSON이 어긋나면 CI가 실패합니다([근거 색인](docs/EVIDENCE.md)).

## 어디로 갈까요

| 하고 싶은 일 | 가는 곳 |
|---|---|
| 결과와 근거를 자세히 본다 | [`docs/talk/`](docs/talk/) 00~05 |
| 발표 슬라이드를 본다 | [PDF, 한국어](docs/talk/slides/pytorchday2026_slides_ko.pdf) (2026-11-21 발표 그대로) |
| GPU 없이 이 레포가 제대로 도는지 본다 | 바로 아래 「1분 확인」 |
| 학습·INT4·서빙 코드를 읽고 한 단계씩 돌린다 | [`steps/`](steps/) 1~5 (단계마다 파일 하나, 함수마다 설명) |
| GPU에서 처음부터 다시 돌린다 | [따라하기](docs/guide/) 1~8번 (단계마다 명령, 걸린 시간, 확인 기준, 안 될 때) |
| 어느 폴더에 무엇이 있는지 본다 | 아래 「레포 구성」 |
| 학습된 모델만 쓴다 | [`HyeonSang/Midm-2.0-KorMedMCQA`](https://huggingface.co/HyeonSang/Midm-2.0-KorMedMCQA), [`…-QAT-bf16`](https://huggingface.co/HyeonSang/Midm-2.0-KorMedMCQA-QAT-bf16) |

## 1분 확인 (CPU만, 계정 없이)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/cpu.txt
python -m pytest -q
python scripts/build_evidence_index.py --check
python scripts/make_figures.py --check
python -m quantization.managed_compare --check
```

**이 1분이 확인하는 것:**
- 데이터 분리, 프롬프트, 채점, 서빙 벤치 코드가 테스트를 통과하는지
- 문서의 숫자가 결과 JSON과 같은지
- 차트가 결과 JSON에서 똑같이 다시 그려지는지
- 매니지드 API 비교표가 결과 JSON에서 똑같이 다시 만들어지는지

**확인하지 않는 것:** 학습, 양자화, 서빙을 GPU에서 다시 하지는 않습니다. 매니지드 API도 호출하지 않습니다. 모델과 데이터도 내려받지 않습니다. 학습·양자화·서빙 코드는 [`steps/`](steps/)에 있고, GPU에서 돌리는 방법은 바로 아래에 있습니다.

## GPU로 다시 돌리기

H100(또는 80GB급 GPU) 한 장이면 됩니다. 방법은 두 가지입니다.

**코드를 읽으며 한 단계씩:** [`steps/`](steps/)는 학습, INT4, 채점, 서빙을 단계마다 Python 파일 하나로 정리한 것입니다. 파일 맨 위에 설정값이 있고, 함수마다 무엇을 왜 하는지 주석이 있습니다.

```bash
source ~/venv-q/bin/activate                  # 아래 bootstrap.sh가 만든 가상환경
python steps/1_lora_train.py                  # LoRA 학습 (transformers+PEFT, --engine unsloth도 됨)
python steps/2_int4_ptq.py                    # INT4 PTQ (TorchAO)
python steps/3_int4_qat.py                    # INT4 QAT (TorchAO)
python steps/4_evaluate.py outputs/2_int4_ptq # 2022~2024년 3,009문항 정답률 (vLLM)
bash steps/5_serve.sh outputs/2_int4_ptq      # vLLM 서버. 다른 터미널에서 python steps/5_ask.py
```

2026-11-16~17에 H100 NVL에서 이 코드로 1단계부터 다시 학습하고 채점했습니다. 정답률은 발표 때와 0.4%p 안에서 같았고(BF16 LoRA 72.3%, Unsloth 72.5%, INT4 PTQ 69.1%, INT4 QAT 70.1%), 학습 시간 차이는 1분 이내였습니다. 1·3단계 학습을 도중에 강제로 끄고 `--resume`으로 이어 가는 것도 확인했습니다. 자세한 것은 [`steps/README.md`](steps/README.md)에 있습니다.

**발표 숫자를 만든 순서 그대로:** 아래 `stage.sh` 단계들이 중간 점검, 끊긴 학습 이어 가기, 서빙 속도 측정, 결과 JSON 저장까지 합니다. 단계마다 명령, 확인 기준, 안 될 때 할 일은 [따라하기](docs/guide/)에 있습니다. 큰 흐름과 이번 실행에 걸린 시간은 이렇습니다.

```bash
bash scripts/gpu/bootstrap.sh                 # 가상환경 2개 설치, 드라이버·GPU·테스트 확인 (약 6분)
bash scripts/gpu/stage.sh selftest            # QAT의 가짜 양자화가 서빙 INT4와 같은지 확인
bash scripts/gpu/stage.sh base_dev            # 튜닝 전 모델, 2021년 문제
BACKEND=hf bash scripts/gpu/stage.sh a        # LoRA 학습, transformers+PEFT (24분)
CONFIG=quantization/configs/kormedmcqa_midm_unsloth.yaml bash scripts/gpu/stage.sh a   # 같은 학습, Unsloth (16분)
bash scripts/gpu/stage.sh a_dev               # 2021년 문제로 중간 점검 (튜닝 전보다 2%p 이상 올라야 계속)
bash scripts/gpu/stage.sh b                   # INT4 PTQ (17초)
bash scripts/gpu/stage.sh c                   # INT4 QAT (1시간 42분)
bash scripts/gpu/stage.sh final_A             # 2022~2024년 문제, 모델마다 한 번 (final_base, final_B, final_C 도)
bash scripts/gpu/stage.sh bench_A             # 오프라인 처리량 (bench_B, bench_C 도)
FRESH=1 bash scripts/gpu/stage.sh online_A    # 온라인 서빙, prefix caching 켬·끔, 동시 요청마다 새 문항 (online_B 도)
```

모든 단계는 `--detach`를 붙이면 백그라운드로 돕니다. 진행 상황은 `bash scripts/gpu/status.sh`로 봅니다.

## 레포 구성

```
README.md                  여기. 결과 요약과 시작점
steps/                     단계별 코드: LoRA 학습 → INT4 PTQ → INT4 QAT → 정답률 → vLLM 서빙. 파일 하나에 단계 하나
docs/
  talk/                    결과 문서 00~05 (발표 순서), 차트(figures/), 발표 슬라이드 PDF(slides/)
  guide/                   따라하기 1~8: GPU 없이 확인 → GPU VM → 학습 → INT4 → 서빙 → API 비교 → 점수 올리기 → 안 될 때
  EVIDENCE.md              근거 색인: 문서의 숫자마다 어느 결과 JSON의 어느 값인지 (evidence_ledger.yaml에서 생성)
  DATA_AND_LICENSES.md     모델·데이터·결과의 라이선스와 무엇을 레포에 넣지 않았는지
quantization/              코드 (폴더 안내는 quantization/README.md)
  configs/                 설정. kormedmcqa_midm.yaml 한 파일이 이번 실험 전체, sweep/는 점수 올리기 실험
  results/kormedmcqa_midm/ 결과 JSON (문서의 모든 숫자가 여기서 나옴)
  serving/                 vLLM 서버 실행 스크립트와 온라인 벤치
  tests/                   CPU 테스트
requirements/              설치 파일: cpu · gpu · unsloth · slides (requirements/README.md)
scripts/                   GPU VM 스크립트(gpu/), 차트·슬라이드·근거 색인 만들기 (scripts/README.md)
archive/qwen3_korquad/     이전 실험(Qwen3-8B · KorQuAD) 결과, 노트북, 설정. 본편과 비교하지 않음
```

## 이전 실험 (Qwen3-8B · KorQuAD)

이 레포는 처음에 Qwen3-8B와 KorQuAD로 같은 비교(BF16 LoRA, INT4 PTQ, INT4 QAT)를 했습니다(A100, seed 3개). 그 결과는 [`archive/qwen3_korquad/README.md`](archive/qwen3_korquad/README.md)에 그대로 있습니다. 두 실험은 모델과 데이터가 함께 달라 서로 인과 비교를 하지 않습니다.

## 데이터와 라이선스

- **코드와 결과 JSON:** [MIT](LICENSE)입니다.
- **모델과 데이터:** 레포에 넣지 않고 실행할 때 내려받습니다. 문서에는 형식을 보여 주려고 학습용 문항 하나만 출처와 함께 실었습니다([예시](docs/talk/00_setup.md#한-문항은-이렇게-생겼습니다)). Mi:dm 2.0 Base는 MIT, KorMedMCQA는 CC BY-NC 2.0, KorQuAD 1.0은 CC BY-ND 2.0 KR입니다.
- **KorMedMCQA로 학습한 가중치:** Hugging Face에 공개했습니다. 데이터 라이선스에 따라 비상업 용도로만 쓸 수 있습니다. 의료 조언에 쓰는 모델이 아닙니다.
- **KorQuAD로 학습한 가중치:** 공개하지 않습니다.

자세한 내용은 [`docs/DATA_AND_LICENSES.md`](docs/DATA_AND_LICENSES.md)에 있습니다.

이 레포의 코드는 [`hyeonsangjeon/PDF2LLM-Tuning-Studio`](https://github.com/hyeonsangjeon/PDF2LLM-Tuning-Studio)의 `pdf_qa_extraction/quantization/`(커밋 `760a92c`)에서 시작했습니다.
