# 3. vLLM 서빙 — 배치, 동시 요청, prefix caching

공통 조건은 [0. 실험 설정](00_setup.md)에 있습니다. 비교한 모델은 BF16 LoRA 모델과 INT4 모델입니다([2. INT4 양자화](02_int4_quantization.md)). INT4 PTQ와 INT4 QAT는 저장 형식이 같아 속도도 같습니다. 오프라인 측정에서 둘 다 쟀더니 배치별 처리량 차이가 1% 안쪽이었습니다. 그래서 온라인 측정은 INT4 PTQ로만 했습니다.

## 두 가지를 쟀습니다

| | 오프라인 처리량 | 온라인 서빙 |
|---|---|---|
| 무엇을 보나 | GPU가 한 번에 몇 개를 처리할 때 얼마나 빠른가 | 사용자 여러 명이 동시에 물을 때 응답이 얼마나 걸리는가 |
| 방법 | vLLM `LLM.generate`에 요청 1~256개를 한꺼번에 넣음 | vLLM OpenAI 호환 서버(`vllm.entrypoints.openai.api_server`)에 HTTP 스트리밍 요청을 동시에 1·8·32·64개씩 보냄 |
| 프롬프트 | 2022~2024년 문항 앞 256개 | ① 시험 답: 2021년 문항을 섞어, 동시 요청 수마다 아직 보내지 않은 256개 ② 128토큰 생성: 2021년 문항 앞 256개 |
| 생성 길이 | 요청마다 정확히 128토큰(`ignore_eos`) | ① 시험 답: 최대 16토큰, `<|eot_id|>`에서 멈춤(실제로 1~2토큰) ② 128토큰 생성(`ignore_eos`) |
| 코드 | [`quantization/v2_bench.py`](../../quantization/v2_bench.py) | [`quantization/serving/v2_bench_serve.py`](../../quantization/serving/v2_bench_serve.py), [`serve_vllm.sh`](../../quantization/serving/serve_vllm.sh) |

> **측정 조건**
>
> - 실행: 2026-11-11(오프라인, 128토큰 생성 요청), 2026-11-14(시험 답 요청), H100 NVL 94GB 1장, vLLM 0.23.0, 각 조건 1회
> - 공통 서버 설정: BF16 연산, `gpu_memory_utilization=0.85`, `max_model_len=2048`, CUDA graphs 켬, greedy, 모델 기본 샘플링 설정은 끔(`--generation-config vllm`)
> - INT4: `--quantization torchao`, tile-packed 형식
> - 온라인: 동시 요청 수마다 요청 256개. 클라이언트와 서버가 같은 VM에 있어 네트워크 지연은 빠졌습니다
> - 시험 답 요청의 문항: 2021년 1,059문항을 seed 0으로 섞어 256개씩 네 묶음(1,024문항)으로 나누고, 동시 요청 1·8·32·64에 한 묶음씩 보냈습니다. prefix caching 켬과 끔에 같은 묶음을 같은 순서로 보냈습니다(`FRESH=1 bash scripts/gpu/stage.sh online_A`, `online_B`)
> - prefix caching: vLLM 기본값은 켬입니다. 온라인 측정만 켬과 끔(`--no-enable-prefix-caching`)을 비교했습니다
> - 결과 파일: `bench_*_seed42_h100.json`(오프라인), `online_*_seed42_h100.json`(온라인), [`prompt_shared_prefix.json`](../../quantization/results/kormedmcqa_midm/prompt_shared_prefix.json)(모든 프롬프트가 같은 앞부분의 토큰 수)

## 오프라인 처리량

![한 번에 넣은 요청 수별 초당 생성 토큰, BF16과 INT4](figures/offline_throughput.svg)

| 한 번에 넣은 요청 수 | 1 | 4 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|
| BF16 (초당 토큰) | 116 | 451 | 1,652 | 2,892 | 4,703 | 5,955 | 7,647 |
| INT4 (초당 토큰) | **139** | **551** | 668 | 759 | 834 | 729 | 844 |

요청 하나만 보낼 때(24회 반복의 가운데 값):

| | 첫 토큰까지 | 128토큰 다 받을 때까지 |
|---|---|---|
| BF16 | 14 ms | 1.10초 |
| INT4 | 15 ms | **0.70초** |

**본 것.**
- 요청이 1~4개일 때는 INT4가 20% 안팎 빠릅니다.
- 요청이 16개 이상이면 BF16이 앞서고 256개에서는 9배 차이가 납니다.
- INT4는 초당 800토큰대에서 더 늘지 않습니다.

**그럴 법한 이유.** 요청이 적을 때는 가중치를 메모리에서 읽는 시간이 대부분이라 가중치가 3분의 1인 INT4가 유리합니다. 요청이 많아지면 계산량이 늘어납니다. 이때 BF16은 텐서 코어 행렬곱을 그대로 쓰고 tile-packed INT4는 tinygemm 커널로 가중치를 풀면서 계산합니다. 이 커널은 한 번에 처리하는 행이 적을 때에 맞춰져 있습니다. 다만 커널 단위 프로파일링은 하지 않았습니다.

## 온라인 서빙

### 시험 답 요청 (답 1~2토큰)

이 서비스의 실제 모양입니다. 프롬프트는 약 650토큰이고 답은 숫자 하나라 거의 모든 시간이 프롬프트 처리(prefill)입니다.

![동시 요청 수별 초당 처리 요청, BF16과 INT4, prefix caching 켬과 끔](figures/online_mcqa_requests.svg)

**먼저 읽어 둘 것.** 동시 요청 수마다 아직 보내지 않은 문항 256개를 보냈습니다. 그래서 prefix caching을 켠 서버가 캐시에서 다시 쓴 것은 모든 요청에 같은 앞부분뿐입니다. 이 앞부분은 KT 시스템 문구, 날짜, 과제 지시문으로 530토큰이고, 프롬프트 평균 653토큰의 80% 남짓입니다. 켬과 끔에는 같은 문항을 같은 순서로 보냈으므로 모든 동시 요청 수에서 둘을 같은 조건으로 비교할 수 있습니다.

BF16 (초당 처리 요청 / 첫 토큰까지 가운데 값 · 99번째 백분위):

| 동시 요청 | prefix caching 켬 | prefix caching 끔 |
|---|---|---|
| 1 | 21.2 / 37 ms · 48 ms | 14.1 / 62 ms · 73 ms |
| 8 | 39.3 / 198 ms · 298 ms | 26.3 / 270 ms · 469 ms |
| 32 | 33.2 / 950 ms · 1,092 ms | 26.3 / 1,040 ms · 1,814 ms |
| 64 | 32.8 / 1,818 ms · 2,541 ms | 25.9 / 2,070 ms · 3,578 ms |

INT4:

| 동시 요청 | prefix caching 켬 | prefix caching 끔 |
|---|---|---|
| 1 | 7.6 / 110 ms · 302 ms | 1.8 / 542 ms · 735 ms |
| 8 | 11.0 / 724 ms · 986 ms | 1.9 / 4,151 ms · 4,485 ms |
| 32 | 10.4 / 2,805 ms · 4,328 ms | 1.9 / 14,349 ms · 18,342 ms |
| 64 | 11.1 / 5,542 ms · 7,052 ms | 1.9 / 26,174 ms · 34,840 ms |

**본 것.**
- 모든 요청이 성공했습니다.
- 같은 앞부분만 다시 써도 모든 동시 요청 수에서 효과가 납니다.
  - BF16: 초당 요청이 1.3~1.5배. 동시 요청 1에서 첫 토큰이 62 → 37 ms
  - INT4: 초당 요청이 4~6배. 동시 요청 1에서 첫 토큰이 542 → 110 ms
- 동시 요청이 8개 이상이면 켬이든 끔이든 BF16이 더 많이 처리합니다. 켬끼리 보면 BF16은 초당 33~39개, INT4는 10~11개로 약 3배 차이입니다.
- prefix caching을 끈 INT4는 초당 2개에서 더 늘지 않습니다. 동시 요청이 64개면 첫 토큰까지 26초를 기다립니다.
- **2026-11-11 측정과 다른 점.** 그때는 동시 요청 수마다 같은 256문항을 다시 보냈고, 켬에서 INT4가 BF16보다 많이 처리하는 것처럼 보였습니다. 문항을 새로 보내자 그 역전은 나타나지 않았습니다.

**그럴 법한 이유.** INT4 모델은 프롬프트 처리가 특히 느립니다. tinygemm INT4 커널은 요청 하나에 650토큰을 한꺼번에 처리하는 일에 약합니다(이 문서 「오프라인 처리량」 절과 같은 이유). prefix caching은 이미 처리한 앞부분의 KV 캐시를 다시 써서 그 일을 줄입니다. 그래서 INT4에서 차이가 훨씬 큽니다. 다만 앞부분을 건너뛰어도 문항마다 다른 나머지 약 120토큰은 처리해야 하고, 요청이 몰리면 여기서 INT4 커널의 한계가 다시 드러납니다. 지난 측정에서 INT4가 앞서 보인 것은 프롬프트 전체가 캐시에 있어 이 처리까지 건너뛰었기 때문으로 보입니다. 커널 단위로 나눠 재지는 않았습니다.

### 128토큰 생성 요청 (prefix caching 켬)

대화형 서비스처럼 답이 긴 경우입니다. 2026-11-11 측정이고, 프롬프트는 바로 앞에 보낸 시험 답 요청과 같은 문항이라 캐시에 있습니다. 그래서 생성 속도를 비교하는 표로 읽어 주세요.

| 동시 요청 | BF16: 초당 토큰 / 전체 응답 가운데 값 · 99번째 | INT4: 초당 토큰 / 전체 응답 가운데 값 · 99번째 |
|---|---|---|
| 1 | 116 / 1.10초 · 1.11초 | **180** / **0.71초** · 0.72초 |
| 8 | **854** / 1.18초 · 1.36초 | 723 / 1.40초 · 1.54초 |
| 32 | **2,432** / 1.47초 · 2.19초 | 974 / 4.08초 · 4.66초 |
| 64 | **3,541** / 2.02초 · 3.41초 | 1,025 / 7.64초 · 9.07초 |

오프라인 결과와 같은 흐름입니다. 요청이 하나면 INT4가 빠르고 동시 요청이 8개 이상이면 BF16이 빠릅니다.

## 이 결과로 말할 수 없는 것

- **prefix caching은 모든 요청에 같은 앞부분만 다시 쓴 경우를 쟀습니다.** 같은 질문이 반복해서 들어오는 서비스(예: 자주 묻는 질문)라면 더 많이 다시 쓸 수 있지만, 그 경우는 이번 표에 없습니다.
- **시험 답 요청과 128토큰 생성 요청은 다른 날 쟀습니다.** 128토큰 생성 표는 2026-11-11 측정 그대로이고 문항을 새로 보내지 않았습니다.
- **오프라인 측정도 vLLM 기본값(prefix caching 켬)이고 배치 크기마다 같은 문항이 다시 들어갔습니다.** 128토큰 생성이 대부분이라 영향은 작을 것으로 보지만 따로 재지는 않았습니다.
- **INT4 방식 하나(TorchAO tile-packed, tinygemm 커널)만 쟀습니다.** 다른 INT4 커널(예: Marlin)을 쓰는 방식은 큰 배치에서 결과가 다를 수 있습니다.
- **H100 한 장, vLLM 0.23 한 버전입니다.** 반복 측정은 하지 않았습니다.
- **네트워크 지연이 빠져 있습니다.** 실제 사용자에게는 이 값에 네트워크 왕복 시간이 더해집니다.

## 다시 돌리려면

```bash
CFG=quantization/configs/kormedmcqa_midm.yaml

# 오프라인 처리량
python quantization/v2_bench.py --config $CFG --model-dir quantization/artifacts/kormedmcqa_midm/A_bf16_seed42 \
  --method A_bf16 --precision bf16 --max-model-len 2048 --out bench_A.json

# 온라인 서빙: 서버를 띄우고(INT4면 셋째 인자 torchao), 실제 시험 문항으로 요청
quantization/serving/serve_vllm.sh quantization/artifacts/kormedmcqa_midm/B_int4_ptq_seed42 8000 torchao
python -m quantization.serving.v2_bench_serve --model quantization/artifacts/kormedmcqa_midm/B_int4_ptq_seed42 \
  --concurrency 1,8,32,64 --num-prompts 256 --prompts-config $CFG --prompts-split selection_dev \
  --stop-token-ids 131301,2 --max-tokens 16 --fresh-per-level --shuffle-seed 0 --output online_B.json

# prefix caching 끄고 비교하려면 서버를 이렇게 띄웁니다
EXTRA_ARGS=--no-enable-prefix-caching quantization/serving/serve_vllm.sh <모델 폴더> 8000 [torchao]
```

`--fresh-per-level`은 동시 요청 수마다 아직 보내지 않은 문항을 보냅니다. 켬과 끔에 같은 `--shuffle-seed`를 주면 같은 문항이 같은 순서로 갑니다.

GPU VM에서는 `FRESH=1 bash scripts/gpu/stage.sh online_A`와 `online_B`가 위 과정을 켬·끔 두 번씩 돌립니다. `FRESH=1`을 빼면 2026-11-11 방식(같은 256문항을 다시 보냄, 128토큰 생성 요청 포함)으로 돕니다.
