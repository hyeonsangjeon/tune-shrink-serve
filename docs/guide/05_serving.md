# 5. 서빙 속도 재기

**한국어** | [English](05_serving_en.md)

BF16 LoRA 모델과 INT4 모델을 vLLM으로 띄워 두 가지를 잽니다.
- **오프라인 처리량:** 요청 1~256개를 한꺼번에 넣었을 때 초당 생성 토큰
- **온라인 서빙:** OpenAI 호환 서버에 동시 요청 1·8·32·64개를 보냈을 때 초당 처리 요청과 첫 토큰까지 시간. prefix caching 켬·끔을 비교합니다

| 필요한 것 | 걸린 시간 (H100 NVL) | 결과 문서 |
|---|---|---|
| [4번](04_int4.md)의 BF16 LoRA 모델과 INT4 모델 | 온라인 두 모델 약 30분(서버 띄우기 포함) | [3. vLLM 서빙](../talk/03_vllm_serving.md) |

서버를 띄우고 질문 하나를 보내는 가장 짧은 코드는 [`steps/5_serve.sh`](../../steps/5_serve.sh)와 [`steps/5_ask.py`](../../steps/5_ask.py)에 있습니다. 아래 순서는 속도를 재는 방법입니다.

## 순서

```bash
bash scripts/gpu/stage.sh bench_A          # 오프라인, BF16 LoRA
bash scripts/gpu/stage.sh bench_B          # 오프라인, INT4 PTQ
bash scripts/gpu/stage.sh bench_C          # 오프라인, INT4 QAT (PTQ와 같은 형식이라 속도가 같은지 확인)

FRESH=1 bash scripts/gpu/stage.sh online_A # 온라인, BF16 LoRA, prefix caching 켬 → 끔
FRESH=1 bash scripts/gpu/stage.sh online_B # 온라인, INT4 PTQ, prefix caching 켬 → 끔
```

- **`FRESH=1`:** 동시 요청 수마다 아직 보내지 않은 2021년 문항 256개를 보내고, 켬과 끔에 같은 문항을 같은 순서로 보냅니다. 발표 숫자는 이 방식입니다. 빼면 같은 256문항을 단계마다 다시 보내고(캐시에 프롬프트 전체가 남습니다), 128토큰 생성 요청도 함께 잽니다.
- 서버는 [`quantization/serving/serve_vllm.sh`](../../quantization/serving/serve_vllm.sh)가 띄우고 끕니다. 클라이언트와 서버가 같은 VM에 있어 네트워크 지연은 빠집니다.

## 확인

| 단계 | 파일 (`quantization/results/kormedmcqa_midm/`) | 볼 것 |
|---|---|---|
| `bench_*` | `bench_<모델>_seed42_h100.json` | 배치마다 `throughput_tok_s` |
| `online_*` | `online_<모델>_mcqa_cache{on,off}_seed42_h100.json` | 단계마다 `failed` 0, `request_throughput_rps`, `ttft_s`. `FRESH=1`이면 `config.fresh_per_level: true`이고 단계마다 `prompt_slice`가 겹치지 않음 |

온라인 결과를 바꿨다면 서버 비용도 바뀝니다. [4번 결과 문서](../talk/04_cost.md)의 「서버로 받을 때」 표와 `python -m quantization.managed_compare`를 다시 돌립니다([8번 문서](08_troubleshooting.md#결과를-바꾼-뒤)).

## 안 될 때

- **INT4 서버가 늦게 뜸:** INT4 모델은 읽는 데 BF16보다 오래 걸립니다(이번 실행에서 채점할 때 모델 읽기 106초 대 22초). `HEALTH_TRIES`가 900(약 15분)으로 잡혀 있습니다.
- **prefix caching을 끈 INT4에서 첫 토큰이 수십 초:** 정상입니다. tile-packed INT4는 프롬프트 처리가 느려서, 동시 요청이 쌓이면 줄이 깁니다([3번 결과 문서](../talk/03_vllm_serving.md)).
- **다음 서버가 안 뜸(GPU 메모리 부족):** 앞 서버가 아직 GPU를 놓지 않은 경우입니다. `nvidia-smi`로 확인하고 남은 프로세스를 끕니다. 스테이지는 서버를 끈 뒤 GPU 메모리가 2GB 아래로 내려갈 때까지 기다립니다.

다음: [6. API 모델과 비교](06_managed_api.md)
