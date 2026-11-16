#!/usr/bin/env bash
# 5단계(서버): 모델 하나를 vLLM의 OpenAI 호환 서버로 띄웁니다. Ctrl-C로 끕니다.
#
#   bash steps/5_serve.sh                            # outputs/1_lora (BF16)
#   bash steps/5_serve.sh outputs/2_int4_ptq         # INT4 (폴더를 보고 --quantization torchao를 붙입니다)
#   NO_PREFIX_CACHE=1 bash steps/5_serve.sh ...      # prefix caching 끄기 (발표 ③의 "끔")
#
# 서버가 "Application startup complete"를 찍으면, 다른 터미널에서 질문을 보냅니다:
#   python steps/5_ask.py
#
# 발표 ③의 서빙 숫자는 같은 서버 설정으로 quantization/serving/v2_bench_serve.py 가 잰 값입니다.
set -euo pipefail
cd "$(dirname "$0")/.."                         # 레포 최상위 폴더에서 실행

MODEL_DIR="${1:-outputs/1_lora}"
PORT="${PORT:-8000}"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "[5단계] 모델 폴더가 없습니다: $MODEL_DIR  (1~3단계를 먼저 돌리세요)" >&2
  exit 2
fi

ARGS=(
  --model "$MODEL_DIR"
  --served-model-name midm           # 요청의 "model" 자리에 쓸 이름
  --host 127.0.0.1 --port "$PORT"    # 이 VM 안에서만 받습니다
  --dtype bfloat16
  --max-model-len 4096               # 프롬프트+답의 최대 토큰 수. 이 과제는 1,000토큰 안쪽
  --gpu-memory-utilization 0.85      # 가중치를 올리고 남는 GPU 메모리를 KV 캐시로 씁니다
  --generation-config vllm           # 모델 폴더의 샘플링 기본값(temperature 0.8 등)을 쓰지 않습니다
)

# INT4로 저장한 폴더(config.json에 "quant_method": "torchao")면 TorchAO 형식으로 읽습니다.
if grep -q '"quant_method": *"torchao"' "$MODEL_DIR/config.json"; then
  ARGS+=(--quantization torchao)
fi

# prefix caching: 요청들이 같은 앞부분(Mi:dm 시스템 블록 + 시스템 프롬프트)으로 시작하면
# 그 부분의 계산 결과(KV 캐시)를 다시 씁니다. vLLM 기본값은 켬입니다.
if [[ "${NO_PREFIX_CACHE:-0}" == 1 ]]; then
  ARGS+=(--no-enable-prefix-caching)
fi

# FlashInfer 샘플러는 처음 쓸 때 nvcc로 컴파일해야 해서, CUDA 컴파일러가 없는 환경에서는 끕니다.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

echo "+ python -m vllm.entrypoints.openai.api_server ${ARGS[*]}"
exec python -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
