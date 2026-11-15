# 6. API 모델과 비교 (GPU 필요 없음)

같은 2022~2024년 3,009문항을 매니지드 API 모델에 풀게 하고, 자체 GPU의 Mi:dm과 정답률·비용을 나란히 놓습니다. OpenAI 호환 Chat Completions 엔드포인트면 되고, 이번에는 Azure AI Foundry의 `/openai/v1`을 썼습니다.

| 필요한 것 | 걸린 시간 (이번 실행) | 비용 | 결과 문서 |
|---|---|---|---|
| Azure AI Foundry 리소스와 모델 배포, `az` CLI, Python 3.12 | 모델마다 2.6~28.7분 | 쓴 토큰 × 모델 정가. 1,000문항당 값은 결과 문서의 표 | [4. 비용](../talk/04_cost.md#-매니지드-api와-비교--slm-대-api) |

## 순서

1. **로그인:** `az login`으로 Foundry 리소스에 접근할 수 있는 계정에 들어갑니다. 인증은 Entra ID 토큰이고 API 키는 쓰지 않습니다.
2. **엔드포인트는 환경 변수로만 넘깁니다.** 레포와 결과 파일에는 남지 않습니다.
   ```bash
   export MANAGED_BASE_URL=https://<리소스 이름>.services.ai.azure.com/openai/v1
   export MANAGED_AZ_SUBSCRIPTION=<구독 이름 또는 ID>      # 토큰을 받을 테넌트를 고릅니다(선택)
   ```
3. **파일럿:** 2021년 문제 30문항으로 토큰 수, 답 형식, 오류를 봅니다. 2022~2024년 결과 파일은 건드리지 않습니다.
   ```bash
   python -m quantization.eval_managed --config quantization/configs/kormedmcqa_midm.yaml \
       --deployment <배포 이름> --split selection_dev --eval-size 30
   ```
4. **가격:** Azure Retail Prices API에서 그 모델의 입력·출력·캐시 입력 가격(USD/100만 토큰)을 읽습니다.
5. **본 측정:** 배포마다 한 번 잽니다. 결과 파일이 이미 있으면 멈춥니다(`--force`는 그 실행이 잘못됐을 때만).
   ```bash
   python -m quantization.eval_managed --config quantization/configs/kormedmcqa_midm.yaml \
       --deployment <배포 이름> --concurrency 32 --region eastus2 \
       --price-in <입력> --price-out <출력> --price-cached-in <캐시 입력> --price-source "Azure Retail Prices API, read <날짜>"
   ```
6. **비교표:** 짝지은 비교와 요약표(`managed/compare_final_managed.json`)를 만듭니다.
   ```bash
   python -m quantization.managed_compare
   ```

## 확인

| 파일 (`quantization/results/kormedmcqa_midm/managed/`) | 볼 것 |
|---|---|
| `eval_managed_<배포>.json` | `n_eval` 3,009. `n_errors`는 콘텐츠 필터에 막힌 문항 수. `finish_reasons`의 `length`는 출력 한도(2,048토큰)에 걸린 문항 수 |
| `compare_final_managed.json` | 모델마다 정답률, 1,000문항당 비용, Mi:dm과 같은 문항 비교 |

결과 파일에는 문항 id, 정답, 예측 번호, 답 앞 40자, 사용 토큰, 지연 시간, 끝난 이유, 짧은 오류 이유가 들어 있습니다. 문항 원문과 엔드포인트 주소는 없습니다.

## 안 될 때

- **HTTP 429:** 배포의 요청 한도에 걸렸습니다. 스크립트가 `Retry-After`를 따라 다시 보냅니다. 분당 250회인 배포는 `--concurrency 12` 정도로 둡니다.
- **HTTP 400 (콘텐츠 필터):** 의료 문항 일부가 Azure 콘텐츠 필터에 막힙니다(이번 실행은 모델마다 7~10문항). 오답으로 세고 `n_errors`에 남깁니다.
- **`datasets`가 pickle 오류로 깨짐:** Python 3.14입니다. Python 3.12 가상환경을 씁니다.
- **zsh에서 옵션을 변수에 담아 넘기면 argparse가 멈춤:** zsh는 따옴표 없는 변수를 단어로 나누지 않습니다. 명령을 풀어 쓰거나 bash로 돌립니다.

다음: [7. (선택) 점수 올리기 실험](07_recipe_sweep.md)
