# 따라하기

이 레포의 결과를 직접 확인하고 다시 만드는 순서입니다. 위에서부터 차례로 하면 됩니다. 1번만 GPU 없이 되고, 2~5번과 7번은 GPU가, 6번은 Azure AI Foundry 계정이 필요합니다.

| 단계 | 하는 일 | 필요한 것 | 걸린 시간 (이번 실행) | 나오는 결과 | 결과 문서 |
|---|---|---|---|---|---|
| [1. GPU 없이 확인](01_cpu_check.md) | 테스트, 문서 숫자와 결과 JSON 대조, 차트 다시 그리기 | Python 3.10 이상 | 약 1분 | 화면의 `OK` | — |
| [2. GPU VM 준비](02_gpu_setup.md) | VM 만들기, 접속, 가상환경 두 개 설치 | GPU 80GB 이상 1장 | 설치 약 6분 | `env_<gpu>.json` | [0. 실험 설정](../talk/00_setup.md) |
| [3. LoRA 학습](03_finetune.md) | 튜닝 전 점수, LoRA 학습(transformers+PEFT, Unsloth), 중간 점검 | 2번 | 24.3분 / 15.7분 | `A_train_seed42.json` | [1. 파인튜닝](../talk/01_finetune_unsloth.md) |
| [4. INT4로 줄이고 채점](04_int4.md) | INT4 PTQ, INT4 QAT, 2022~2024년 문제로 한 번 채점, 같은 문항 비교 | 3번 | QAT 학습 102분, 채점 모델마다 1~6분 | `eval_vllm_*_h100.json`, `compare_final_*.json` | [2. INT4 양자화](../talk/02_int4_quantization.md) |
| [5. 서빙 속도 재기](05_serving.md) | 오프라인 처리량, 온라인 서버(prefix caching 켬·끔) | 4번 | 온라인 두 모델 약 30분 | `bench_*.json`, `online_*.json` | [3. vLLM 서빙](../talk/03_vllm_serving.md) |
| [6. API 모델과 비교](06_managed_api.md) | 같은 문항을 매니지드 API 모델에 풀게 하기 | Azure AI Foundry, `az login` | 모델마다 2.6~28.7분 | `managed/*.json` | [4. 비용](../talk/04_cost.md) |
| [7. (선택) 점수 올리기 실험](07_recipe_sweep.md) | 학습 방법을 바꿔 2021년 문제로 비교 | 3번 | 약 4~5시간 | `sweep/summary_dev.json` | [5. 점수 올리기](../talk/05_recipe_sweep.md) |
| [8. 안 될 때와 커밋할 것](08_troubleshooting.md) | 알려진 함정, 결과를 바꾼 뒤 할 일 | — | — | — | — |

- **비용:** GPU 단계는 걸린 시간 × GPU 시간당 가격입니다. 이번 실행은 Azure H100 NVL 1장, 온디맨드 정가 시간당 $6.98이었습니다. 모델 다운로드(약 23GB)와 설치 시간은 위 표에 넣지 않았습니다.
- **설정은 한 파일입니다:** [`quantization/configs/kormedmcqa_midm.yaml`](../../quantization/configs/kormedmcqa_midm.yaml). 모델, 데이터 나누기, 프롬프트, 학습값, 결과 폴더가 모두 여기 있습니다.
- **명령은 모두 레포 루트에서 실행합니다.**
- **GPU 단계는 [`scripts/gpu/stage.sh`](../../scripts/gpu/stage.sh)로 돌립니다.** `bash scripts/gpu/stage.sh <단계>`가 한 단계를 돌리고 로그와 끝난 상태를 남깁니다. `stage.sh` 없이 Python 명령으로 직접 돌리는 방법은 각 결과 문서의 「다시 돌리려면」 절에 있습니다.
- **이름 읽는 법:** 단계와 파일 이름에 남은 `a`·`b`·`c`와 `A_bf16`·`B_int4_ptq`·`C_int4_qat`은 차례로 BF16 LoRA 모델, INT4 PTQ, INT4 QAT입니다. `base`는 튜닝 전 모델입니다. 파일 이름의 `selection_dev`는 2021년 문제(고르기용), `final_holdout`과 `_h100`이 붙은 채점 파일은 2022~2024년 문제(마지막에 한 번)입니다.
