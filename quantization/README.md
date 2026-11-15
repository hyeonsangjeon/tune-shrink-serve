# quantization/ — 코드

학습, INT4 변환, 채점, 서빙 측정, API 비교 코드가 모두 이 패키지에 있습니다. 명령은 레포 루트에서 `python -m quantization.<모듈>`로 돌립니다. 순서대로 돌리는 방법은 [따라하기](../docs/guide/)에 있습니다.

## 폴더

| 폴더 | 무엇 |
|---|---|
| [`configs/`](configs/) | 설정. [`kormedmcqa_midm.yaml`](configs/kormedmcqa_midm.yaml) 한 파일이 이번 실험 전체(모델, 데이터 나누기, 프롬프트, 학습값, INT4 방식, 결과 폴더)입니다. `kormedmcqa_midm_unsloth.yaml`은 Unsloth 학습용, [`sweep/`](configs/sweep/)은 점수 올리기 실험 |
| [`results/kormedmcqa_midm/`](results/kormedmcqa_midm/) | 결과 JSON. 문서의 숫자는 모두 여기서 나옵니다 |
| [`artifacts/`](artifacts/) | 학습한 가중치가 생기는 곳. 커밋하지 않습니다 |
| [`serving/`](serving/) | vLLM 서버 실행 스크립트, 온라인 벤치, 질문 하나 보내는 클라이언트 |
| [`tests/`](tests/) | CPU 테스트 (`python -m pytest -q`) |

## 모듈

| 모듈 | 하는 일 | 따라하기 |
|---|---|---|
| `kormedmcqa.py` | KorMedMCQA 불러오기, 연도로 나누기(학습 2012~2020, 2021년, 2022~2024년), 프롬프트, 답 번호 읽기, 같은 문항 비교(McNemar)와 신뢰구간 | 3 |
| `v2_run.py` | GPU 단계 실행: `selftest`, `a`(LoRA 학습), `b`(INT4 PTQ), `c`(INT4 QAT), `merge` | 3, 4 |
| `v2_pipeline.py` | 학습과 변환의 실제 구현: 채팅 템플릿, 답 토큰에만 손실, transformers+PEFT와 Unsloth, TorchAO INT4 | 3, 4 |
| `eval_vllm.py` | vLLM으로 채점(번호만 답하기, 보기 확률, 해설 쓰고 답하기), 어댑터 채점 | 3, 4, 7 |
| `v2_bench.py` | 오프라인 처리량과 요청 하나의 지연 | 5 |
| `serving/v2_bench_serve.py` | 온라인 서버 벤치(동시 요청, prefix caching 켬·끔, `--fresh-per-level`) | 5 |
| `eval_managed.py` | 같은 문항을 매니지드 API 모델에 풀게 하기(Entra ID, 키 없음) | 6 |
| `managed_compare.py` | API 결과와 Mi:dm 결과를 같은 문항끼리 비교, 요약표 | 6 |
| `self_explain.py` | 튜닝 전 Mi:dm이 학습 문항의 해설을 직접 쓰게 하고 거르기 | 7 |
| `sweep_report.py` | 점수 올리기 실험을 2021년 문제로 정리: 체크포인트 고르기, 통과 기준 | 7 |
| `publish_hf.py` | 학습한 가중치를 모델 카드와 함께 Hugging Face에 올리기 | 4 |
| `data_korquad.py` | 설정 읽기(`load_config`, 모든 단계가 씀)와 이전 실험의 KorQuAD 불러오기 | — |

**이름 읽는 법.** `v2_`는 이 코드의 두 번째 판이라는 뜻으로 남은 이름입니다. 단계와 파일 이름의 `a`·`b`·`c`, `A_bf16`·`B_int4_ptq`·`C_int4_qat`은 차례로 BF16 LoRA 모델, INT4 PTQ, INT4 QAT입니다.

## 이전 실험(Qwen3-8B · KorQuAD)에만 쓰는 모듈

`train_lora.py`, `eval_qa.py`, `bench_baseselect.py`, `splits.py`, `v2_report.py`. 설정과 결과는 [`archive/qwen3_korquad/`](../archive/qwen3_korquad/)에 있고, 코드의 기본 설정이 그 폴더의 `config.yaml`입니다. 이번 실험에서는 `--config quantization/configs/kormedmcqa_midm.yaml`을 항상 붙입니다.
