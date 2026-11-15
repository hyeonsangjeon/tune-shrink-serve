# 8. 안 될 때와 커밋할 것

## 알려진 함정

- **Mi:dm 채팅 템플릿은 KT 시스템 문구(약 450토큰)와 오늘 날짜를 항상 앞에 붙입니다.**
  - 날짜가 바뀌면 프롬프트가 바뀌므로 설정의 `template_kwargs.date_string`이 날짜를 고정합니다.
  - 온라인 서빙에서도 같은 값을 요청에 넣습니다(`chat_template_kwargs`).
- **턴 종료와 정지 토큰:** 턴 종료는 `<|eot_id|>`(131301), eos는 `<|end_of_text|>`(2)입니다. 학습 정답은 `[번호, 131301]`이고, 생성은 `[131301, 2]`에서 멈춥니다.
- **모델의 기본 생성 설정은 샘플링입니다**(temperature 0.8, top_p 0.7, repetition_penalty 1.05). 채점과 서버는 `generation_config="vllm"`으로 이 값을 무시하고 greedy로 돕니다.
- **Unsloth 2026.11.3과 Mi:dm 토크나이저:** transformers 5.17에서도, 4.57.6에 GPU가 보이는 상태에서도 토크나이저 로딩이 깨집니다. `train.unsloth_external_tokenizer: true`로 토크나이저만 따로 불러 넘깁니다([1번 결과 문서](../talk/01_finetune_unsloth.md#unsloth를-쓰려면-필요했던-것)).
- **시험 학습 체크포인트:** 손으로 5 step 시험 학습을 돌렸다면 본학습 전에 `*_run/` 폴더를 치웁니다. 남겨 두면 `--resume`이 5 step짜리 체크포인트에서 이어 갑니다. `stage.sh a_smoke`·`c_smoke`는 이 일을 대신 합니다.
- **실행 중인 스크립트 파일을 덮어쓰지 않습니다.** bash는 스크립트를 읽어 가며 실행합니다. 도는 중에 파일을 바꾸면 끝부분에서 깨집니다(이번에 QAT 학습이 끝난 뒤 실제로 이렇게 멈췄습니다). 오래 도는 큐는 레포 밖으로 복사해서 돌립니다.
- **INT4 속도:** tile-packed INT4(tinygemm 커널)는 요청이 적을 때 빠르고 많을 때 느립니다. 프롬프트 처리도 느려서 prefix caching 여부에 크게 좌우됩니다([3번 결과 문서](../talk/03_vllm_serving.md)).
- **온라인 벤치의 문항:** `FRESH=1` 없이 돌리면 동시 요청 수마다 같은 256문항을 다시 보냅니다. 이때 prefix caching을 켜면 프롬프트 전체가 캐시에 있어 실제보다 빨라 보입니다.
- **Python 3.14:** `datasets` 4.3.0이 pickle 오류로 깨집니다. Python 3.12 이하를 씁니다.
- **zsh:** 따옴표 없는 변수가 단어로 나뉘지 않습니다. 옵션을 변수 하나에 담아 넘기면 인자 하나로 들어가 argparse가 멈춥니다.
- **VM 비용:** VM을 멈추기만 하면 계속 청구됩니다. 쓰지 않을 때는 할당 해제합니다(`az vm deallocate`).

## 커밋할 것

- **커밋하는 것:** `quantization/results/kormedmcqa_midm/`의 결과 JSON. 매니지드 결과(`managed/`)에는 문항 id, 정답, 예측 번호, 답 앞 40자, 사용 토큰, 지연 시간, 끝난 이유, 짧은 오류 이유가 들어 있습니다. 먼저 돌린 세 파일(gpt-oss-120b, llama-4-maverick, mistral-medium-3-5)은 짧은 오류 이유 형식 이전이라, 오류 응답 본문 앞 120자(공급자의 응답 id 포함)가 그대로 있습니다. 파일럿 결과는 커밋하지 않습니다.
- **커밋하지 않는 것:** 가중치(`quantization/artifacts/`)는 gitignore 대상입니다. 데이터셋 원문과 Mi:dm이 쓴 해설 원문도 넣지 않습니다(KorMedMCQA는 CC BY-NC 2.0, [데이터와 라이선스](../DATA_AND_LICENSES.md)).
- **비밀값:** 엔드포인트 주소, 구독, 토큰은 환경 변수로만 넘깁니다. 커밋 전에 `python scripts/scan_secrets.py quantization scripts docs README.md`를 돌립니다.

## 결과를 바꾼 뒤

결과 JSON을 새로 만들었다면 문서와 차트도 맞춥니다. CI가 어긋남을 잡습니다.

```bash
python scripts/make_figures.py                     # 차트 다시 그리기 (docs/talk/figures/)
python -m quantization.managed_compare             # SLM 대 API 비교표 다시 만들기 (온라인 결과를 바꿨을 때도)
# 문서 숫자를 고친 뒤 docs/evidence_ledger.yaml의 expected 값을 맞추고
python scripts/build_evidence_index.py --emit      # docs/EVIDENCE.md 다시 만들기
python scripts/build_evidence_index.py --check     # 문서 숫자 = 결과 JSON 값인지
python -m pytest -q
```

[근거 색인](../EVIDENCE.md)은 문서의 숫자마다 어느 결과 JSON의 어느 값에서 왔는지 적어 둔 표입니다. 숫자가 결과 JSON과 다르면 `--check`가 실패합니다.
