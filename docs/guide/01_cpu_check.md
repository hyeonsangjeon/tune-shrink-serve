# 1. GPU 없이 확인 (약 1분)

GPU, 클라우드 계정, 모델 다운로드 없이 이 레포가 제대로 도는지 봅니다. 노트북이면 됩니다.

| 필요한 것 | 걸린 시간 | 비용 |
|---|---|---|
| Python 3.10 이상(CI는 3.10·3.11, 3.14에서도 확인), git | 약 1분 | 없음 |

## 순서

```bash
git clone https://github.com/hyeonsangjeon/tune-shrink-serve.git && cd tune-shrink-serve
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements/cpu.txt

python -m pytest -q                                   # 테스트
python scripts/build_evidence_index.py --check        # 문서의 숫자 = 결과 JSON의 값인지
python scripts/make_figures.py --check                # 차트가 결과 JSON에서 똑같이 다시 그려지는지
python -m quantization.managed_compare --check        # SLM 대 API 비교표가 결과 JSON에서 똑같이 나오는지
```

## 확인

| 명령 | 통과하면 보이는 줄 |
|---|---|
| `pytest` | `passed` (GPU가 없어 건너뛰는 테스트 몇 개는 `skipped`) |
| `build_evidence_index.py --check` | `[evidence] OK — 72 claims verified` |
| `make_figures.py --check` | `[figures] OK — 6 figures match the result JSONs` |
| `managed_compare --check` | `[managed_compare] OK` |

**이 1분이 확인하는 것:**
- 데이터 나누기, 프롬프트, 채점, 서빙 벤치 코드가 테스트를 통과하는지
- 문서에 적힌 숫자가 결과 JSON의 값과 같은지 ([근거 색인](../EVIDENCE.md))
- 차트와 비교표가 결과 JSON에서 똑같이 다시 만들어지는지

**확인하지 않는 것:** 학습, 양자화, 서빙을 다시 하지 않습니다. 결과 JSON 자체가 맞는지는 GPU에서 다시 돌려야 알 수 있습니다(2번부터).

## 안 될 때

- **`ModuleNotFoundError`:** 가상환경을 켜지 않았거나 `requirements/cpu.txt`를 설치하지 않은 경우입니다.
- **`[evidence] ... not found in README.md`:** 문서 숫자를 손으로 바꾼 경우입니다. 결과 JSON과 [`evidence_ledger.yaml`](../evidence_ledger.yaml)을 함께 고칩니다([8번 문서](08_troubleshooting.md#결과를-바꾼-뒤)).
- **`make_figures.py --check`가 실패:** 결과 JSON이 바뀌었는데 차트를 다시 그리지 않은 경우입니다. `python scripts/make_figures.py`로 다시 그립니다.

같은 검사가 GitHub Actions([`.github/workflows/test.yml`](../../.github/workflows/test.yml))에서 push와 PR마다 돕니다.

다음: [2. GPU VM 준비](02_gpu_setup.md)
