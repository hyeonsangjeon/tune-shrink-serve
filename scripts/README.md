# scripts/

**한국어** | [English](README_en.md)

## GPU VM에서 쓰는 것 (`gpu/`)

| 스크립트 | 하는 일 | 따라하기 |
|---|---|---|
| [`gpu/bootstrap.sh`](gpu/bootstrap.sh) | 가상환경 두 개(`~/venv-q`, `~/venv-unsloth`) 설치, 드라이버·GPU·테스트 확인, 설치 버전 기록 | 2 |
| [`gpu/stage.sh`](gpu/stage.sh) | 단계 하나를 돌리고 로그(`~/work/logs/`)와 끝난 상태(`~/work/state/`)를 남김. `--detach`면 백그라운드 | 3~7 |
| [`gpu/status.sh`](gpu/status.sh) | GPU 사용량, 도는 단계, 최근 로그 | 2 |
| [`gpu/sweep_queue.sh`](gpu/sweep_queue.sh) | 점수 올리기 실험의 단계를 GPU 한 장에서 차례로 돌리는 큐 | 7 |
| [`gpu/bastion_tunnel.sh`](gpu/bastion_tunnel.sh) | (Azure) SSH가 막혔을 때 Bastion 터널 열기 | 2 |
| [`gpu/vmrun.sh`](gpu/vmrun.sh) | (Azure) 들어오는 포트 없이 `az vm run-command`로 명령 보내기 | 2 |
| [`gpu/hf_login_vm.sh`](gpu/hf_login_vm.sh), [`gpu/hf_login.py`](gpu/hf_login.py) | VM에서 Hugging Face 로그인(토큰은 직접 넣음). 가중치를 올릴 때만 필요 | 4 |

## 노트북에서 쓰는 것 (GPU 필요 없음)

| 스크립트 | 하는 일 | CI |
|---|---|---|
| [`build_evidence_index.py`](build_evidence_index.py) | 문서의 숫자가 결과 JSON의 값과 같은지 확인(`--check`), 근거 색인 `docs/EVIDENCE.md` 다시 만들기(`--emit`) | ✓ |
| [`make_figures.py`](make_figures.py) | 결과 JSON으로 차트(`docs/talk/figures/*.svg`) 그리기. `--check`는 다시 그려도 같은지 확인 | ✓ |
| [`scan_secrets.py`](scan_secrets.py) | 키, 토큰, 엔드포인트 주소, 개인 경로가 커밋에 들어가지 않았는지 검사 | ✓ |
| [`check_asset_ledger.py`](check_asset_ledger.py) | 커밋한 데이터·모델 관련 파일이 모두 라이선스 기록(`docs/asset_ledger.yaml`)에 있는지 | ✓ |
| [`make_slides.py`](make_slides.py) | `docs/talk/` 문서로 발표 슬라이드와 대본 만들기(`build/slides/`, `requirements/slides.txt` 필요). 슬라이드의 숫자가 모두 문서에 있는지도 확인 | |
