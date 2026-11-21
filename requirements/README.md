# 설치 파일

**한국어** | [English](README_en.md)

무엇을 하려는지에 따라 하나를 고릅니다. GPU 쪽은 Python 3.10 기준이고, CPU 확인은 3.10 이상이면 됩니다. API 비교(`datasets` 사용)는 Python 3.14에서 깨지므로 3.12 이하를 씁니다.

| 파일 | 언제 | 어디서 |
|---|---|---|
| [`cpu.txt`](cpu.txt) | 테스트, 숫자 대조, 차트 다시 그리기 (GPU 없이) | 노트북, CI |
| [`gpu.txt`](gpu.txt) | INT4 변환, 평가, 서빙 측정, transformers+PEFT 학습 | GPU VM의 `~/venv-q` |
| [`unsloth.txt`](unsloth.txt) | Unsloth로 LoRA 학습 | GPU VM의 `~/venv-unsloth` (따로 만듭니다) |
| [`slides.txt`](slides.txt) | 발표 슬라이드 만들기 | 노트북 |

- `gpu.txt` 앞에 CUDA 13.0용 torch를 먼저 설치합니다: `pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.11.0`
- GPU VM에서는 [`scripts/gpu/bootstrap.sh`](../scripts/gpu/bootstrap.sh)가 두 가상환경을 이 파일들로 만들어 줍니다.
- 이전 실험(Qwen3-8B · KorQuAD)의 설치 파일은 [`archive/qwen3_korquad/requirements.txt`](../archive/qwen3_korquad/requirements.txt)에 있습니다.
- 실제로 설치된 버전은 [`env_h100.json`](../quantization/results/kormedmcqa_midm/env_h100.json)에 남아 있습니다.
