"""모든 단계가 같이 쓰는 것: 모델 이름, 폴더, 데이터, 프롬프트, INT4 형식.

데이터 나누기, 프롬프트 만들기, 채점은 발표 숫자를 만든 실행 코드(`quantization/`)의
함수를 그대로 부릅니다. 그래서 이 폴더로 돌려도 문항, 프롬프트, 채점이 발표 때와 같습니다.

이 파일은 GPU 라이브러리(torch, transformers, torchao, vllm)를 함수 안에서만 불러옵니다.
그래서 GPU가 없는 노트북에서도 `import common`은 됩니다(CI가 설정값을 확인할 때 씁니다).
"""
from __future__ import annotations

import json
import os
import sys

# 이 폴더의 스크립트를 `python steps/1_lora_train.py`처럼 바로 돌려도
# `quantization` 패키지를 찾을 수 있게 레포 최상위 폴더를 import 경로에 넣습니다.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from quantization import kormedmcqa as K  # noqa: E402  문항 불러오기, 답 번호 읽기, 채점
from quantization import v2_pipeline as V  # noqa: E402  프롬프트, 학습 데이터 만들기
from quantization.data_korquad import load_config  # noqa: E402  YAML 설정 읽기

# --------------------------------------------------------------------------- #
# 고정값. 발표 때 쓴 값이고, quantization/configs/kormedmcqa_midm.yaml 과 같아야 합니다.
# 다르면 CI(quantization/tests/test_steps.py)가 실패합니다.
# --------------------------------------------------------------------------- #
BASE_MODEL = "K-intelligence/Midm-2.0-Base-Instruct"           # KT Mi:dm 2.0 Base, 11.5B
BASE_REVISION = "35479c5fc9a18a5db7cc6dbadcf1db68db7beab0"     # 받을 때마다 같은 가중치가 오도록 고정
MAX_SEQ_LEN = 1024        # 프롬프트+정답 토큰 수. Mi:dm 토크나이저로 재 보면 가장 긴 것이 980
SEED = 42                 # 학습 순서와 LoRA 초깃값을 정하는 seed

# 데이터와 프롬프트 설정(시스템 프롬프트, 날짜 고정, 멈춤 토큰 등)은 이 YAML 파일에서 읽습니다.
CONFIG = os.path.join(ROOT, "quantization", "configs", "kormedmcqa_midm.yaml")

# 각 단계가 만든 모델이 들어가는 폴더. 기본값끼리 이어져 있어서 인자 없이 차례로 돌리면 됩니다.
OUT = os.path.join(ROOT, "outputs")
LORA_DIR = os.path.join(OUT, "1_lora")              # 1단계: LoRA를 합친 BF16 모델
PTQ_DIR = os.path.join(OUT, "2_int4_ptq")           # 2단계: 1단계 모델을 INT4로 바꾼 것
QAT_BF16_DIR = os.path.join(OUT, "3_int4_qat_bf16")  # 3단계 중간: INT4를 흉내 내며 더 학습한 BF16 모델
QAT_DIR = os.path.join(OUT, "3_int4_qat")           # 3단계: 위 모델을 INT4로 바꾼 것

# 문항 묶음 이름. KorMedMCQA의 train/dev/test가 시험 연도로 나뉘어 있습니다.
SPLITS = {
    "train": "train",          # 2012~2020년 문제 3,401개: 학습에만 씁니다
    "dev": "selection_dev",    # 2021년 문제 1,059개: 중간 점검용
    "test": "final_holdout",   # 2022~2024년 문제 3,009개: 발표 숫자. 모델마다 한 번만 잽니다
}


def load_settings() -> tuple[dict, dict]:
    """YAML 설정과, 거기서 뽑은 프롬프트 설정을 돌려줍니다.

    프롬프트 설정(`st`)에 들어 있는 것:
      - system: 시스템 프롬프트("…정답의 번호 하나만 숫자로 출력하세요")
      - user_template: 문제와 보기 다섯 개를 사용자 메시지로 만드는 틀
      - template_kwargs: Mi:dm 채팅 템플릿에 넘기는 날짜. 고정해야 프롬프트가 매번 같습니다
      - answer_end_token, stop_tokens, pad_token: Mi:dm에 맞춘 특수 토큰
    """
    cfg = load_config(CONFIG)
    return cfg, V.chat_settings(cfg)


def load_questions(split: str, limit: int | None = None) -> list:
    """KorMedMCQA 문항을 Hugging Face에서 받아 돌려줍니다.

    split: "train", "dev", "test" 중 하나(위 SPLITS).
    limit: 일부만 쓰고 싶을 때. 직군(의사, 간호사, 약사, 치과의사) 비율을 맞춰 seed 42로 고릅니다.
           비율을 맞추며 반올림해서 한두 개 적게 나올 수 있습니다.

    학습 문항은 seed 42로 섞인 순서로, 평가 문항은 id 순서로 옵니다.
    학습 문항과 평가 문항이 겹치면 여기서 바로 에러가 납니다.
    """
    cfg, _ = load_settings()
    if split == "train":
        examples = V.load_slices(cfg, train_seed=SEED, split=SPLITS["dev"])["train"]
    else:
        examples = V.load_slices(cfg, split=SPLITS[split])["eval"]
    if limit:
        examples = K.stratified_subset(examples, limit, SEED)
    return examples


def prepare_tokenizer(tokenizer, st: dict):
    """패딩 토큰을 정합니다. Mi:dm은 전용 패딩 토큰이 있어서 그것을 씁니다."""
    return V.prepare_tokenizer(tokenizer, st)


def make_train_dataset(tokenizer, examples: list, st: dict):
    """학습 문항을 토큰으로 바꿉니다. 손실(loss)은 정답 토큰에만 걸립니다.

    문항 하나가 이렇게 바뀝니다:
      입력  = [시스템 프롬프트 + 문제 + 보기] + [정답 번호 + 턴 끝 토큰]
      라벨  = [-100, -100, ..., -100      ] + [정답 번호 + 턴 끝 토큰]
    라벨이 -100인 자리는 손실 계산에서 빠집니다. 그래서 모델은 문제를 외우지 않고
    "이 문제의 답은 몇 번인가"만 배웁니다.
    """
    return V.tokenize_completion_only(tokenizer, examples, st["system"], MAX_SEQ_LEN,
                                      st["enable_thinking"], settings=st)


def make_collator(tokenizer):
    """길이가 다른 문항을 한 배치로 묶을 때 오른쪽을 패딩으로 채웁니다(라벨은 -100)."""
    return V.DataCollatorCompletionOnly(tokenizer.pad_token_id)


def int4_config(group_size: int):
    """INT4 형식: TorchAO `Int4WeightOnlyConfig`, 가중치만 4비트, 128개씩 묶어 scale 하나.

    가중치를 128개씩 묶어(group_size=128) 묶음마다 scale과 zero point를 따로 둡니다.
    활성값(activation)은 BF16 그대로입니다.

    `TILE_PACKED_TO_4D`는 PyTorch에 들어 있는 tinygemm INT4 커널이 읽는 저장 방식입니다.
    다른 커널 라이브러리를 따로 설치하지 않아도 되고, vLLM도 이 형식을 그대로 읽습니다.
    PTQ(2단계)와 QAT(3단계)가 같은 형식으로 저장되어야 둘을 비교할 수 있어서 여기 한 곳에 둡니다.
    """
    from torchao.quantization import Int4WeightOnlyConfig
    from torchao.quantization.quantize_.workflows.int4.int4_packing_format import Int4PackingFormat

    return Int4WeightOnlyConfig(group_size=group_size,
                                int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D)


def save_as_int4(src_dir: str, out_dir: str, group_size: int) -> None:
    """BF16 모델 폴더를 읽어 INT4로 바꾼 뒤 저장합니다. 학습은 하지 않습니다.

    transformers의 `TorchAoConfig`를 넘기면 모델을 GPU에 올리는 동안 선형층(nn.Linear)의
    가중치가 INT4로 바뀝니다.

    저장할 때 safe_serialization=False인 이유: TorchAO의 INT4 텐서는 safetensors로 저장할 수
    없어서 pickle로 저장합니다. 그래서 저장한 torch, torchao 버전과 읽는 버전이 같아야 합니다.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

    _, st = load_settings()
    tok = prepare_tokenizer(AutoTokenizer.from_pretrained(src_dir), st)
    model = AutoModelForCausalLM.from_pretrained(
        src_dir, dtype=torch.bfloat16, device_map="cuda",
        quantization_config=TorchAoConfig(quant_type=int4_config(group_size)))
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=False)
    tok.save_pretrained(out_dir)
    del model
    torch.cuda.empty_cache()


def is_int4_dir(model_dir: str) -> bool:
    """이 폴더가 TorchAO INT4로 저장된 모델인지(config.json의 quantization_config로 판단)."""
    path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(path):          # Hugging Face 모델 이름이면 BF16
        return False
    with open(path, encoding="utf-8") as fh:
        qc = json.load(fh).get("quantization_config") or {}
    return qc.get("quant_method") == "torchao"


def folder_size_gib(path: str) -> float:
    """폴더 안 파일 크기의 합(GiB)."""
    total = 0
    for root, _dirs, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return total / 2**30


def write_json(path: str, record: dict) -> None:
    """결과를 사람이 읽을 수 있는 JSON으로 남깁니다."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)
    print(f"[저장] {os.path.relpath(path, ROOT)}")
