#!/usr/bin/env python3
"""Build the PyTorch Day Korea 2026 talk deck (.pptx) and talk script from the talk docs.

    pip install -r requirements/slides.txt
    python scripts/make_slides.py

Writes to build/slides/:
    pytorchday2026_small_llm_int4.pptx   16:9 deck, speaker notes = talk script
    talk_script.md                       per-slide script (same text as the notes) and timing
    numbers_check.txt                    every number on the slides and in the notes vs the docs
    figures/*.png                        charts (light theme, 3x); slide versions from make_figures.py

Sources are README.md and docs/talk/*.md only. Tables on the slides are read from the docs' markdown
tables; slide titles, short lines and notes are written here. Exits non-zero (after writing the files)
when a number in the slides or notes does not appear verbatim in the sources, when a chart does not
render with the light theme, or when the text breaks the talk's wording rules.
The .pptx is byte-for-byte reproducible on the same machine (fixed zip timestamps and metadata).
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TALK = os.path.join(ROOT, "docs", "talk")
FIG_SRC = os.path.join(TALK, "figures")
OUT = os.path.join(ROOT, "build", "slides")
PPTX_NAME = "pytorchday2026_small_llm_int4.pptx"

TITLE = "작은 모델을 더 작고 빠르게"
SUBTITLE = "Unsloth LoRA·TorchAO INT4 QAT부터 vLLM 서빙까지"
EVENT = "PyTorch Day Korea 2026 · Serve Track"
SPEAKER = "전현상 (Hyeonsang Jeon)"
AFFILIATION = ["Sr. Solution Engineer", "Microsoft · Global Black Belt, Cloud & AI Apps"]
REPO_URL = "github.com/hyeonsangjeon/tune-shrink-serve"  # shown on the last slide

# Light-theme values of the chart palette (docs/talk/figures), reused for the slides.
INK, INK2, MUTE, GRID, PANEL = "0B0B0B", "52514E", "898781", "E1E0D9", "F3F2EE"
CODE_SCALE = 0.85  # inline `code` size relative to the surrounding text
BF16_BLUE, INT4_ORANGE, UNSLOTH_AQUA = "2A78D6", "EB6834", "1BAF7A"

SLIDE_W_IN, SLIDE_H_IN = 13.333, 7.5
NOTES_MIN, NOTES_MAX = 120, 300

# Wording rules from the task brief: no repo-only aliases, no hype words.
ALIAS_PATTERNS = [
    (re.compile(r"(?<![A-Za-z0-9_])[ABC](?![A-Za-z0-9_])"), "모델 별칭 A/B/C"),
    (re.compile(r"A_bf16|B_int4|C_int4|selection_dev|final_holdout|bf16adapt"), "레포 설정 이름"),
    (re.compile(r"게이트"), "게이트"),
]
STYLE_PATTERNS = [(re.compile(w), w) for w in ("혁신적", "강력한", "획기적", "손쉽게", "에 대한", "통해")]


# --------------------------------------------------------------------------- #
# Sources: README.md + docs/talk/*.md
# --------------------------------------------------------------------------- #
def source_files() -> List[str]:
    return [os.path.join(ROOT, "README.md")] + sorted(glob.glob(os.path.join(TALK, "*.md")))


def read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def doc(name: str) -> str:
    return read(os.path.join(ROOT, name) if name == "README.md" else os.path.join(TALK, name))


def split_row(line: str) -> List[str]:
    """Split a markdown table row on '|', ignoring pipes inside `code` spans."""
    cells, cur, in_code = [], [], False
    for ch in line.strip().strip("|"):
        if ch == "`":
            in_code = not in_code
        if ch == "|" and not in_code:
            cells.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    cells.append("".join(cur).strip())
    return cells


def table_after(name: str, marker: str, nth: int = 0) -> List[List[str]]:
    """The nth markdown table after the first line that starts with `marker` (header row first)."""
    lines = read_lines(name)
    start = next((i for i, ln in enumerate(lines) if ln.strip().startswith(marker)), None)
    if start is None:
        raise SystemExit(f"[slides] '{marker}' not found in {name}")
    tables: List[List[List[str]]] = []
    i = start
    while i < len(lines) and len(tables) <= nth:
        if lines[i].lstrip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i])
                i += 1
            rows = [split_row(b) for b in block if not re.match(r"^\s*\|[\s:|-]+\|\s*$", b)]
            tables.append(rows)
        i += 1
    if len(tables) <= nth:
        raise SystemExit(f"[slides] table #{nth} after '{marker}' not found in {name}")
    return tables[nth]


_LINES_CACHE: Dict[str, List[str]] = {}


def read_lines(name: str) -> List[str]:
    if name not in _LINES_CACHE:
        _LINES_CACHE[name] = doc(name).splitlines()
    return _LINES_CACHE[name]


def clean_md(cell: str) -> str:
    """Drop links (keep their text) and link-only parentheticals; keep **bold** and `code`."""
    cell = re.sub(r"\s*\(\[[^\]]*\]\([^)]*\)\)", "", cell)
    cell = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", cell)
    return cell.strip()


def row_by_first(table: List[List[str]], first: str) -> List[str]:
    for row in table[1:]:
        if plain(row[0]) == first:
            return [clean_md(c) for c in row]
    raise SystemExit(f"[slides] row '{first}' not found in table {table[0]}")


def part(cell: str, sep: str, idx: int) -> str:
    """One piece of a doc cell such as '21.3 / 37 ms · 43 ms' (text taken as written)."""
    return [p.strip() for p in cell.split(sep)][idx]


def plain(s: str) -> str:
    return s.replace("**", "").replace("`", "").replace("\n", " ")


# --------------------------------------------------------------------------- #
# Slide model
# --------------------------------------------------------------------------- #
@dataclass
class Text:
    box: Tuple[float, float, float, float]
    lines: List[str]
    size: int = 20
    color: str = INK
    bullets: bool = True
    space: int = 8
    quote: bool = False  # verbatim data quote: exempt from the wording rules (numbers still checked)


@dataclass
class Table:
    box: Tuple[float, float, float, float]
    rows: List[List[str]]
    widths: List[float]
    size: int = 14
    first_col_bold: bool = True


@dataclass
class Image:
    box: Tuple[float, float, float]  # x, y, width (height follows the chart's aspect ratio)
    name: str


@dataclass
class Stat:
    box: Tuple[float, float, float, float]
    value: str
    label: str
    color: str = INK


@dataclass
class Code:
    box: Tuple[float, float, float, float]
    lines: List[str]
    size: int = 15
    center: bool = False


@dataclass
class Slide:
    kicker: str
    title: str
    blocks: list
    notes: str
    seconds: int
    sources: List[str]
    must_say: List[str] = field(default_factory=list)  # caveats that have to stay in the notes
    cover: bool = False
    section: bool = False  # agenda title slide: big title, one line, the agenda row


AGENDA = ["① Unsloth LoRA", "② TorchAO INT4", "③ vLLM 서빙", "④ SLM 대 API"]
SECTION_NOTES_MIN = 40  # section slides are a short spoken transition


def section_slide(no: int, line: str, notes: str, source: str) -> Slide:
    row = "      ".join(f"**{a}**" if i == no else a for i, a in enumerate(AGENDA))
    return Slide(kicker="아젠다", title=AGENDA[no], section=True, seconds=10, sources=[source],
                 blocks=[Text((0.8, 3.6, 11.7, 1.2), [line], size=24, color=INK2, bullets=False),
                         Text((0.8, 6.2, 11.7, 0.5), [row], size=16, color=MUTE, bullets=False)],
                 notes=notes)


def build_slides() -> List[Slide]:
    s: List[Slide] = []

    # 1 ---------------------------------------------------------------------
    s.append(Slide(
        kicker=EVENT, title=TITLE, cover=True, seconds=30, sources=["README.md"],
        blocks=[Text((0.8, 3.55, 11.7, 0.8), [SUBTITLE], size=26, color=INK2, bullets=False),
                Text((0.8, 5.3, 11.7, 0.6), [SPEAKER], size=22, color=INK, bullets=False),
                Text((0.8, 5.85, 11.7, 0.8), AFFILIATION, size=15, color=INK2, bullets=False, space=2)],
        notes=("안녕하세요, 전현상입니다. 오늘은 작은 모델 하나를 튜닝하고 INT4로 줄여서 vLLM으로 서빙하기까지 "
               "직접 잰 숫자를 보여 드립니다. 모든 숫자는 같은 모델, 같은 문항, 같은 GPU에서 나왔어요. 잘된 "
               "결과와 잘 안 된 결과, 그리고 이 실험으로 말할 수 없는 것도 같이 말씀드리겠습니다.")))

    # 2 ---------------------------------------------------------------------
    s.append(Slide(
        kicker="오늘 볼 것", title="같은 모델, 같은 문항, 같은 GPU로 잰 세 가지",
        seconds=60, sources=["README.md"],
        blocks=[Text((0.8, 2.1, 11.7, 3.6), [
            "**① 튜닝하고 얻은 성능** — Unsloth LoRA",
            "**② 모델 줄이고 잃은 성능** — TorchAO INT4 PTQ · QAT",
            "**③ 서빙 속도와 비용** — vLLM 서빙, ④ SLM 대 API",
        ], size=26, space=22)],
        notes=("작은 모델을 튜닝해서 서비스에 쓰려면 궁금한 게 세 가지 있어요. 튜닝하면 얼마나 좋아지는지, 줄이면 "
               "얼마나 잃는지, 서빙하면 속도와 비용은 어떤지입니다. 학습, 양자화, 서빙, 비용 순서로 이 세 질문에 "
               "답하겠습니다. 답은 모두 같은 모델과 문항, GPU로 잰 숫자입니다.")))

    # 3 ---------------------------------------------------------------------
    split = table_after("00_setup.md", "## 데이터")
    hdr = [plain(c) for c in split[0]]
    cols = [hdr.index("쓰임"), hdr.index("시험 연도"), hdr.index("문항 수")]
    split_rows = [[split[0][c] for c in cols]] + [[clean_md(r[c]) for c in cols] for r in split[1:]]
    s.append(Slide(
        kicker="실험 한 장", title="모델 하나, 시험 연도로 나눈 문항, H100 한 장, seed 하나",
        seconds=90, sources=["00_setup.md"],
        blocks=[Text((0.8, 2.0, 6.0, 3.6), [
            "**모델** Mi:dm 2.0 Base, 11.5B (BF16 21.51 GiB)",
            "**데이터** KorMedMCQA: 의사·간호사·약사·치과의사 국가시험 5지선다 기출",
            "**장비** H100 NVL 94GB 1장",
            "**seed** 42 하나",
        ], size=20, space=14),
                Table((7.2, 2.1, 5.3, 2.0), split_rows, [1.7, 2.0, 1.6], size=18),
                Text((7.2, 4.6, 5.3, 1.2), ["시험 연도가 겹치지 않게 분리",
                                            "2022~2024년 문제는 모델마다 한 번만 측정"],
                     size=15, color=INK2, space=6)],
        notes=("모델은 KT가 공개한 Mi:dm 2.0 Base, 11.5B입니다. 데이터는 KorMedMCQA입니다. "
               "의사, 간호사, 약사, 치과의사 국가시험 기출문제예요. 2012~2020년 문제로 학습하고 2021년 "
               "문제로 중간 점검만 했습니다. 최종 정답률은 2022~2024년 3,009문항으로 모델마다 한 번만 "
               "쟀습니다. 장비는 H100 NVL 한 장, seed는 42 하나입니다."),
        must_say=["seed는 42 하나"]))

    # 4 ---------------------------------------------------------------------
    item = table_after("00_setup.md", "### 한 문항은 이렇게 생겼습니다")
    question = plain(row_by_first(item, "question")[1])
    options = [o.strip() for o in plain(row_by_first(item, "A ~ E")[1]).split("/")]
    s.append(Slide(
        kicker="데이터 한 문항", title="학습 대상은 정답 번호와 끝 토큰, 두 토큰뿐",
        seconds=90, sources=["00_setup.md"],
        blocks=[Text((0.8, 1.95, 6.6, 0.5), ["간호사 국가시험 2019년 1교시 1번 (학습용 문항)"],
                     size=16, color=MUTE, bullets=False),
                Text((0.8, 2.45, 6.6, 3.7), [question] + [f"{i}. {o}" for i, o in enumerate(options, 1)],
                     size=19, bullets=False, space=6, quote=True),
                Text((7.9, 1.95, 4.6, 0.5), ["학습 대상"], size=16, color=MUTE, bullets=False),
                Code((7.9, 2.45, 4.6, 0.8), ["1<|eot_id|>"], size=26, center=True),
                Text((7.9, 3.55, 4.6, 2.6), [
                    "손실은 이 두 토큰에만",
                    "프롬프트 앞: Mi:dm 템플릿의 고정 시스템 문구(약 450토큰)와 날짜",
                    "생성: greedy",
                ], size=16, color=INK2, space=8),
                Text((0.8, 6.45, 11.7, 0.4), ["출처: KorMedMCQA (arXiv 2403.01469), CC BY-NC 2.0"],
                     size=11, color=MUTE, bullets=False)],
        notes=("실제 문항으로 간호사 국가시험 2019년 1교시 1번을 가져왔습니다. 보기 중 정답 번호 하나만 숫자로 "
               "답하게 했습니다. 학습할 때 손실은 정답 번호와 턴을 닫는 토큰, 이 두 토큰에만 겁니다. 실제 요청 "
               "앞에는 Mi:dm 템플릿의 고정 시스템 문구가 약 450토큰 붙어요. 이 문구는 뒤에서 prefix "
               "caching과 함께 다시 나옵니다.")))

    # 5 ---------------------------------------------------------------------
    s.append(Slide(
        kicker="결과 먼저", title="LoRA 튜닝으로 63.3% → 72.3%, INT4로 줄이면 2.6~2.9%p 하락",
        seconds=90, sources=["02_int4_quantization.md", "README.md"],
        blocks=[Image((1.87, 1.95, 9.6), "accuracy_test"),
                Text((1.87, 6.7, 9.6, 0.35), ["2022~2024년 3,009문항 · vLLM greedy · 막대 끝 선은 95% 신뢰구간 · "
                                              "seed 하나, H100 한 장"],
                     size=13, color=MUTE, bullets=False)],
        notes=("정답률은 2022~2024년 3,009문항 기준이에요. 튜닝 전 63.3%가 LoRA 학습 뒤 "
               "72.3%가 됐습니다. Unsloth로 학습한 모델도 72.2%로 같습니다. INT4로 줄이면 PTQ "
               "69.5%, QAT 69.7%로 내려갑니다. 둘 다 합격 참고선 60%는 넘습니다. 이 숫자는 모두 "
               "seed 하나로 H100 한 장에서 쟀습니다."),
        must_say=["seed 하나"]))

    s.append(section_slide(
        0, "같은 LoRA 학습, 엔진만 바꿨을 때 학습 시간·GPU 메모리·정답률",
        "첫 번째는 학습입니다. 같은 LoRA 학습을 transformers+PEFT와 Unsloth로 돌려서 "
        "엔진만 바꿨을 때 학습 시간과 GPU 메모리, 정답률이 어떻게 달라지는지 비교해요.", "01_finetune_unsloth.md"))

    # 6 ---------------------------------------------------------------------
    cfg = table_after("01_finetune_unsloth.md", "## 학습 설정")
    keep = ["학습 데이터", "optimizer step", "LoRA", "학습률", "정밀도", "최대 길이", "gradient checkpointing", "손실"]
    cfg_rows = [["학습 설정", "값"]] + [row_by_first(cfg, k) for k in keep]
    s.append(Slide(
        kicker="① Unsloth LoRA", title="학습 엔진만 교체, LoRA 설정은 그대로",
        seconds=90, sources=["01_finetune_unsloth.md"],
        blocks=[Text((0.8, 1.95, 11.7, 0.9), [
            "**transformers Trainer + PEFT LoRA**   VS   **Unsloth FastLanguageModel + get_peft_model**"],
            size=19, bullets=False),
                Table((0.8, 2.75, 11.7, 4.0), cfg_rows, [2.6, 9.1], size=14)],
        notes=("바꾼 것은 학습 엔진 하나입니다. 한쪽은 transformers Trainer와 PEFT, 다른 쪽은 "
               "Unsloth예요. 데이터, LoRA 설정, 학습 일정, seed, GPU는 같게 두었습니다. 다만 "
               "gradient checkpointing은 엔진마다 자기 방식을 썼습니다. 학습이 끝나면 둘 다 "
               "LoRA를 합쳐 BF16 모델 하나로 저장합니다.")))

    # 7 ---------------------------------------------------------------------
    s.append(Slide(
        kicker="① Unsloth LoRA", title="같은 LoRA 학습 24.3분 → 15.7분, 정답률은 그대로(p=0.89)",
        seconds=90, sources=["01_finetune_unsloth.md"],
        blocks=[Image((0.8, 1.9, 11.7), "finetune_engines"),
                Stat((0.8, 5.65, 3.7, 1.2), "24.3분 → 15.7분", "학습 시간, 426 step", UNSLOTH_AQUA),
                Stat((4.8, 5.65, 3.9, 1.2), "82.3 → 25.5 GiB", "학습 중 GPU 메모리, nvidia-smi로 한 번 읽은 값",
                     UNSLOTH_AQUA),
                Stat((9.0, 5.65, 3.6, 1.2), "72.3% · 72.2%", "2022~2024년 정답률, McNemar p=0.89")],
        notes=("426 step 학습이 24.3분에서 15.7분으로 35% 줄었습니다. nvidia-smi로 본 메모리는 "
               "82.3에서 25.5 GiB가 됐는데요. 한 시점에 읽은 값이라 최댓값은 아닙니다. 정답률은 72.3%와 "
               "72.2%로 p=0.89, 차이가 없습니다. gradient checkpointing 방식이 달라 차이 "
               "일부는 그 설정에서 왔을 수 있고 torch 버전도 다릅니다. seed 하나, GPU 한 종류에서 잰 "
               "결과입니다."),
        must_say=["seed 하나", "한 시점에 읽은 값"]))

    # 8 ---------------------------------------------------------------------
    s.append(Slide(
        kicker="① Unsloth LoRA · 해 보려면", title="Mi:dm을 Unsloth로 학습하려면 토크나이저 우회 필요",
        seconds=45, sources=["01_finetune_unsloth.md"],
        blocks=[Text((0.8, 2.0, 11.7, 3.4), [
            "**transformers 4.57.6 고정** — Unsloth가 같이 설치하는 5.17은 Mi:dm 토크나이저를 못 읽음",
            "**토크나이저는 따로 불러 넘김** — GPU가 보이는 상태에서 `import unsloth` 하면 다시 깨짐",
            "모델 로딩과 학습은 Unsloth",
        ], size=20, space=16),
                Code((0.8, 4.75, 11.7, 0.7), ["train.unsloth_external_tokenizer: true"], size=18),
                Text((0.8, 5.75, 11.7, 0.9), [
                    "우회 뒤 확인: 병합 모델 어휘 131,384개 그대로, 프롬프트 토큰·정지 토큰도 튜닝 전 모델과 같음"],
                    size=15, color=INK2, bullets=False)],
        notes=("직접 해 보실 분께 드리는 이야기예요. Mi:dm 2.0을 Unsloth 2026.11.3으로 그대로 "
               "불러오면 토크나이저에서 멈춥니다. 먼저 transformers를 4.57.6으로 고정합니다. 그래도 "
               "GPU가 보이는 상태에서 unsloth를 import하면 다시 깨져요. 그래서 Unsloth를 불러오지 "
               "않은 별도 프로세스에서 토크나이저를 읽어 넘겼습니다. 원인이 되는 패치는 찾지 못했습니다.")))

    s.append(section_slide(
        1, "BF16 LoRA 모델을 INT4로 줄였을 때 크기와 정답률, PTQ 대 QAT",
        "두 번째는 모델을 줄이는 일입니다. BF16 LoRA 모델을 TorchAO로 INT4로 줄이는데요. 학습 "
        "없이 바꾸는 PTQ와 더 학습하는 QAT가 정답률을 얼마나 잃는지 잽니다.", "02_int4_quantization.md"))

    # 9 ---------------------------------------------------------------------
    q = table_after("02_int4_quantization.md", "## 무엇을 바꿨나")
    q_rows = [["", plain(q[0][1]), plain(q[0][2])]] + [row_by_first(q, k) for k in
                                                       ("하는 일", "TorchAO 설정", "추가 학습", "걸린 시간")]
    s.append(Slide(
        kicker="② TorchAO INT4", title="INT4는 두 가지로 생성: 학습 없이(PTQ), 더 학습해서(QAT)",
        seconds=90, sources=["02_int4_quantization.md"],
        blocks=[Table((0.8, 1.95, 11.7, 3.6), q_rows, [1.9, 4.6, 5.2], size=14),
                Text((0.8, 5.75, 11.7, 1.2), [
                    "가중치만 INT4, 계산은 BF16 (weight-only) · 128개 가중치마다 배율 하나",
                    "`TILE_PACKED_TO_4D`: PyTorch에 들어 있는 tinygemm INT4 커널 사용",
                ], size=16, color=INK2, space=6)],
        notes=("같은 BF16 LoRA 모델을 TorchAO로 두 가지 방법으로 INT4로 줄였어요. PTQ는 학습 없이 "
               "가중치만 바꿔 17초 걸렸습니다. QAT는 가짜 양자화를 끼운 채 600 step을 더 학습해 1시간 "
               "42분 걸렸습니다. 같은 데이터로 더 학습한 차이도 비교에 함께 들어갑니다. 둘 다 가중치만 INT4이고 "
               "계산은 BF16이에요."),
        must_say=["더 학습한 차이"]))

    # 10 --------------------------------------------------------------------
    cmp_ = table_after("02_int4_quantization.md", "## 결과", nth=1)
    s.append(Slide(
        kicker="② TorchAO INT4", title="크기 21.51 → 7.20 GiB, 정답률 −2.9%p(PTQ) · −2.6%p(QAT)",
        seconds=90, sources=["02_int4_quantization.md"],
        blocks=[Stat((0.8, 2.0, 5.6, 1.3), "21.51 → 7.20 GiB", "크기, BF16 LoRA → INT4 (3분의 1)", INT4_ORANGE),
                Stat((6.9, 2.0, 5.6, 1.3), "−2.9%p · −2.6%p", "정답률, BF16 LoRA 대비 INT4 PTQ · INT4 QAT",
                     INT4_ORANGE),
                Text((0.8, 3.65, 11.7, 0.5), ["같은 문항끼리 짝지은 비교 (2022~2024년 3,009문항) · "
                                              "한쪽만 맞힌 문항 = 앞 모델만 맞힌 수 대 뒤 모델만 맞힌 수"],
                     size=15, color=MUTE, bullets=False),
                Table((0.8, 4.1, 11.7, 2.0), [[plain(c) for c in cmp_[0]]] + [[clean_md(c) for c in r]
                                                                              for r in cmp_[1:]],
                      [3.6, 2.4, 3.0, 2.7], size=16),
                Text((0.8, 6.35, 11.7, 0.5), ["두 INT4 모델 모두 튜닝 전보다 6%p 넘게 높고 합격 참고선 60%도 넘음"],
                     size=15, color=INK2, bullets=False)],
        notes=("INT4로 바꾸면 크기가 21.51에서 7.20 GiB, 3분의 1이 됩니다. 정답률은 PTQ가 "
               "2.9%p, QAT가 2.6%p 내려갔습니다. 같은 문항끼리 짝지어 McNemar 검정을 했어요. 그 "
               "결과 두 차이 모두 우연으로 보기 어렵습니다. 그래도 두 INT4 모델 모두 튜닝 전보다 6%p 넘게 "
               "높습니다. 여기 숫자도 seed 하나에서 나왔습니다."),
        must_say=["seed 하나"]))

    # 11 --------------------------------------------------------------------
    job = table_after("02_int4_quantization.md", "### 직군별 정답률")
    s.append(Slide(
        kicker="② TorchAO INT4", title="QAT는 PTQ보다 근소하게 높음(+0.2~1.0%p), 나은지는 정하지 못함",
        seconds=90, sources=["02_int4_quantization.md"],
        blocks=[Text((0.8, 1.95, 5.2, 0.5), ["직군별 정답률 (%), 2022~2024년"], size=15, color=MUTE,
                     bullets=False),
                Table((0.8, 2.45, 6.4, 2.6), [[plain(c) for c in r] for r in job], [1.9, 1.1, 1.2, 1.1, 1.1],
                      size=14),
                Text((0.8, 5.35, 6.4, 1.4), [
                    "BF16 LoRA 대비 하락: 의사·간호사·약사 2.5~4.1%p, 치과의사 0.7~1.2%p"],
                    size=15, color=INK2, bullets=False),
                Text((7.7, 1.95, 4.8, 0.5), ["이 결과로 말할 수 없는 것"], size=15, color=MUTE, bullets=False),
                Text((7.7, 2.45, 4.8, 4.3), [
                    "QAT가 PTQ보다 낫다고도, 같다고도\n할 수 없음",
                    "QAT 설정 하나: 학습률 1e-5, 600 step",
                    "INT4 방식 하나: weight-only, group 128,\ntile-packed",
                    "출발한 BF16 LoRA 모델은 둘뿐\n(여섯 번 중 네 번이 같은 모델)",
                ], size=17, space=12)],
        notes=("이 실행에서 QAT는 PTQ보다 0.2%p 높았습니다. 다시 학습하거나 seed를 바꿔 여섯 번 비교해도 "
               "QAT가 매번 0.2~1.0%p 높았습니다. 다만 그 차이가 문항에서 오는 오차와 비슷해서 나은지는 정할 "
               "수 없습니다. 정답률이 내려간 정도는 직군마다 달라요. 치과의사는 0.7~1.2%p만 내려갔습니다. "
               "QAT 설정은 학습률 1e-5, 600 step 하나뿐입니다."),
        must_say=["QAT 설정은", "하나뿐"]))

    s.append(section_slide(
        2, "BF16 대 INT4, 요청 수와 prefix caching에 따른 서빙 속도",
        "세 번째는 서빙입니다. 요청 수와 prefix caching에 따라 BF16과 INT4 중 어느 쪽이 "
        "빠른지 가립니다. 두 모델 모두 vLLM으로 띄웠어요.", "03_vllm_serving.md"))

    # 12 --------------------------------------------------------------------
    two = table_after("03_vllm_serving.md", "## 두 가지를 쟀습니다")
    two_rows = [[plain(c) for c in two[0]]] + [row_by_first(two, k) for k in ("무엇을 보나", "방법", "프롬프트",
                                                                              "생성 길이")]
    two_rows = [[c.replace(" ②", "\n②") for c in r] for r in two_rows]  # one line per request type
    s.append(Slide(
        kicker="③ vLLM 서빙", title="서빙 속도는 두 가지로 측정: 오프라인 처리량, 온라인 서빙",
        seconds=45, sources=["03_vllm_serving.md"],
        blocks=[Table((0.8, 1.95, 11.7, 3.9), two_rows, [1.8, 4.4, 5.5], size=14),
                Text((0.8, 6.1, 11.7, 0.7), [
                    "INT4 PTQ와 INT4 QAT는 저장 형식이 같아 처리량 차이 1% 안쪽 → 온라인은 INT4 PTQ만 측정"],
                    size=15, color=INK2, bullets=False)],
        notes=("서빙 속도는 두 가지로 쟀습니다. 오프라인에서는 요청 1~256개를 GPU에 한꺼번에 넣고 처리량을 "
               "봤습니다. 온라인에서 본 건 응답 시간이에요. OpenAI 호환 서버에 여러 사용자가 동시에 묻는 "
               "상황입니다. INT4 PTQ와 QAT는 형식이 같아 처리량 차이가 1% 안쪽입니다. 그래서 온라인은 "
               "PTQ로만 쟀습니다.")))

    # 13 --------------------------------------------------------------------
    s.append(Slide(
        kicker="③ vLLM 서빙 · 오프라인", title="오프라인: 요청 1~4개면 INT4가 20%, 256개면 BF16이 9배 빠름",
        seconds=90, sources=["03_vllm_serving.md"],
        blocks=[Text((0.6, 1.62, 8.2, 0.3), ["초당 생성 토큰, 요청당 128토큰 · INT4 PTQ와 INT4 QAT는 처리량이 같아 선 하나로 "
                                             "표시"], size=13, color=MUTE, bullets=False),
                Image((0.6, 2.0, 8.2), "offline_throughput"),
                Text((9.1, 2.1, 3.6, 4.6), [
                    "요청 하나, 128토큰 다 받을 때까지: BF16 **1.10초**, INT4 **0.70초**",
                    "요청 16개 이상부터 BF16이 앞섬",
                    "INT4는 초당 800토큰대에서 정체",
                    "H100 한 종류, vLLM 0.23",
                ], size=16, space=12)],
        notes=("오프라인에서 한 번에 넣은 요청이 1~4개면 INT4가 20% 안팎 빠릅니다. 요청 하나에 128토큰을 "
               "다 받는 데 BF16은 1.10초, INT4는 0.70초입니다. 요청이 16개 이상이면 BF16이 앞서고 "
               "256개에서는 9배 차이가 납니다. INT4는 초당 800토큰대에서 멈춥니다. 측정에 쓴 GPU는 "
               "H100 한 종류입니다."),
        must_say=["H100 한 종류"]))

    # 14 --------------------------------------------------------------------
    bf = table_after("03_vllm_serving.md", "BF16 (초당 처리 요청")
    i4 = table_after("03_vllm_serving.md", "INT4:")
    bf1, i41 = row_by_first(bf, "1"), row_by_first(i4, "1")
    bf64, i464 = row_by_first(bf, "64"), row_by_first(i4, "64")
    ttft = lambda cell: part(part(cell, " / ", 1), " · ", 0)  # noqa: E731  first-token median as written
    rps = lambda cell: part(cell, " / ", 0)  # noqa: E731
    online_rows = [["", "prefix caching 끔", "prefix caching 켬"],
                   ["INT4 첫 토큰 · 동시 1", ttft(i41[2]), "**" + ttft(i41[1]) + "**"],
                   ["BF16 초당 요청 · 동시 1", rps(bf1[2]), rps(bf1[1])],
                   ["INT4 초당 요청 · 동시 1", rps(i41[2]), rps(i41[1])],
                   ["BF16 초당 요청 · 동시 64", rps(bf64[2]), "**" + rps(bf64[1]) + "**"],
                   ["INT4 초당 요청 · 동시 64", rps(i464[2]), rps(i464[1])]]
    s.append(Slide(
        kicker="③ vLLM 서빙 · 온라인", title="prefix caching은 INT4에 4~6배, 그래도 요청이 몰리면 BF16이 약 3배",
        seconds=120, sources=["03_vllm_serving.md"],
        blocks=[Text((0.5, 1.62, 7.5, 0.3), ["초당 처리한 요청, 2021년 시험 문항 · 실선 prefix caching 켬, 점선 끔"],
                     size=13, color=MUTE, bullets=False),
                Image((0.5, 1.95, 7.5), "online_mcqa_requests"),
                Table((8.25, 2.0, 4.6, 2.9), online_rows, [2.2, 1.2, 1.2], size=12),
                Text((8.25, 5.1, 4.6, 1.9), [
                    "동시 요청마다 새 256문항, 켬·끔에 같은 문항. 다시 쓴 것은 모든 요청에 같은 앞부분(530토큰)뿐",
                    "prefix caching을 끈 INT4: 초당 2개에서 막힘",
                ], size=13, space=6),
                Text((0.5, 6.4, 7.5, 0.6), [
                    "시험 답 요청: 프롬프트 약 650토큰, 답 1~2토큰 → 거의 모든 시간이 프롬프트 처리(prefill)"],
                    size=14, color=INK2, bullets=False)],
        notes=("이 서비스는 프롬프트가 약 650토큰이고 답은 숫자 하나죠. 그래서 거의 모든 시간이 프롬프트 "
               "처리입니다. 동시 요청마다 새 문항을 보내서 모든 요청에 같은 앞부분만 캐시로 다시 썼어요. INT4는 "
               "prefix caching을 켜면 4~6배 빨라지고 첫 토큰이 542에서 110 ms로 줄었습니다. "
               "그래도 동시 8 이상에서는 켜도 BF16이 약 3배 많이 처리합니다. 캐싱을 끈 INT4는 초당 2개에서 "
               "막힙니다."),
        must_say=["같은 앞부분"]))

    # 15 --------------------------------------------------------------------
    gen = table_after("03_vllm_serving.md", "### 128토큰 생성 요청")
    gen_rows = [["동시 요청", "BF16 초당 토큰", "INT4 초당 토큰"]] + [
        [plain(r[0]), part(r[1], " / ", 0), part(r[2], " / ", 0)] for r in gen[1:]]
    s.append(Slide(
        kicker="③ vLLM 서빙 · 128토큰 생성", title="128토큰 생성은 요청 하나면 INT4, 동시 8 이상이면 BF16이 빠름",
        seconds=45, sources=["03_vllm_serving.md"],
        blocks=[Table((0.8, 2.0, 6.6, 3.2), gen_rows, [2.0, 2.3, 2.3], size=18),
                Text((7.9, 2.1, 4.6, 3.8), [
                    "prefix caching 켬, 프롬프트는 이미 캐시에 있음",
                    "생성 속도만 비교한 표",
                    "오프라인 결과와 같은 흐름",
                ], size=17, space=12)],
        notes=("답이 긴 대화형 경우도 봤어요. 128토큰을 생성하면 요청 하나일 때 INT4가 초당 180토큰으로 "
               "BF16의 116보다 빠릅니다. 동시 요청이 8개 이상이면 BF16이 앞서고 64개에서는 3,541 대 "
               "1,025토큰입니다. 프롬프트가 이미 캐시에 있는 상태라 생성 속도만 비교한 표로 읽으시면 됩니다.")))

    # 16 --------------------------------------------------------------------
    s.append(Slide(
        kicker="③ vLLM 서빙 · 왜", title="가설: 요청이 적으면 메모리 읽기가, 많으면 계산이 병목",
        seconds=90, sources=["03_vllm_serving.md"],
        blocks=[Text((0.8, 1.95, 5.6, 0.5), ["요청이 적을 때"], size=18, color=INT4_ORANGE, bullets=False),
                Text((0.8, 2.5, 5.6, 2.6), [
                    "가중치를 메모리에서 읽는 시간이 대부분",
                    "가중치가 3분의 1인 INT4가 유리",
                ], size=18, space=10),
                Text((6.9, 1.95, 5.6, 0.5), ["요청이 많을 때"], size=18, color=BF16_BLUE, bullets=False),
                Text((6.9, 2.5, 5.6, 2.6), [
                    "계산량 증가",
                    "BF16: 텐서 코어 행렬곱 그대로",
                    "tile-packed INT4: tinygemm 커널이 가중치를 풀면서 계산",
                ], size=18, space=10),
                Text((0.8, 5.2, 11.7, 1.6), [
                    "tinygemm 커널은 한 번에 처리하는 행이 적을 때에 맞춰져 있어 650토큰 프롬프트 처리에도 약함",
                    "커널 단위 프로파일링은 안 함. 다른 INT4 커널(예: Marlin)은 큰 배치에서 다를 수 있음",
                ], size=15, color=INK2, space=8)],
        notes=("왜 이렇게 갈리는지 짚어 볼게요. 요청이 적으면 가중치를 메모리에서 읽는 시간이 대부분입니다. 그래서 "
               "가중치가 3분의 1인 INT4가 유리합니다. 요청이 많으면 계산이 병목이 돼요. BF16은 텐서 코어 "
               "행렬곱을 그대로 쓰고 tile-packed INT4는 tinygemm 커널이 가중치를 풀면서 계산합니다. "
               "커널 단위 프로파일링은 하지 않았으므로 이 설명은 가설입니다."),
        must_say=["프로파일링은 하지"]))

    s.append(section_slide(
        3, "같은 3,009문항, 자체 GPU의 Mi:dm 대 매니지드 API 11개: 정답률과 1,000문항당 비용",
        "마지막은 비용이에요. 자체 GPU 비용을 먼저 봅니다. 그다음 같은 문항을 매니지드 API 모델 11개에 "
        "풀게 해 정답률과 비용을 나란히 놓습니다.",
        "04_cost.md"))

    # 17 --------------------------------------------------------------------
    cost = table_after("04_cost.md", "## ② 서버로 받을 때")
    cost_rows = [["조건", "BF16: 초당 요청 → 1,000문항당", "INT4: 초당 요청 → 1,000문항당"]] + [
        [clean_md(c) for c in r] for r in cost[1:]]
    s.append(Slide(
        kicker="④ 비용", title="비용은 모델보다 서빙 방식이 더 크게 좌우",
        seconds=90, sources=["04_cost.md"],
        blocks=[Text((0.6, 1.95, 6.3, 0.5), ["3,009문항을 한꺼번에 풀 때, 측정한 시간으로 계산"],
                     size=14, color=MUTE, bullets=False),
                Image((0.6, 2.4, 6.3), "cost_per_1k"),
                Text((7.2, 1.95, 5.5, 0.5), ["서버로 받을 때, 온디맨드 시간당 $6.98로 계산"], size=14, color=MUTE,
                     bullets=False),
                Table((7.2, 2.4, 5.5, 2.9), cost_rows, [2.1, 1.7, 1.7], size=12),
                Text((0.6, 5.85, 12.1, 1.0), [
                    "계산값: 청구서가 아니라 GPU 사용 시간만으로 계산",
                    "매니지드 API와의 비교는 다음 장",
                ], size=15, color=INK2, space=6)],
        notes=("비용은 계산으로 낸 값인데요. 잰 시간과 처리량에 H100 온디맨드 정가 시간당 $6.98을 곱했습니다. "
               "3,009문항을 한꺼번에 풀면 BF16이 INT4보다 15배 쌉니다. 서버로 받으면 동시 64에서 "
               "INT4는 prefix caching을 끄면 1,000문항에 $1.023, 켜면 $0.175입니다. "
               "BF16은 동시 64에서 prefix caching을 켜면 1,000문항에 $0.059입니다. 서버가 "
               "쉬지 않고 바쁘다고 가정한 값입니다. 매니지드 API와는 다음 장에서 비교해요."),
        must_say=["가정한 값"]))

    s.append(Slide(
        kicker="④ SLM 대 API", title="API 11개 중 10개가 더 정확, 한꺼번에 풀면 자체 GPU가 가장 쌈",
        seconds=75, sources=["04_cost.md"],
        blocks=[Text((0.6, 1.62, 7.0, 0.3), ["2022~2024년 3,009문항 · Azure AI Foundry Global Standard · "
                                             "비용은 정가로 계산한 값"], size=13, color=MUTE, bullets=False),
                Image((0.6, 1.95, 7.0), "slm_vs_api"),
                Text((7.9, 2.0, 4.8, 4.8), [
                    "최고 98.4%(gpt-6-astra), 튜닝한 Mi:dm은 72.3%",
                    "Mi:dm보다 낮은 API는 gpt-5.4-nano(64.9%) 하나",
                    "한꺼번에 풀면 1,000문항당 Mi:dm $0.0111이 가장 쌈",
                    "서버 동시 1·caching 끔이면 Mi:dm $0.137,\ngpt-6-luna(95.9%)는 $0.0836",
                    "API는 의료 문항 7~10개가 콘텐츠 필터에 막힘",
                ], size=15, space=10)],
        notes=("같은 3,009문항을 매니지드 API 모델 11개에도 풀게 했어요. 10개가 튜닝한 Mi:dm보다 "
               "정확했고 가장 높은 모델은 98.4%입니다. 3,009문항을 한꺼번에 풀면 자체 GPU가 가장 싸지만 한 "
               "요청씩 받는 서버에서는 더 싸고 정확한 API도 있습니다. API 모델은 크기가 공개되지 않았거나 훨씬 "
               "크고 학습 데이터는 알 수 없습니다."),
        must_say=["학습 데이터는 알 수 없"]))

    s.append(Slide(
        kicker="덧붙임 · 점수 올리기", title="학습 방법을 바꿔도 seed만 바꾼 차이를 넘지 못함",
        seconds=40, sources=["05_recipe_sweep.md"],
        blocks=[Text((0.6, 1.95, 6.9, 0.4), ["2021년 1,059문항 · 실험마다 1회 · 기준은 Unsloth LoRA"],
                     size=13, color=MUTE, bullets=False),
                Table((0.6, 2.4, 6.9, 4.1), [
                    ["바꾼 것", "2021년 정답률", "기준과 차이"],
                    ["기준 (바꾸지 않음)", "70.4%", "—"],
                    ["seed만 42 → 43", "72.0%", "+1.5%p"],
                    ["epoch 2 → 4", "70.6%", "+0.2%p"],
                    ["epoch 4 + 보기 순서 섞기", "71.6%", "+1.1%p"],
                    ["보기 섞기 + 학습률 1e-4", "71.8%", "+1.3%p"],
                    ["보기 섞기 + 학습률 4e-4", "68.9%", "−1.5%p"],
                    ["보기 섞기 + 직군 비중 조정", "70.1%", "−0.4%p"],
                    ["Mi:dm이 직접 쓴 해설 함께 학습", "71.6%", "+1.1%p"],
                ], [3.7, 1.6, 1.6], size=13),
                Text((7.9, 2.4, 4.8, 4.1), [
                    "다른 모델의 출력은 쓰지 않음:\n기출문제, 정답, 튜닝 전 Mi:dm이 직접 쓴 해설만",
                    "통과 기준(+2.0%p, p<0.05)을 넘은 실험 없음 → 2022~2024년 문제는 다시 재지 않음",
                    "seed만 바꿔도 0.9~1.5%p 차이,\n학습 방법을 바꾼 차이도 모두 이 안",
                ], size=16, space=14)],
        notes=("API와 차이가 컸거든요. 그래서 다른 모델의 출력 없이 Mi:dm 점수를 더 올릴 수 있는지 따로 "
               "쟀습니다. epoch과 학습률 말고도 보기 순서를 섞어 정답 번호까지 바꿔 봤습니다. 문항이 적은 직군을 "
               "더 자주 보여 줬어요. 튜닝 전 Mi:dm이 4번 중 2번 이상 맞힌 해설로 함께 학습했습니다. 통과 "
               "기준을 넘은 실험은 없었습니다. seed만 바꿔도 0.9~1.5%p 움직였고 모든 차이가 그 "
               "안이었습니다."),
        must_say=["다른 모델의 출력 없이", "seed만 바꿔도"]))

    # 18 --------------------------------------------------------------------
    s.append(Slide(
        kicker="가져갈 것", title="튜닝은 +9.0%p, INT4는 −2.6~2.9%p, 속도와 비용은 서빙 방식이 좌우",
        seconds=90, sources=["README.md"],
        blocks=[Text((0.8, 1.95, 11.7, 4.9), [
            "**① Unsloth LoRA** 같은 LoRA 학습 24.3분 → 15.7분, 정답률 차이 없음(p=0.89)",
            "**② TorchAO INT4** 크기 3분의 1, 정답률 −2.9%p(PTQ)·−2.6%p(QAT), QAT는 PTQ보다 근소하게 높지만 나은지는 정하지 못함",
            "**③ vLLM 서빙** INT4·동시 요청 1에서 prefix caching으로 첫 토큰 542 → 110 ms. "
            "요청이 몰리면 BF16이 앞섬: 온라인 약 3배, 오프라인 초당 7,600 대 800토큰대",
            "**④ SLM 대 API** 정답률은 API 11개 중 10개가 위(최고 98.4%), 한꺼번에 풀 때 비용은 자체 GPU BF16 $0.0111이 가장 쌈(정가로 계산)",
        ], size=19, space=18)],
        notes=("네 줄로 정리할게요. Unsloth로 같은 LoRA 학습을 24.3분에서 15.7분으로 줄였고 정답률은 "
               "같았습니다. TorchAO INT4로 줄인 모델은 크기가 3분의 1이 됐고 정답률을 2.6~2.9%p "
               "잃었습니다. QAT는 PTQ보다 조금 높았지만 나은지는 정하지 못했습니다. 서빙에서는 요청 수에 따라 "
               "빠른 쪽이 뒤집혀요. 요청이 몰리면 BF16이 앞섭니다. API 대부분이 더 정확했고 한꺼번에 풀 때는 "
               "자체 GPU가 가장 쌌습니다.")))

    # 19 --------------------------------------------------------------------
    s.append(Slide(
        kicker="한계와 다음 측정", title="이 숫자는 seed 하나, GPU 한 종류, 설정 하나에서 나온 값",
        seconds=60, sources=["00_setup.md", "01_finetune_unsloth.md", "02_int4_quantization.md",
                             "03_vllm_serving.md", "04_cost.md", "05_recipe_sweep.md"],
        blocks=[Text((0.8, 1.95, 7.0, 0.5), ["이 실험이 말할 수 없는 것"], size=16, color=MUTE, bullets=False),
                Text((0.8, 2.45, 7.0, 4.4), [
                    "seed는 42 하나: seed만 바꿔도 2021년 점수 0.9~1.5%p 차이",
                    "GPU는 H100 한 종류: 다른 GPU에서는 특히 INT4 속도가 다를 수 있음",
                    "QAT 설정 하나, INT4 방식 하나(weight-only, group 128, tile-packed)",
                    "prefix caching은 같은 앞부분만 다시 쓴 경우만 잼(같은 질문 반복은 안 잼)",
                    "API 비교: 크기 미공개이거나 훨씬 큰 모델, 학습 데이터 모름, 비용 기준이 다름(GPU 시간 대 토큰)",
                    "기출문제는 공개 자료라 사전학습 때 봤는지 확인할 수 없음",
                ], size=16, space=8),
                Text((8.3, 1.95, 4.4, 0.5), ["다음 측정"], size=16, color=MUTE, bullets=False),
                Text((8.3, 2.45, 4.4, 4.4), [
                    "다른 INT4 커널(예: Marlin)로\n요청이 몰릴 때 다시 재기",
                    "API 설정(reasoning_effort)을 바꿔 다시 재기, 반복 측정",
                    "재지 않은 것: QAT 학습률·step 수·범위, GPTQ·AWQ·W4A8",
                ], size=16, space=8)],
        notes=("이 숫자로는 말할 수 없는 것도 있어요. seed는 42 하나이고 seed만 바꿔도 점수가 1%p 안팎 "
               "움직입니다. GPU는 H100 한 종류, QAT 설정과 INT4 방식도 하나씩입니다. prefix "
               "caching은 같은 앞부분만 다시 쓴 경우를 쟀습니다. 매니지드 API 비교는 비용 기준이 서로 다르고 "
               "API 모델의 학습 데이터는 알 수 없습니다. 다음에는 다른 INT4 커널로 다시 재 보려고 해요."),
        must_say=["seed는 42 하나", "H100 한 종류", "QAT 설정", "같은 앞부분", "매니지드 API"]))

    # 20 --------------------------------------------------------------------
    s.append(Slide(
        kicker="직접 해 보기", title="GPU 없이 1분이면 이 레포가 제대로 도는지 확인",
        seconds=30, sources=["README.md"],
        blocks=[Text((0.8, 1.8, 11.9, 0.7), [f"**{REPO_URL}**"], size=26, bullets=False),
                Code((0.8, 2.7, 7.4, 2.1), [
            "pip install -r requirements/cpu.txt",
            "python -m pytest -q",
            "python scripts/build_evidence_index.py --check",
            "python scripts/make_figures.py --check",
        ], size=15),
                Text((0.8, 4.95, 7.4, 0.9), ["테스트, 문서 숫자와 결과 JSON 대조, 차트 다시 그리기 확인"],
                     size=15, color=INK2, bullets=False),
                Text((8.6, 2.7, 4.1, 0.5), ["Hugging Face 공개 모델"], size=15, color=MUTE, bullets=False),
                Text((8.6, 3.2, 4.1, 1.6), ["`HyeonSang/Midm-2.0-KorMedMCQA`",
                                            "`HyeonSang/Midm-2.0-KorMedMCQA-QAT-bf16`"], size=14, space=10),
                Text((8.6, 4.95, 4.1, 1.0), ["비상업 용도로만 사용 가능", "의료 조언용 모델 아님"],
                     size=13, color=INK2, bullets=False, space=4),
                Text((0.8, 6.0, 11.7, 0.9), ["질문"], size=30, bullets=False)],
        notes=("코드와 결과 JSON, 따라 하는 방법은 GitHub에 모두 공개했습니다. GPU 없이 이 레포가 제대로 "
               "도는지는 1분이면 확인할 수 있습니다. 확인하는 항목은 테스트, 문서 숫자와 결과 JSON 대조, 차트 "
               "다시 그리기예요. 학습된 모델은 Hugging Face에 공개했고 비상업 용도로만 쓸 수 있습니다. 의료 "
               "조언에 쓰는 모델이 아닙니다. 질문 받겠습니다. 감사합니다.")))
    return s


# --------------------------------------------------------------------------- #
# Charts: SVG -> PNG with the light theme
# --------------------------------------------------------------------------- #
def _strip_dark_media(css: str) -> Tuple[str, str]:
    """Remove @media (prefers-color-scheme:dark){...}; return (css without it, the removed block)."""
    m = re.search(r"@media\s*\(\s*prefers-color-scheme\s*:\s*dark\s*\)\s*\{", css)
    if not m:
        return css, ""
    depth, end = 0, None
    for i in range(m.end() - 1, len(css)):
        if css[i] == "{":
            depth += 1
        elif css[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        raise ValueError("unbalanced @media block")
    return css[:m.start()] + css[end:], css[m.start():end]


def light_svg(svg: str) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    """Inline the light-theme CSS variables; renderers that ignore var() or @media then match a browser."""
    m = re.search(r"<style>(.*?)</style>", svg, re.S)
    if not m:
        return svg, {}, {}
    css, dark_block = _strip_dark_media(m.group(1))
    light_vars = dict((k, v.strip()) for k, v in re.findall(r"(--[\w-]+)\s*:\s*([^;}]+)", css))
    dark_vars = dict((k, v.strip()) for k, v in re.findall(r"(--[\w-]+)\s*:\s*([^;}]+)", dark_block))

    def sub(mm: "re.Match[str]") -> str:
        if mm.group(1) not in light_vars:
            raise ValueError(f"CSS variable {mm.group(1)} has no light value")
        return light_vars[mm.group(1)]

    css = re.sub(r"var\(\s*(--[\w-]+)\s*(?:,[^)]*)?\)", sub, css)
    css = re.sub(r"(?<![\w-])--[\w-]+\s*:\s*[^;}]+;?", "", css)  # drop the now unused declarations
    return svg[:m.start(1)] + css + svg[m.end(1):], light_vars, dark_vars


def _hex(c: str) -> Tuple[int, int, int]:
    c = c.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


CHART_ZOOM = 3


def chart_svg(name: str) -> str:
    """The slide version from make_figures.SLIDE_FIGURES (larger text), else docs/talk/figures/<name>.svg."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import make_figures
    finally:
        sys.path.pop(0)
    fn = make_figures.SLIDE_FIGURES.get(name)
    return fn(slide=True) if fn else read(os.path.join(FIG_SRC, name + ".svg"))


def render_chart(name: str) -> Tuple[bytes, Tuple[int, int], List[str]]:
    """Render the chart at CHART_ZOOM with light colors; return (png, svg size, problems)."""
    import resvg_py
    from PIL import Image as PILImage

    svg = chart_svg(name)
    w, h = (int(float(x)) for x in re.search(r'width="([\d.]+)" height="([\d.]+)"', svg).groups())
    lsvg, light, dark = light_svg(svg)
    png = bytes(resvg_py.svg_to_bytes(svg_string=lsvg, zoom=CHART_ZOOM))

    problems: List[str] = []
    im = PILImage.open(io.BytesIO(png)).convert("RGBA")
    if im.size != (CHART_ZOOM * w, CHART_ZOOM * h):
        problems.append(f"{name}: PNG {im.size} is not {CHART_ZOOM}x of the SVG {w}x{h}")
    colors = im.getcolors(im.size[0] * im.size[1]) or []
    count = {}
    for n, (r, g, b, a) in colors:
        if a == 255:
            count[(r, g, b)] = count.get((r, g, b), 0) + n
    total = sum(count.values())
    top = max(count.items(), key=lambda kv: kv[1])[0] if count else None
    if top != _hex(light["--bg"]):
        problems.append(f"{name}: main color {top} is not the light background {light['--bg']}")
    if count.get(_hex(dark.get("--bg", "#000000")), 0) > 0.01 * total:
        problems.append(f"{name}: dark background color found")
    used = set(re.findall(r'class="([^"]+)"', svg))
    used = {c for cls in used for c in cls.split()}
    for var, classes in (("--s1", ("s1", "k1")), ("--s2", ("s2", "k2")), ("--s3", ("s3",)), ("--n", ("n",))):
        if not used.intersection(classes):
            continue
        lc = count.get(_hex(light[var]), 0)
        dc = count.get(_hex(dark[var]), 0) if var in dark else 0
        if lc < 50 or lc <= 10 * dc:
            problems.append(f"{name}: {var} light {light[var]} x{lc} vs dark x{dc}")
    return png, (w, h), problems


# --------------------------------------------------------------------------- #
# Fonts
# --------------------------------------------------------------------------- #
def _fc_has(family: str) -> bool:
    if not shutil.which("fc-list"):
        return False
    try:
        out = subprocess.run(["fc-list", ":", "family"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return any(family.lower() == f.strip().lower() for line in out.splitlines() for f in line.split(","))


def pick_fonts(override: Optional[str]) -> Tuple[str, str]:
    if override:
        ko = override
    elif os.path.exists("/System/Library/Fonts/AppleSDGothicNeo.ttc") or _fc_has("Apple SD Gothic Neo"):
        ko = "Apple SD Gothic Neo"
    else:
        ko = "Noto Sans KR"
        if not (_fc_has("Noto Sans KR") or _fc_has("Noto Sans CJK KR")):
            print("[slides] warning: neither Apple SD Gothic Neo nor Noto Sans KR found; slides name Noto Sans KR")
    mono = "Menlo" if os.path.exists("/System/Library/Fonts/Menlo.ttc") else "Consolas"
    return ko, mono


# --------------------------------------------------------------------------- #
# PPTX writer
# --------------------------------------------------------------------------- #
class Deck:
    def __init__(self, font: str, mono: str):
        from pptx import Presentation
        from pptx.util import Inches

        self.font, self.mono = font, mono
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(SLIDE_W_IN), Inches(SLIDE_H_IN)
        self.layout = self.prs.slide_layouts[6]  # blank

    # -- text helpers -------------------------------------------------------
    def _run(self, p, text: str, size: int, color: str, bold: bool = False, code: bool = False):
        # Apple SD Gothic Neo has no U+2212 (minus sign); give that glyph a font that has it.
        pieces = [t for t in re.split(r"(−)", text) if t]
        if len(pieces) > 1:
            for t in pieces:
                self._one_run(p, t, size, color, bold, code, face="Arial" if t == "−" else None)
            return
        self._one_run(p, text, size, color, bold, code)

    def _one_run(self, p, text: str, size: int, color: str, bold: bool, code: bool, face: Optional[str] = None):
        from pptx.dml.color import RGBColor
        from pptx.oxml.ns import qn
        from pptx.util import Pt
        from lxml import etree

        r = p.add_run()
        r.text = text
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = RGBColor.from_string(color)
        face = face or (self.mono if code else self.font)
        r.font.name = face
        latin = r._r.get_or_add_rPr().find(qn("a:latin"))
        ea = etree.SubElement(latin.getparent(), qn("a:ea"))
        ea.set("typeface", face if face == "Arial" else self.font)
        latin.addnext(ea)
        cs = etree.SubElement(latin.getparent(), qn("a:cs"))
        cs.set("typeface", face)
        ea.addnext(cs)
        return r

    def _rich(self, p, text: str, size: int, color: str, bold: bool = False):
        # "\n" inside one line is a soft line break (same paragraph, same bullet).
        for k, seg in enumerate(text.split("\n")):
            if k:
                p.add_line_break()
            for tok in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", seg):
                if not tok:
                    continue
                if tok.startswith("**"):
                    self._run(p, tok[2:-2], size, color if bold else INK, bold=True)
                elif tok.startswith("`"):
                    # Menlo looks larger than Apple SD Gothic Neo at the same point size.
                    self._run(p, tok[1:-1], round(size * CODE_SCALE * 2) / 2, color, bold=bold, code=True)
                else:
                    self._run(p, tok, size, color, bold=bold)

    @staticmethod
    def _bullet(p, size: int):
        from pptx.oxml.ns import qn
        from pptx.util import Pt
        from lxml import etree

        pPr = p._p.get_or_add_pPr()
        indent = int(Pt(size) * 1.1)
        pPr.set("marL", str(indent))
        pPr.set("indent", str(-indent))
        clr = etree.SubElement(pPr, qn("a:buClr"))
        etree.SubElement(clr, qn("a:srgbClr")).set("val", MUTE)
        etree.SubElement(pPr, qn("a:buFont")).set("typeface", "Arial")
        etree.SubElement(pPr, qn("a:buChar")).set("char", "•")

    def _box(self, slide, x, y, w, h):
        from pptx.util import Inches

        tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        return tb, tf

    def text(self, slide, t: Text):
        from pptx.util import Pt

        _, tf = self._box(slide, *t.box)
        for i, line in enumerate(t.lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.space_after = Pt(t.space)
            p.line_spacing = 1.1
            if t.bullets:
                self._bullet(p, t.size)
            self._rich(p, line, t.size, t.color)

    def stat(self, slide, st: Stat):
        from pptx.util import Pt

        _, tf = self._box(slide, *st.box)
        p = tf.paragraphs[0]
        self._run(p, st.value, 34, st.color, bold=True)
        p2 = tf.add_paragraph()
        p2.space_before = Pt(4)
        self._run(p2, st.label, 14, MUTE)

    def code(self, slide, c: Code):
        from pptx.dml.color import RGBColor
        from pptx.enum.shapes import MSO_SHAPE
        from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
        from pptx.util import Inches, Pt

        x, y, w, h = c.box
        shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        shp.adjustments[0] = 0.06
        shp.fill.solid()
        shp.fill.fore_color.rgb = RGBColor.from_string(PANEL)
        shp.line.fill.background()
        shp.shadow.inherit = False
        tf = shp.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = tf.margin_right = Inches(0.25)
        for i, line in enumerate(c.lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER if c.center else PP_ALIGN.LEFT
            p.space_after = Pt(4)
            self._run(p, line, c.size, INK, code=True)

    def table(self, slide, t: Table):
        from pptx.dml.color import RGBColor
        from pptx.enum.text import MSO_ANCHOR
        from pptx.oxml.ns import qn
        from pptx.util import Inches, Pt
        from lxml import etree

        x, y, w, h = t.box
        nrows, ncols = len(t.rows), len(t.rows[0])
        gf = slide.shapes.add_table(nrows, ncols, Inches(x), Inches(y), Inches(w), Inches(h))
        tbl = gf.table
        tblPr = gf._element.graphic.graphicData.tbl.tblPr
        tblPr.set("firstRow", "1")
        tblPr.set("bandRow", "0")
        style = tblPr.find(qn("a:tableStyleId"))
        if style is not None:
            style.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"  # "No Style, No Grid"
        for j, cw in enumerate(t.widths):
            tbl.columns[j].width = Inches(cw)
        head_h = min(h / nrows, 0.5)
        body_h = (h - head_h) / max(1, nrows - 1)
        for i, row in enumerate(t.rows):
            tbl.rows[i].height = Inches(head_h if i == 0 else body_h)
            for j, val in enumerate(row):
                cell = tbl.cell(i, j)
                tcPr = cell._tc.get_or_add_tcPr()
                for side in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
                    ln = etree.SubElement(tcPr, qn(side))
                    if side == "a:lnB" or (side == "a:lnT" and i == 0):
                        ln.set("w", str(Pt(1.25 if i == 0 else 0.75)))
                        sf = etree.SubElement(ln, qn("a:solidFill"))
                        etree.SubElement(sf, qn("a:srgbClr")).set("val", INK2 if i == 0 else GRID)
                    else:
                        ln.set("w", "0")
                        etree.SubElement(ln, qn("a:noFill"))
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor.from_string(PANEL if i == 0 else "FFFFFF")
                cell.margin_left = cell.margin_right = Inches(0.1)
                cell.margin_top = cell.margin_bottom = Inches(0.05)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                tf = cell.text_frame
                tf.word_wrap = True
                p = tf.paragraphs[0]
                if i == 0:
                    self._run(p, plain(val), t.size, INK2, bold=True)
                else:
                    self._rich(p, val, t.size, INK, bold=(j == 0 and t.first_col_bold))

    def image(self, slide, im: Image, png: bytes):
        from pptx.util import Inches

        x, y, w = im.box
        slide.shapes.add_picture(io.BytesIO(png), Inches(x), Inches(y), width=Inches(w))

    def notes(self, slide, text: str):
        tf = slide.notes_slide.notes_text_frame
        tf.text = ""
        self._run(tf.paragraphs[0], text, 14, INK)

    def add(self, sl: Slide, charts: Dict[str, bytes]):
        slide = self.prs.slides.add_slide(self.layout)
        if sl.section:
            self.text(slide, Text((0.8, 1.2, 11.7, 0.5), [sl.kicker], size=16, color=MUTE, bullets=False))
            self.text(slide, Text((0.8, 2.2, 11.7, 1.3), [sl.title], size=54, bullets=False))
            for r in slide.shapes[-1].text_frame.paragraphs[0].runs:
                r.font.bold = True
        elif sl.cover:
            self.text(slide, Text((0.8, 1.2, 11.7, 0.5), [sl.kicker], size=16, color=MUTE, bullets=False))
            self.text(slide, Text((0.8, 2.15, 11.7, 1.3), [sl.title], size=48, bullets=False))
            slide.shapes[-1].text_frame.paragraphs[0].runs[0].font.bold = True
        else:
            self.text(slide, Text((0.8, 0.4, 11.7, 0.4), [sl.kicker], size=14, color=MUTE, bullets=False))
            self.text(slide, Text((0.8, 0.78, 11.7, 1.0), [sl.title], size=28, bullets=False))
            for r in slide.shapes[-1].text_frame.paragraphs[0].runs:
                r.font.bold = True
            self.text(slide, Text((0.8, 7.08, 11.7, 0.3), [footer(sl)], size=10, color=MUTE, bullets=False))
        for b in sl.blocks:
            if isinstance(b, Text):
                self.text(slide, b)
            elif isinstance(b, Table):
                self.table(slide, b)
            elif isinstance(b, Image):
                self.image(slide, b, charts[b.name])
            elif isinstance(b, Stat):
                self.stat(slide, b)
            elif isinstance(b, Code):
                self.code(slide, b)
        self.notes(slide, sl.notes)

    def save(self, path: str):
        cp = self.prs.core_properties
        cp.title = f"{TITLE}: {SUBTITLE}"
        cp.subject = EVENT
        cp.author = cp.last_modified_by = SPEAKER
        cp.revision = 1
        cp.created = cp.modified = cp.last_printed = dt.datetime(2026, 1, 1)
        buf = io.BytesIO()
        self.prs.save(buf)
        # Rewrite the zip with fixed timestamps so the same inputs give the same bytes.
        src = zipfile.ZipFile(io.BytesIO(buf.getvalue()))
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as dst:
            for info in src.infolist():
                zi = zipfile.ZipInfo(info.filename, date_time=(1980, 1, 1, 0, 0, 0))
                zi.compress_type = zipfile.ZIP_DEFLATED
                zi.external_attr = 0o644 << 16
                dst.writestr(zi, src.read(info.filename))


def footer(sl: Slide) -> str:
    return "근거: " + ", ".join(n if n == "README.md" else "docs/talk/" + n for n in sl.sources)


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
NUM_RE = re.compile(r"(?<![0-9.,])[0-9]+(?:\.[0-9]+)?e-[0-9]+|(?<![0-9])[0-9]+(?:[.,][0-9]+)*")
NAME_CHAR = re.compile(r"[A-Za-z_/]")
WORD_CHAR = re.compile(r"[A-Za-z0-9_./:-]")


def numbers(text: str) -> List[Tuple[str, int, str]]:
    """(token, position, kind). kind '이름' = digits inside a name (INT4, BF16, H100, 00_setup.md), else '값'."""
    out = []
    for m in NUM_RE.finditer(text):
        before = text[m.start() - 1] if m.start() else ""
        after = text[m.end()] if m.end() < len(text) else ""
        name = bool(before and NAME_CHAR.match(before)) or after == "_"
        out.append((m.group(0), m.start(), "이름" if name else "값"))
    return out


def word_at(text: str, pos: int) -> str:
    a, b = pos, pos
    while a > 0 and WORD_CHAR.match(text[a - 1]):
        a -= 1
    while b < len(text) and WORD_CHAR.match(text[b]):
        b += 1
    return text[a:b].strip(".:-")


def slide_texts(sl: Slide) -> List[Tuple[str, str, bool]]:
    """(where, plain text, is_quote) for everything visible on the slide plus the notes."""
    out = [("kicker", sl.kicker, False), ("제목", sl.title, False)]
    if not (sl.cover or sl.section):
        out.append(("꼬리말", footer(sl), False))
    for b in sl.blocks:
        if isinstance(b, Text):
            out += [("본문", plain(x), b.quote) for x in b.lines]
        elif isinstance(b, Code):
            out += [("코드", x, False) for x in b.lines]
        elif isinstance(b, Stat):
            out += [("숫자", b.value, False), ("숫자", b.label, False)]
        elif isinstance(b, Table):
            out += [("표", plain(c), False) for r in b.rows for c in r]
    out.append(("노트", sl.notes, False))
    return out


def source_index() -> Dict[Tuple[str, str], List[str]]:
    idx: Dict[Tuple[str, str], List[str]] = {}
    for path in source_files():
        rel = os.path.relpath(path, ROOT)
        for ln_no, line in enumerate(read(path).splitlines(), 1):
            for tok, _, kind in numbers(line):
                idx.setdefault((kind, tok), []).append(f"{rel}:{ln_no}")
    return idx


def check_numbers(slides: List[Slide]) -> Tuple[str, int, int, int]:
    idx = source_index()
    lines: List[str] = []
    values, name_digits, missing, distinct = 0, 0, [], set()
    names: Dict[str, list] = {}
    for no, sl in enumerate(slides, 1):
        lines.append(f"\n#{no:02d} {sl.title}")
        for where, text, _ in slide_texts(sl):
            for tok, pos, kind in numbers(text):
                distinct.add(tok)
                found = idx.get((kind, tok))
                ctx = text[max(0, pos - 14):pos + len(tok) + 14].replace("\n", " ")
                if not found:
                    missing.append((no, where, kind, tok, ctx))
                if kind == "이름":
                    name_digits += 1
                    w = word_at(text, pos)
                    entry = names.setdefault(w, [set(), True])
                    entry[0].add((no, where, text.find(w, max(0, pos - len(w)))))
                    entry[1] = entry[1] and bool(found)
                    if not found:
                        lines.append(f"  없음 {tok:<12} [{where}] 이름 {w}")
                    continue
                values += 1
                if found:
                    more = f" 외 {len(found) - 2}곳" if len(found) > 2 else ""
                    lines.append(f"  OK   {tok:<12} [{where}] …{ctx}…  ← {', '.join(found[:2])}{more}")
                else:
                    lines.append(f"  없음 {tok:<12} [{where}] …{ctx}…")
    total = values + name_digits
    head = [
        "numbers_check — 슬라이드와 발표자 노트의 숫자를 원본 문서와 대조",
        "",
        "원본: " + ", ".join(os.path.relpath(p, ROOT) for p in source_files()),
        "대상: 슬라이드 글(kicker, 제목, 본문, 표, 큰 숫자, 코드, 꼬리말)과 발표자 노트.",
        "      차트 그림 안의 숫자는 docs/talk/figures/*.svg 그대로라 대상이 아닙니다(make_figures.py --check가 확인).",
        "방법: 숫자를 쉼표·소수점까지 그대로 뽑아(예: 3,009 / 0.000001 / 2e-4), 원본에서 같은 규칙으로 뽑은 숫자에",
        "      똑같은 문자열이 있는지 봅니다. 값 숫자는 원본의 값 숫자와만 맞춥니다. INT4·BF16·H100·파일 이름처럼",
        "      이름에 붙은 숫자는 '이름' 숫자로 따로 세고, 원본의 이름 숫자와 맞춥니다.",
        "",
        f"결과: 숫자 {total}개 (값 {values}개 + 이름에 붙은 숫자 {total - values}개, 서로 다른 숫자 {len(distinct)}개),"
        f" 원본에 없는 숫자 {len(missing)}개 → {'OK' if not missing else 'FAIL'}",
        "",
        "[원본에 없는 숫자]",
    ]
    head += [f"  #{n:02d} [{w}] {k} {t}  …{c}…" for n, w, k, t, c in missing] or ["  (없음)"]
    head += ["", "[이름에 붙은 숫자]  이름 (나온 횟수) — 이름 속 숫자가 모두 원본에 있으면 OK"]
    head += [f"  {'OK  ' if ok else '없음'} {w} ({len(seen)})" for w, (seen, ok) in sorted(names.items())]
    head += ["", "[슬라이드별 값 숫자]  OK = 원본에 있음, ← 처음 찾은 곳 두 군데"]
    return "\n".join(head + lines) + "\n", total, len(distinct), len(missing)


def check_wording(slides: List[Slide]) -> List[str]:
    problems = []
    for no, sl in enumerate(slides, 1):
        for where, text, quote in slide_texts(sl):
            for pat, label in ALIAS_PATTERNS:
                if pat.search(text):
                    problems.append(f"#{no:02d} [{where}] 레포 별칭 '{label}': {text}")
            if quote:
                continue
            for pat, label in STYLE_PATTERNS:
                if pat.search(text):
                    problems.append(f"#{no:02d} [{where}] 쓰지 않을 말 '{label}': {text}")
        n = len(sl.notes)
        lo = SECTION_NOTES_MIN if sl.section else NOTES_MIN
        if not lo <= n <= NOTES_MAX:
            problems.append(f"#{no:02d} 노트 {n}자 (기준 {lo}~{NOTES_MAX}자)")
        for sentence in re.split(r"(?<=[.?!])\s+", sl.notes.strip()):
            if not sentence.endswith(("다.", "요.", "죠.")):  # spoken script: 합니다체 with some 해요체
                problems.append(f"#{no:02d} 노트 문장이 '~다.', '~요.', '~죠.'로 끝나지 않음: {sentence}")
        for phrase in sl.must_say:
            if phrase not in sl.notes:
                problems.append(f"#{no:02d} 노트에 꼭 남길 말이 빠짐: {phrase}")
    return problems


def mmss(sec: int) -> str:
    return f"{sec // 60}:{sec % 60:02d}"


def talk_script(slides: List[Slide]) -> str:
    total = sum(s.seconds for s in slides)
    out = [f"# 발표 대본 — {TITLE}", "",
           f"{EVENT} · {SUBTITLE}", "",
           "이 파일은 `scripts/make_slides.py`가 만듭니다. 대본을 고치려면 스크립트를 고친 뒤 다시 돌립니다. "
           "아래 대본은 슬라이드 발표자 노트와 같은 글입니다.", "",
           "| # | 슬라이드 제목 | 예상 시간 | 대본 글자 수 |", "|---|---|---|---|"]
    for i, s in enumerate(slides, 1):
        out.append(f"| {i} | {s.title} | {mmss(s.seconds)} | {len(s.notes)} |")
    out += [f"| | **합계** | **{mmss(total)}** | |", "",
            f"발표 {mmss(total)} + 질문 {mmss(30 * 60 - total)} = 30:00", ""]
    for i, s in enumerate(slides, 1):
        out += [f"## {i}. {s.title}", "", f"예상 시간 {mmss(s.seconds)} · {s.kicker}", "", s.notes, ""]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=OUT, help="output folder (default: build/slides)")
    ap.add_argument("--font", default=None, help="Korean font name (default: Apple SD Gothic Neo, else Noto Sans KR)")
    args = ap.parse_args(argv)

    slides = build_slides()
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)

    charts: Dict[str, bytes] = {}
    chart_problems: List[str] = []
    for name in sorted({b.name for s in slides for b in s.blocks if isinstance(b, Image)}):
        png, _, probs = render_chart(name)
        charts[name] = png
        chart_problems += probs
        with open(os.path.join(args.out, "figures", name + ".png"), "wb") as fh:
            fh.write(png)

    font, mono = pick_fonts(args.font)
    deck = Deck(font, mono)
    for sl in slides:
        deck.add(sl, charts)
    pptx_path = os.path.join(args.out, PPTX_NAME)
    deck.save(pptx_path)

    report, total, distinct, missing = check_numbers(slides)
    with open(os.path.join(args.out, "numbers_check.txt"), "w", encoding="utf-8") as fh:
        fh.write(report)
    with open(os.path.join(args.out, "talk_script.md"), "w", encoding="utf-8") as fh:
        fh.write(talk_script(slides))

    wording = check_wording(slides)
    secs = sum(s.seconds for s in slides)
    print(f"[slides] wrote {os.path.relpath(pptx_path, ROOT)} ({len(slides)} slides, font {font}, {mmss(secs)})")
    print(f"[slides] charts: {len(charts)} rendered at {CHART_ZOOM}x with the light theme"
          + (" — PROBLEMS" if chart_problems else ", colors checked"))
    print(f"[slides] numbers: {total} ({distinct} distinct), not in sources: {missing}")
    for p in chart_problems + wording:
        print("  -", p)
    if missing or chart_problems or wording:
        print("[slides] FAIL — see build/slides/numbers_check.txt and the lines above")
        return 1
    print("[slides] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
