#!/usr/bin/env python3
"""Draw the talk's charts (SVG) from the committed result JSONs. Standard library only.

    python scripts/make_figures.py            # write docs/talk/figures/*.svg
    python scripts/make_figures.py --check    # CI: fail if a committed figure differs from the JSONs

Every number in a figure is read from quantization/results/kormedmcqa_midm/. All figure text
(titles, axis names, labels) lives in TEXT below, so wording changes happen in one place.
Colors follow a CVD-validated categorical palette (light and dark steps); the SVGs switch with
prefers-color-scheme. Values are printed on the marks and every figure has a table in the docs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "quantization", "results", "kormedmcqa_midm")
OUT = os.path.join(ROOT, "docs", "talk", "figures")

TEXT = {
    "accuracy": {
        "title": "2022–2024년 국가시험 3,009문항 정답률 (H100, vLLM, greedy)",
        "desc": "튜닝 전 Mi:dm 2.0 Base, BF16 LoRA 두 엔진, INT4 PTQ, INT4 QAT의 정답률 막대와 95% 신뢰구간",
        "rows": ["튜닝 전 Mi:dm 2.0 Base", "BF16 LoRA (transformers+PEFT)", "BF16 LoRA (Unsloth)",
                 "INT4 PTQ (TorchAO)", "INT4 QAT (TorchAO)"],
        "axis": "정답률 (%)", "ref60": "합격 참고선 60%", "ref20": "찍기 20%",
        "legend": ["튜닝 전", "BF16", "INT4"],
    },
    "engines": {
        "title": "같은 LoRA 학습, 엔진만 바꿨을 때 (H100 NVL, 426 step)",
        "desc": "transformers+PEFT와 Unsloth의 학습 시간과 학습 중 GPU 메모리",
        "panel_time": "학습 시간 (분)", "panel_mem": "학습 중 GPU 메모리 (GiB, nvidia-smi)",
        "rows": ["transformers+PEFT", "Unsloth"],
    },
    "offline": {
        "title": "오프라인 처리량: 한 번에 넣은 요청 수별 초당 생성 토큰 (H100, 요청당 128토큰)",
        "desc": "vLLM LLM.generate 배치 1~256에서 BF16 모델과 INT4 모델의 출력 토큰 처리량",
        "x": "한 번에 처리한 요청 수 (배치)", "y": "초당 생성 토큰",
        "series": ["BF16", "INT4"],
        "note": "INT4 PTQ와 INT4 QAT는 같은 형식이라 처리량이 같습니다(두 번 재서 확인). 선 하나로 그렸습니다.",
    },
    "online": {
        "title": "온라인 서빙: 동시 요청 수별 초당 처리 요청 (H100, 2021년 시험 문항, 답 1~2토큰)",
        "desc": "vLLM OpenAI 호환 서버에 동시 요청 1·8·32·64를 걸었을 때 BF16과 INT4의 초당 요청 수, prefix caching 켬과 끔",
        "x": "동시 요청 수", "y": "초당 처리한 요청",
        "series": ["BF16 켬", "BF16 끔", "INT4 켬", "INT4 끔"],
        "note": "실선 prefix caching 켬, 점선 끔. 동시 요청 수마다 새 256문항을 보냈고, 켬과 끔에 같은 문항을 보냈습니다.",
    },
    "slm_api": {
        "title": "SLM 대 API: 2022–2024년 3,009문항 정답률과 1,000문항당 비용",
        "desc": "자체 H100의 Mi:dm 2.0 Base LoRA(BF16, INT4 PTQ)와 Azure AI Foundry 매니지드 API 모델 11개의 "
                "정답률(세로)과 1,000문항당 계산 비용(가로, 로그 눈금). 자체 GPU BF16은 한꺼번에 풀 때와 "
                "서버 동시 요청 1·prefix caching 끔일 때를 선으로 이었다",
        "x": "1,000문항당 비용 (달러, 로그 눈금) · 계산값, 청구서 아님", "y": "정답률 (%)",
        "legend": ["자체 H100 · BF16 LoRA", "자체 H100 · INT4 PTQ", "매니지드 API"],
        "bf16_label": "Mi:dm BF16 LoRA", "bf16_range": "한꺼번에 ↔ 서버 동시 1·caching 끔",
        "int4_label": "Mi:dm INT4 PTQ",
        # label placement per deployment: (dx, dy, anchor) in px from the point; presentation only
        "place": {"gpt-6-astra": (6, -10, "end"), "gpt-6.1-sol": (0, -10, "middle"),
                  "gpt-5.6-sol": (0, 22, "middle"), "gpt-6-luna": (-8, 4, "end"),
                  "gpt-5.4": (8, 4, "start"), "deepseek-v4-1-flash": (-8, -6, "end"),
                  "gpt-5.4-mini": (6, -9, "start"), "gpt-oss-120b": (8, 13, "start"),
                  "llama-4-maverick": (-8, 6, "end"), "mistral-medium-3-5": (8, 4, "start"),
                  "gpt-5.4-nano": (-8, 4, "end")},
        "place_slide": {"gpt-6.1-sol": (-8, 4, "end")},   # the slide canvas is narrower
    },
    "cost": {
        "title": "시험 3,009문항을 한 번에 넣어 풀 때 1,000문항당 GPU 비용 (H100 NVL 정가)",
        "desc": "BF16과 INT4 모델의 1,000문항당 비용, 온디맨드와 Spot 시간당 정가 기준",
        "rows": ["BF16 · 온디맨드 $6.98/h", "BF16 · Spot $3.54/h", "INT4 · 온디맨드 $6.98/h", "INT4 · Spot $3.54/h"],
        "axis": "1,000문항당 달러",
    },
}

# Categorical slots 1-3 (blue, orange, aqua) + neutral, light/dark steps (validated set).
STYLE = """<style>
.r{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--mute:#898781;--grid:#e1e0d9;--base:#c3c2b7;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--n:#a9a79f}
@media (prefers-color-scheme:dark){.r{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--mute:#898781;
--grid:#2c2c2a;--base:#383835;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--n:#6d6b66}}
.bg{fill:var(--bg)}.t{fill:var(--ink);font:600 15px system-ui,-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif}
.l{fill:var(--ink2);font:13px system-ui,-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif}
.v{fill:var(--ink);font:600 13px system-ui,-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif}
.m{fill:var(--mute);font:12px system-ui,-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif}
.g{stroke:var(--grid);stroke-width:1}.b{stroke:var(--base);stroke-width:1}
.s1{fill:var(--s1)}.s2{fill:var(--s2)}.s3{fill:var(--s3)}.n{fill:var(--n)}
.k1{stroke:var(--s1)}.k2{stroke:var(--s2)}.ki{stroke:var(--ink2)}
</style>"""


def esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def load(rel: str) -> dict:
    with open(os.path.join(RES, rel), encoding="utf-8") as fh:
        return json.load(fh)


class Svg:
    def __init__(self, w: int, h: int, title: str, desc: str, heading: bool = True):
        self.w, self.h = w, h
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" class="r" width="{w}" height="{h}" '
                      f'viewBox="0 0 {w} {h}" role="img" aria-labelledby="t d">',
                      f'<title id="t">{esc(title)}</title><desc id="d">{esc(desc)}</desc>', STYLE,
                      f'<rect class="bg" width="{w}" height="{h}" rx="8"/>']
        if heading:
            self.parts.append(f'<text class="t" x="20" y="30">{esc(title)}</text>')

    def add(self, s: str) -> None:
        self.parts.append(s)

    def text(self, x, y, s, cls="l", anchor="start") -> None:
        self.add(f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}">{esc(s)}</text>')

    def line(self, x1, y1, x2, y2, cls="g", extra="") -> None:
        self.add(f'<line class="{cls}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" {extra}/>')

    def hbar(self, x0, y, length, height, cls) -> None:
        # rounded data end, square at the baseline
        r = min(4.0, height / 2, max(length, 0.0))
        x1 = x0 + max(length, 0.0)
        self.add(f'<path class="{cls}" d="M{x0:.1f},{y:.1f} H{x1 - r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} '
                 f'V{y + height - r:.1f} Q{x1:.1f},{y + height:.1f} {x1 - r:.1f},{y + height:.1f} H{x0:.1f} Z"/>')

    def done(self) -> str:
        return "\n".join(self.parts + ["</svg>"]) + "\n"


def nice_ticks(vmax: float, n: int = 5) -> List[float]:
    raw = vmax / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    return [i * step for i in range(int(math.ceil(vmax / step)) + 1)]


def fmt(v: float) -> str:
    return f"{v:,.0f}" if v >= 100 else f"{v:.1f}"   # one decimal, as in the doc tables


# --------------------------------------------------------------------------- #
# slide=True draws the same chart on a smaller canvas with the same font sizes and no heading
# (the slide carries the title), so text is larger once the chart is scaled onto a slide.
# The docs use slide=False; `--check` compares only those.
def fig_accuracy(slide: bool = False) -> str:
    T = TEXT["accuracy"]
    files = ["eval_vllm_base_seed42_h100.json", "eval_vllm_A_bf16_seed42_h100.json",
             "unsloth_A/eval_vllm_A_bf16_seed42_h100.json", "eval_vllm_B_int4_ptq_seed42_h100.json",
             "eval_vllm_C_int4_qat_seed42_h100.json"]
    cls = ["n", "s1", "s1", "s2", "s2"]
    rows = [load(f) for f in files]
    W, left, right, top, rh = (640, 240, 60, 40, 40) if slide else (760, 250, 60, 70, 40)
    H = top + rh * len(rows) + 70
    s = Svg(W, H, T["title"], T["desc"], heading=not slide)
    x0, x1 = left, W - right
    X = lambda v: x0 + (x1 - x0) * v / 100.0
    for v in range(0, 101, 20):
        s.line(X(v), top - 8, X(v), top + rh * len(rows), "g")
        s.text(X(v), top + rh * len(rows) + 18, f"{v}", "m", "middle")
    for v, lab in ((60, T["ref60"]), (20, T["ref20"])):
        s.line(X(v), top - 14, X(v), top + rh * len(rows), "ki", 'stroke-dasharray="4 3" stroke-width="1.5"')
        s.text(X(v) + 4, top - 18, lab, "m")
    for i, (d, c, name) in enumerate(zip(rows, cls, T["rows"])):
        y = top + i * rh + 8
        s.text(left - 10, y + 17, name, "l", "end")
        s.hbar(x0, y, X(d["accuracy"]) - x0, 22, c)
        lo, hi = d["accuracy_ci95"]
        s.line(X(lo), y + 11, X(hi), y + 11, "ki", 'stroke-width="1.5"')
        for e in (lo, hi):
            s.line(X(e), y + 5, X(e), y + 17, "ki", 'stroke-width="1.5"')
        s.text(X(hi) + 8, y + 16, f"{d['accuracy']:.1f}%", "v")
    s.line(x0, top - 8, x0, top + rh * len(rows), "b")
    s.text((x0 + x1) / 2, H - 22, T["axis"], "m", "middle")
    lx = 20
    for lab, c in zip(T["legend"], ["n", "s1", "s2"]):
        s.add(f'<rect class="{c}" x="{lx}" y="{H - 34}" width="12" height="12" rx="2"/>')
        s.text(lx + 18, H - 24, lab, "l")
        lx += 90
    return s.done()


def fig_engines() -> str:
    T = TEXT["engines"]
    hf, us = load("A_train_seed42.json"), load("unsloth_A/A_train_seed42.json")
    runs = {os.path.basename(os.path.dirname(r["train_json"])) + "/" + os.path.basename(r["train_json"]): r
            for r in load("train_gpu_memory_observed_h100.json")["runs"]}
    mem = [runs["kormedmcqa_midm/A_train_seed42.json"], runs["unsloth_A/A_train_seed42.json"]]
    mem_gib = [max(r["memory_used_mib"] for r in run["readings"]) / 1024 for run in mem]
    panels = [(T["panel_time"], [hf["train_seconds"] / 60, us["train_seconds"] / 60], "분"),
              (T["panel_mem"], mem_gib, "GiB")]
    W, H, top = 760, 230, 80
    s = Svg(W, H, T["title"], T["desc"])
    pw = (W - 40) / 2
    for p, (ptitle, vals, unit) in enumerate(panels):
        px = 20 + p * pw
        left, x1 = px + 150, px + pw - 50
        s.text(px, top - 22, ptitle, "v")
        vmax = max(vals) * 1.15
        X = lambda v: left + (x1 - left) * v / vmax
        for i, (name, v, c) in enumerate(zip(T["rows"], vals, ["s1", "s3"])):
            y = top + i * 46
            s.text(left - 10, y + 17, name, "l", "end")
            s.hbar(left, y, X(v) - left, 24, c)
            s.text(X(v) + 8, y + 17, f"{v:.1f} {unit}", "v")
        s.line(left, top - 6, left, top + 2 * 46 - 14, "b")
    s.text(20, H - 16, "메모리는 학습 중 nvidia-smi로 읽은 값(최댓값 아님). Unsloth의 torch 최대 할당은 24.5 GiB.", "m")
    return s.done()


def _line_chart(T: dict, xs: Sequence, series: List[Tuple[str, List[float], str, bool]],
                xlabels: Sequence[str], ymax: float, slide: bool = False) -> str:
    if slide:
        W, H, left, right, top, bottom = 500, 290, 78, 112, 22, 66
    else:
        W, H, left, right, top, bottom = 840, 400, 80, 140, 80, 70
    s = Svg(W, H, T["title"], T["desc"], heading=not slide)
    if T.get("note") and not slide:
        s.text(20, 52, T["note"], "m")
    x0, x1, y0, y1 = left, W - right, H - bottom, top
    n = len(xs)
    X = lambda i: x0 + (x1 - x0) * i / (n - 1)
    ticks = nice_ticks(ymax)
    Y = lambda v: y0 - (y0 - y1) * v / ticks[-1]
    for t in ticks:
        s.line(x0, Y(t), x1, Y(t), "g")
        s.text(x0 - 8, Y(t) + 4, f"{t:,.0f}", "m", "end")
    s.line(x0, y0, x1, y0, "b")
    for i, lab in enumerate(xlabels):
        s.text(X(i), y0 + 20, lab, "m", "middle")
    s.text((x0 + x1) / 2, H - 24, T["x"], "m", "middle")
    s.add(f'<text class="m" transform="translate(20,{(y0 + y1) / 2:.0f}) rotate(-90)" text-anchor="middle">{esc(T["y"])}</text>')
    label_y: List[float] = []
    for name, ys, k, dashed in series:
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(ys))
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        s.add(f'<polyline class="{k}" fill="none" stroke-width="2" stroke-linejoin="round" points="{pts}"{dash}/>')
        for i, v in enumerate(ys):
            fill = "bg" if dashed else k.replace("k", "s")
            s.add(f'<circle class="{fill} {k}" cx="{X(i):.1f}" cy="{Y(v):.1f}" r="4" stroke-width="2"/>')
        ly = Y(ys[-1])
        while any(abs(ly - o) < 16 for o in label_y):
            ly += 16
        label_y.append(ly)
        s.text(x1 + 10, ly + 4, f"{name} {fmt(ys[-1])}", "l")
    return s.done()


def fig_offline(slide: bool = False) -> str:
    T = TEXT["offline"]
    a, b = load("bench_A_bf16_seed42_h100.json"), load("bench_B_int4_ptq_seed42_h100.json")
    xs = [r["batch"] for r in a["sweep"]]
    ya = [r["throughput_tok_s"] for r in a["sweep"]]
    yb = [r["throughput_tok_s"] for r in b["sweep"]]
    return _line_chart(T, xs, [(T["series"][0], ya, "k1", False), (T["series"][1], yb, "k2", False)],
                       [str(x) for x in xs], max(ya + yb) * 1.05, slide=slide)


def fig_online(slide: bool = False) -> str:
    T = TEXT["online"]
    get = lambda m, c: load(f"online_{m}_mcqa_cache{c}_seed42_h100.json")["results"]
    data = [(T["series"][0], get("A_bf16", "on"), "k1", False), (T["series"][1], get("A_bf16", "off"), "k1", True),
            (T["series"][2], get("B_int4_ptq", "on"), "k2", False), (T["series"][3], get("B_int4_ptq", "off"), "k2", True)]
    xs = [r["concurrency"] for r in data[0][1]]
    series = [(n, [r["request_throughput_rps"] for r in rs], k, d) for n, rs, k, d in data]
    return _line_chart(T, xs, series, [str(x) for x in xs], max(max(v for v in s[1]) for s in series) * 1.1,
                       slide=slide)


def fig_cost(slide: bool = False) -> str:
    T = TEXT["cost"]
    a, b = load("eval_vllm_A_bf16_seed42_h100.json"), load("eval_vllm_B_int4_ptq_seed42_h100.json")
    vals = [a["usd_per_1k_questions"]["ondemand"], a["usd_per_1k_questions"]["spot"],
            b["usd_per_1k_questions"]["ondemand"], b["usd_per_1k_questions"]["spot"]]
    W, left, right, top, rh = (440, 162, 66, 16, 38) if slide else (760, 230, 90, 60, 38)
    H = top + rh * 4 + (50 if slide else 60)
    s = Svg(W, H, T["title"], T["desc"], heading=not slide)
    x0, x1 = left, W - right
    ticks = nice_ticks(max(vals) * 1.05, 4)
    X = lambda v: x0 + (x1 - x0) * v / ticks[-1]
    for t in ticks:
        s.line(X(t), top - 6, X(t), top + rh * 4, "g")
        s.text(X(t), top + rh * 4 + 18, f"{t:.2f}", "m", "middle")
    for i, (name, v, c) in enumerate(zip(T["rows"], vals, ["s1", "s1", "s2", "s2"])):
        y = top + i * rh + 6
        s.text(left - 10, y + 16, name, "l", "end")
        s.hbar(x0, y, X(v) - x0, 22, c)
        s.text(X(v) + 8, y + 16, f"${v:.4f}", "v")
    s.line(x0, top - 6, x0, top + rh * 4, "b")
    s.text((x0 + x1) / 2, H - 16, T["axis"], "m", "middle")
    return s.done()


def fig_slm_api(slide: bool = False) -> str:
    """Accuracy vs cost per 1,000 questions, log x. Everything comes from managed/compare_final_managed.json,
    including the self-hosted server cost (concurrency 1, prefix caching off; docs/talk/04_cost.md ②)."""
    T = TEXT["slm_api"]
    rows = {r["name"]: r for r in load("managed/compare_final_managed.json")["rows"]}
    bf16_server = rows["midm_bf16_lora"]["server_c1_cacheoff"]["usd_per_1k_questions"]
    W, H, left, right, top, bottom = (600, 420, 58, 24, 18, 86) if slide else (760, 470, 62, 30, 56, 84)
    s = Svg(W, H, T["title"], T["desc"], heading=not slide)
    x0, x1, y0, y1 = left, W - right, H - bottom, top
    lo, hi = math.log10(0.005), math.log10(5.0)
    X = lambda v: x0 + (x1 - x0) * (math.log10(v) - lo) / (hi - lo)
    ylo, yhi = 60.0, 100.0
    Y = lambda v: y0 - (y0 - y1) * (v - ylo) / (yhi - ylo)
    for v in range(60, 101, 10):
        s.line(x0, Y(v), x1, Y(v), "g")
        s.text(x0 - 8, Y(v) + 4, f"{v}", "m", "end")
    for v, lab in ((0.01, "$0.01"), (0.03, "$0.03"), (0.1, "$0.1"), (0.3, "$0.3"), (1, "$1"), (3, "$3")):
        s.line(X(v), y1, X(v), y0, "g")
        s.text(X(v), y0 + 18, lab, "m", "middle")
    s.line(x0, y0, x1, y0, "b")
    s.text((x0 + x1) / 2, y0 + 40, T["x"], "m", "middle")
    s.add(f'<text class="m" transform="translate(16,{(y0 + y1) / 2:.0f}) rotate(-90)" text-anchor="middle">'
          f'{esc(T["y"])}</text>')
    # self-hosted BF16: a range from the batch cost to the concurrency-1 server cost
    bf = rows["midm_bf16_lora"]
    yb = Y(bf["accuracy"])
    s.add(f'<line class="k1" x1="{X(bf["usd_per_1k_questions"]):.1f}" y1="{yb:.1f}" x2="{X(bf16_server):.1f}" '
          f'y2="{yb:.1f}" stroke-width="3" stroke-linecap="round"/>')
    for v in (bf["usd_per_1k_questions"], bf16_server):
        s.add(f'<circle class="s1 k1" cx="{X(v):.1f}" cy="{yb:.1f}" r="5" stroke-width="2"/>')
    s.text(X(bf["usd_per_1k_questions"]) - 2, yb - 12, f'{T["bf16_label"]} {bf["accuracy"]:.1f}%', "v")
    s.text(X(bf["usd_per_1k_questions"]) - 2, yb + 22, T["bf16_range"], "m")
    i4 = rows["midm_int4_ptq"]
    s.add(f'<circle class="s2 k2" cx="{X(i4["usd_per_1k_questions"]):.1f}" cy="{Y(i4["accuracy"]):.1f}" r="5" '
          f'stroke-width="2"/>')
    s.text(X(i4["usd_per_1k_questions"]) + 8, Y(i4["accuracy"]) + 16, f'{T["int4_label"]} {i4["accuracy"]:.1f}%', "l")
    for name, r in rows.items():
        if r["kind"] != "managed_api":
            continue
        cx, cy = X(r["usd_per_1k_questions"]), Y(r["accuracy"])
        s.add(f'<circle class="n" cx="{cx:.1f}" cy="{cy:.1f}" r="5" stroke="var(--bg)" stroke-width="2"/>')
        dx, dy, anchor = (slide and T["place_slide"].get(name)) or T["place"].get(name, (8, 4, "start"))
        s.text(cx + dx, cy + dy, f'{name} {r["accuracy"]:.1f}%', "l", anchor)
    lx, ly = x0, H - 18
    for lab, cls in zip(T["legend"], ["s1", "s2", "n"]):
        s.add(f'<circle class="{cls}" cx="{lx + 5}" cy="{ly - 4}" r="5"/>')
        s.text(lx + 14, ly, lab, "l")
        lx += 170 if not slide else 150
    return s.done()


FIGURES = {"accuracy_test.svg": fig_accuracy, "finetune_engines.svg": fig_engines,
           "offline_throughput.svg": fig_offline, "online_mcqa_requests.svg": fig_online,
           "cost_per_1k.svg": fig_cost, "slm_vs_api.svg": fig_slm_api}
# Slide versions (scripts/make_slides.py); not written to docs/talk/figures/.
SLIDE_FIGURES = {"accuracy_test": fig_accuracy, "offline_throughput": fig_offline,
                 "online_mcqa_requests": fig_online, "cost_per_1k": fig_cost, "slm_vs_api": fig_slm_api}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="compare with committed figures, write nothing")
    args = ap.parse_args(argv)
    bad = []
    for name, fn in FIGURES.items():
        svg = fn()
        path = os.path.join(OUT, name)
        if args.check:
            old = open(path, encoding="utf-8").read() if os.path.exists(path) else None
            if old != svg:
                bad.append(name)
        else:
            os.makedirs(OUT, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(svg)
            print("[figures] wrote", os.path.relpath(path, ROOT))
    if args.check:
        if bad:
            print("[figures] out of date (run python scripts/make_figures.py):", ", ".join(bad))
            return 1
        print(f"[figures] OK — {len(FIGURES)} figures match the result JSONs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
