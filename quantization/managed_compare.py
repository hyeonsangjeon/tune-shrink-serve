"""Pair every managed-API result with the self-hosted Mi:dm results on the same 2022-2024 questions.

Reads ``<results_dir>/managed/eval_managed_<deployment>.json`` (full test split only) and the
self-hosted ``eval_vllm_*_h100.json`` files, then writes one summary that the docs, the chart and
the slides all read: per-model accuracy, cost per 1,000 questions, tokens, errors, and a paired
McNemar test against the BF16 LoRA and INT4 PTQ models.

    python -m quantization.managed_compare            # writes managed/compare_final_managed.json
    python -m quantization.managed_compare --check    # CI: fail if the summary is out of date

Standard library only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence

from .kormedmcqa import paired_compare

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "quantization", "results", "kormedmcqa_midm")
SELF = {  # label -> (file, price label used for cost per 1,000 questions)
    "midm_bf16_lora": ("eval_vllm_A_bf16_seed42_h100.json", "ondemand"),
    "midm_int4_ptq": ("eval_vllm_B_int4_ptq_seed42_h100.json", "ondemand"),
}
SERVER = {"midm_bf16_lora": "online_A_bf16_mcqa_cacheoff_seed42_h100.json"}  # see docs/talk/04_cost.md ②
OUT_NAME = "compare_final_managed.json"


def _load(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def managed_files(res: str = RES) -> List[str]:
    """Full-split results only: no ``_selection_dev`` and no ``_n<size>`` suffix."""
    out = []
    for p in sorted(glob.glob(os.path.join(res, "managed", "eval_managed_*.json"))):
        r = _load(p)
        if r.get("split") == "final_holdout" and r.get("n_eval") and not re.search(r"_n\d+\.json$", p):
            out.append(p)
    return out


def summarize(res: str = RES) -> Dict[str, Any]:
    selfs = {k: _load(os.path.join(res, f)) for k, (f, _) in SELF.items()}
    rows: List[Dict[str, Any]] = []
    for k, (f, price) in SELF.items():
        r = selfs[k]
        server = None
        if k in SERVER:   # one request at a time on a vLLM server: price per hour / measured requests per second
            on = _load(os.path.join(res, SERVER[k]))
            c1 = next(x for x in on["results"] if x["concurrency"] == 1)
            usd_h = (r.get("prices_usd_per_hour") or {})[price]
            server = {"source": SERVER[k], "basis": "concurrency 1, prefix caching off, 256 questions from 2021",
                      "request_throughput_rps": c1["request_throughput_rps"],
                      "usd_per_1k_questions": round(usd_h / 3600.0 / c1["request_throughput_rps"] * 1000.0, 4)}
        rows.append({"name": k, "kind": "self_hosted", "source": f, "accuracy": r["accuracy"],
                     "server_c1_cacheoff": server,
                     "accuracy_ci95": r["accuracy_ci95"], "n_eval": r["n_eval"],
                     "usd_per_1k_questions": (r.get("usd_per_1k_questions") or {}).get(price),
                     "cost_basis": f"H100 NVL {price} list price x measured batch time (3,009 at once)",
                     "prompt_tokens_mean": r.get("prompt_tokens_mean"),
                     "gen_tokens_mean": r.get("gen_tokens_mean"), "n_errors": 0})
    pairs = []
    for p in managed_files(res):
        m = _load(p)
        rel = os.path.relpath(p, res)
        rows.append({"name": m["deployment"], "kind": "managed_api", "source": rel,
                     "models_reported": m.get("models_reported"), "accuracy": m["accuracy"],
                     "accuracy_ci95": m["accuracy_ci95"], "n_eval": m["n_eval"],
                     "usd_per_1k_questions": m.get("usd_per_1k_questions"),
                     "cost_basis": "reported usage tokens x list price per 1M tokens",
                     "prompt_tokens_mean": m["tokens"]["prompt_mean"],
                     "gen_tokens_mean": m["tokens"]["completion_mean"],
                     "reasoning_tokens_mean": m["tokens"]["reasoning_mean"],   # 0.0 in older files = not reported
                     "n_errors": m.get("n_errors"), "n_unparsed": m.get("n_unparsed"),
                     "latency_p50_s": (m.get("latency_s") or {}).get("p50"),
                     "concurrency": m.get("concurrency"), "measured_at_utc": m.get("measured_at_utc")})
        for k, (f, _) in SELF.items():
            pc = paired_compare(selfs[k]["items"], m["items"])
            pc.update({"a": k, "b": m["deployment"]})
            pairs.append(pc)
    return {"split": "final_holdout", "note": "a = self-hosted Mi:dm, b = managed deployment; "
            "only_a_correct / only_b_correct are the discordant counts for McNemar",
            "rows": rows, "pairs": pairs}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", default=RES)
    ap.add_argument("--check", action="store_true", help="fail if the written summary is out of date")
    args = ap.parse_args(argv)
    out = os.path.join(args.results_dir, "managed", OUT_NAME)
    text = json.dumps(summarize(args.results_dir), ensure_ascii=False, indent=2) + "\n"
    if args.check:
        old = open(out, encoding="utf-8").read() if os.path.exists(out) else None
        if old != text:
            print(f"[managed_compare] {os.path.relpath(out, ROOT)} is out of date "
                  "(run python -m quantization.managed_compare)")
            return 1
        print("[managed_compare] OK")
        return 0
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print("[managed_compare] wrote", os.path.relpath(out, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
