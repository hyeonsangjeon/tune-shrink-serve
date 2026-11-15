"""Summarize the recipe sweep on the 2021 questions (selection only; never reads 2022-2024 results).

For each arm folder under ``<results_dir>/sweep/<arm>/`` it reads the adapter-checkpoint scores
(``ckpt_dev/eval_vllm_lora_*_selection_dev.json``) and the merged-model score
(``eval_vllm_A_bf16_seed<S>_selection_dev.json``), picks a checkpoint with a rule fixed before the
run, and compares it with a reference on the same 1,059 questions (McNemar). It also counts how
many 2021 scorings were looked at in total, so the selection optimism can be reported, and can
average option probabilities across ``*_probs.json`` files (seed/adapter ensembles).

    python -m quantization.sweep_report                       # writes sweep/summary_dev.json
    python -m quantization.sweep_report --ensemble a_probs.json b_probs.json
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
REFERENCE = os.path.join(RES, "unsloth_A", "eval_vllm_A_bf16_seed42_selection_dev.json")   # today's Unsloth LoRA
TIE_PT = 0.5        # within this many points of the best checkpoint, prefer the earlier (less trained) one
GATE_PT = 2.0       # an arm must beat the reference by this much on 2021 ...
GATE_P = 0.05       # ... and with McNemar p below this


def _load(p: str) -> Dict[str, Any]:
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _step(path: str) -> int:
    m = re.search(r"checkpoint-(\d+)", path)
    return int(m.group(1)) if m else -1


def pick_checkpoint(rows: Sequence[Dict[str, Any]], tie_pt: float = TIE_PT) -> Optional[Dict[str, Any]]:
    """Highest 2021 accuracy; among those within ``tie_pt`` of it, the earliest step."""
    if not rows:
        return None
    best = max(r["accuracy"] for r in rows)
    near = [r for r in rows if r["accuracy"] >= best - tie_pt]
    return min(near, key=lambda r: r["step"])


def gate(pair: Dict[str, Any], gate_pt: float = GATE_PT, gate_p: float = GATE_P) -> bool:
    return (pair["acc_b"] - pair["acc_a"]) >= gate_pt and pair["mcnemar_p"] < gate_p


def ensemble(files: Sequence[str]) -> Dict[str, Any]:
    """Average P(option) across probs-mode results on the same questions; accuracy of the argmax."""
    runs = [_load(f) for f in files]
    by_id: Dict[str, Dict[str, Any]] = {}
    for r in runs:
        for it in r["items"]:
            d = by_id.setdefault(it["id"], {"gold": it["gold"], "sum": [0.0] * 5, "n": 0})
            p = it.get("probs")
            if p is None:
                raise ValueError(f"{r.get('adapter') or r.get('model_dir')}: no probs (run with --mode probs)")
            d["sum"] = [a + b for a, b in zip(d["sum"], p)]
            d["n"] += 1
    common = [k for k, v in by_id.items() if v["n"] == len(runs)]
    correct = sum(1 for k in common
                  if max(range(5), key=lambda i: by_id[k]["sum"][i]) + 1 == by_id[k]["gold"])
    return {"files": [os.path.relpath(f, ROOT) for f in files], "n": len(common),
            "accuracy": round(100.0 * correct / len(common), 3) if common else None}


LOOKS_LOG = "looks.jsonl"     # one line per ad-hoc 2021 scoring (ensembles) under sweep/


def _vs(ref: Dict[str, Any], items: List[Dict[str, Any]]) -> Dict[str, Any]:
    pc = paired_compare(ref["items"], items)
    if pc["missing_in_a"] or pc["missing_in_b"]:
        raise ValueError(f"paired comparison is not on the same questions: {pc}")
    out = {k: pc[k] for k in ("acc_a", "acc_b", "only_a_correct", "only_b_correct", "mcnemar_p")}
    out["passes_gate"] = gate(pc)
    return out


def summarize(res: str = RES, reference: str = REFERENCE,
              reference_lora: Optional[str] = None) -> Dict[str, Any]:
    """Per arm: every adapter's 2021 score, the picked one, the final step, merged scores. Gates
    are reported against the merged reference and, when given, against the reference's OWN adapter
    scored through the same vLLM+LoRA path (so the path gap does not count as a gain). p-values are
    nominal: many 2021 comparisons are made; ``dev_scorings_looked_at`` says how many."""
    ref = _load(reference)
    ref_l = _load(reference_lora) if reference_lora else None
    n_ref = ref["n_eval"]
    arms, looks, skipped = {}, 0, []
    for arm_dir in sorted(glob.glob(os.path.join(res, "sweep", "*"))):
        if not os.path.isdir(arm_dir):
            continue
        all_dev = glob.glob(os.path.join(arm_dir, "ckpt_dev", "*.json")) + \
            glob.glob(os.path.join(arm_dir, "eval_vllm_*_selection_dev*.json"))
        looks += len(all_dev)
        ck = []
        for p in sorted(glob.glob(os.path.join(arm_dir, "ckpt_dev", "eval_vllm_lora_*_selection_dev.json"))):
            r = _load(p)
            if r.get("n_eval") != n_ref:
                skipped.append(os.path.relpath(p, res))
                continue
            ck.append({"file": os.path.relpath(p, res), "step": _step(r.get("adapter", p)),
                       "accuracy": r["accuracy"], "items": r["items"]})
        pick = pick_checkpoint(ck)
        last = max(ck, key=lambda c: c["step"]) if ck else None
        rec: Dict[str, Any] = {"checkpoints": [{k: v for k, v in c.items() if k != "items"} for c in ck],
                               "picked": None, "final_step": None, "merged": []}
        for key, c in (("picked", pick), ("final_step", last)):
            if c:
                rec[key] = {"file": c["file"], "step": c["step"], "accuracy": c["accuracy"],
                            "vs_reference": _vs(ref, c["items"])}
                if ref_l:
                    rec[key]["vs_reference_lora_path"] = _vs(ref_l, c["items"])
        for m in sorted(glob.glob(os.path.join(arm_dir, "eval_vllm_A_bf16_seed*_selection_dev.json"))):
            r = _load(m)
            if r.get("n_eval") != n_ref:
                skipped.append(os.path.relpath(m, res))
                continue
            mm = re.search(r"_ckpt(\d+)_", os.path.basename(m))
            rec["merged"].append({"file": os.path.relpath(m, res), "accuracy": r["accuracy"],
                                  "merged_from_step": int(mm.group(1)) if mm else "final (training end)",
                                  "vs_reference": _vs(ref, r["items"])})
        arms[os.path.basename(arm_dir)] = rec
    lp = os.path.join(res, "sweep", LOOKS_LOG)
    if os.path.exists(lp):
        with open(lp, encoding="utf-8") as fh:
            looks += sum(1 for line in fh if line.strip())
    return {"split": "selection_dev", "reference": os.path.relpath(reference, res),
            "reference_accuracy": ref["accuracy"],
            "reference_lora_path": os.path.relpath(reference_lora, res) if reference_lora else None,
            "reference_lora_accuracy": ref_l["accuracy"] if ref_l else None,
            "rule": {"tie_pt": TIE_PT, "gate_pt": GATE_PT, "gate_p": GATE_P, "p_values": "nominal"},
            "dev_scorings_looked_at": looks, "skipped_not_full_2021": skipped, "arms": arms,
            "note": "selection only; the 2022-2024 questions are measured once, after this, for one model"}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", default=RES)
    ap.add_argument("--reference", default=REFERENCE)
    ap.add_argument("--reference-lora", default=None,
                    help="the reference adapter scored through vLLM+LoRA (same path as the arms)")
    ap.add_argument("--ensemble", nargs="+", default=None, help="probs-mode result files to average")
    args = ap.parse_args(argv)
    if args.ensemble:
        r = ensemble(args.ensemble)
        os.makedirs(os.path.join(args.results_dir, "sweep"), exist_ok=True)
        with open(os.path.join(args.results_dir, "sweep", LOOKS_LOG), "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"kind": "ensemble", **r}, ensure_ascii=False) + "\n")
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return 0
    s = summarize(args.results_dir, args.reference, args.reference_lora)
    out = os.path.join(args.results_dir, "sweep", "summary_dev.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(s, fh, ensure_ascii=False, indent=2)
    for name, a in s["arms"].items():
        p = a["picked"]
        print(f"[sweep] {name}: picked={p and p['step']} acc={p and p['accuracy']} "
              f"gate={p and p['vs_reference']['passes_gate']} "
              f"merged={[(m['merged_from_step'], m['accuracy'], m['vs_reference']['passes_gate']) for m in a['merged']]}")
    print(f"[sweep] reference {s['reference_accuracy']}%, 2021 scorings looked at: {s['dev_scorings_looked_at']} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
