#!/usr/bin/env python3
"""Build quantization/results/kormedmcqa_midm/qat_control/summary.json from the run's text records.

The QAT control run (docs/qat_control.md) kept its per-question eval files on the VM's local NVMe,
which was wiped when the VM was deallocated mid-run. What survived is what the run printed:

    run_log.txt          stage timings, accuracy and correct counts, train records (seeds 42-44)
    paired_printout.txt  per-seed paired comparisons with "only one model right" counts and McNemar p
    eval_printouts.txt   accuracy per profession, printed by steps/4_evaluate.py for every model
    train_printouts.txt  loss every 20 steps and the trainer's final record

This script reads only those files (plus the talk's A and PTQ eval JSONs), checks that they agree with
each other, applies the pre-registered rule, and writes summary.json. Nothing is typed in by hand
except the VM timeline, which comes from the Azure activity log.

    python scripts/qat_control_summary.py           # rewrite summary.json
    python scripts/qat_control_summary.py --check   # CI: fail if summary.json is stale
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "quantization/results/kormedmcqa_midm"
DIR = RES / "qat_control"
N = 3009
SEEDS_PLANNED = (42, 43, 44)
EFFECT_PT, NO_EFFECT_PT = 0.5, 0.2  # docs/qat_control.md, 「비교와 판정 기준」
PRICE_PER_H = 6.98                   # Standard_NC40ads_H100_v5, on-demand list price

# From the Azure activity log (Microsoft.Compute/virtualMachines start/deallocate, read 2026-11-18).
VM_TIMELINE = {
    "start": "2026-11-17T05:31:57Z",
    "deallocated_by_org_automation": "2026-11-17T15:50:12Z",
    "deallocate_done": "2026-11-17T15:51:14Z",
    "restart_to_copy_logs": "2026-11-18T06:06:10Z",
    "deallocated_after_copy": "2026-11-18T06:08:44Z",
}


def lines(name: str) -> list[str]:
    return [l for l in (DIR / name).read_text(encoding="utf-8").splitlines() if l and not l.startswith("#")]


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def pt(a: int, b: int) -> float:
    return round(100 * (a - b) / N, 3)


def verdict(diffs: list[float]) -> str:
    mean = sum(diffs) / len(diffs)
    if (all(d > 0 for d in diffs) or all(d < 0 for d in diffs)) and abs(mean) >= EFFECT_PT:
        return "effect"
    if abs(mean) <= NO_EFFECT_PT:
        return "no effect"
    return "undecided"


def counts_from_professions(block: list[str]) -> int:
    """Correct count recovered from the per-profession lines (percent printed to 0.1, 435-885 questions each).

    With fewer than 1,000 questions one more right answer moves the percent by more than 0.1, so at most
    one count prints as the given value. Raises unless exactly one does.
    """
    total = 0
    for l in block:
        m = re.match(r"\s+(치과의사|의사|간호사|약사)\s+([\d.]+)%\s+\((\d+)문항\)", l)
        if not m:
            continue
        n = int(m.group(3))
        hit = [k for k in range(n + 1) if f"{100 * k / n:.1f}" == m.group(2)]
        if len(hit) != 1:
            raise SystemExit(f"profession line does not pin one count: {l!r}")
        total += hit[0]
    return total


def build() -> dict:
    # --- accuracy: run_log.txt has exact counts for P and seeds 42-44 --------------------------------
    acc: dict[str, dict] = {}
    for l in lines("run_log.txt"):
        m = re.match(r"4_eval_qc_(\w+)_test\.json ([\d.]+) (\d+) unparsed (\d+)$", l)
        if m:
            acc[m.group(1)] = {"accuracy": float(m.group(2)), "correct": int(m.group(3)),
                               "unparsed": int(m.group(4)), "from": "run_log.txt"}
    talk_a = json.loads((RES / "eval_vllm_A_bf16_seed42_h100.json").read_text())
    talk_b = json.loads((RES / "eval_vllm_B_int4_ptq_seed42_h100.json").read_text())
    acc["A"] = {"accuracy": round(talk_a["accuracy"], 3), "correct": sum(x["correct"] for x in talk_a["items"]),
                "unparsed": talk_a.get("n_unparsed"), "from": "eval_vllm_A_bf16_seed42_h100.json (talk)"}

    # eval_printouts.txt: recover counts from profession lines; must match run_log where both exist
    blocks: dict[str, list[str]] = {}
    cur = None
    for l in lines("eval_printouts.txt"):
        if l.startswith("== "):
            cur = l[3:].removeprefix("qc_").replace("_eval_bf16", "_bf16").replace("_eval", "")
            blocks[cur] = []
        else:
            blocks[cur].append(l)
    for name, block in blocks.items():
        c = counts_from_professions(block)
        unparsed = int(re.search(r"번호를 못 읽은 답 (\d+)개", "\n".join(block)).group(1))
        if name in acc:
            assert acc[name]["correct"] == c, (name, acc[name]["correct"], c)
        else:
            acc[name] = {"accuracy": round(100 * c / N, 3), "correct": c, "unparsed": unparsed,
                         "from": "eval_printouts.txt (count recovered from the per-profession lines)"}
    assert acc["p"]["correct"] == sum(x["correct"] for x in talk_b["items"])

    # --- paired comparisons printed on the VM (seeds 42-44) -------------------------------------------
    pairs: dict[str, dict] = {}
    p_same_as_talk = None
    for l in lines("paired_printout.txt"):
        m = re.match(r"(\(\d\) [\w ]+?|QAT vs PTQ) s(\d+)\s+(\w+)-(\w+): ([+-][\d.]+)%p\s+only \w+ (\d+) vs only \w+ (\d+)\s+p=([\d.]+)$", l)
        if m:
            label, seed, a, b, d, oa, ob, p = m.groups()
            oa, ob = int(oa), int(ob)
            assert oa - ob == acc[a]["correct"] - acc[b]["correct"], l
            assert abs(pt(acc[a]["correct"], acc[b]["correct"]) - float(d)) < 0.006, l
            assert f"{mcnemar(oa, ob):.3f}" == p, l
            pairs[f"{a}-{b}"] = {"only_first_right": oa, "only_second_right": ob, "mcnemar_p": round(mcnemar(oa, ob), 3)}
        m = re.match(r"P [\d.]+ (\d+) talk B [\d.]+ (\d+) same answer (\d+)$", l)
        if m:
            p_same_as_talk = int(m.group(3))

    # --- the four comparisons per seed ------------------------------------------------------------------
    spec = {
        "1_fake_quant (Q - X int4)": lambda s: (f"q{s}", f"x{s}"),
        "2_extra_training_int4 (X int4 - P)": lambda s: (f"x{s}", "p"),
        "3_extra_training_bf16 (X bf16 - A)": lambda s: (f"x{s}_bf16", "A"),
        "qat_vs_ptq (Q - P), not one of the three pre-registered": lambda s: (f"q{s}", "p"),
    }
    seeds_run = sorted({int(k[1:3]) for k in acc if re.fullmatch(r"[qx]\d\d(_bf16)?", k)})
    comparisons, verdicts_planned, verdicts_all = {}, {}, {}
    for name, f in spec.items():
        rows = {}
        for s in seeds_run:
            a, b = f(s)
            if a in acc and b in acc:
                row = {"first": a, "second": b, "diff_pt": pt(acc[a]["correct"], acc[b]["correct"])}
                row.update(pairs.get(f"{a}-{b}", {"only_first_right": None, "only_second_right": None, "mcnemar_p": None,
                                                  "note": "per-question files lost before this pair was compared"}))
                rows[str(s)] = row
        comparisons[name] = rows
        for seeds, out in ((SEEDS_PLANNED, verdicts_planned), (tuple(int(s) for s in rows), verdicts_all)):
            d = [rows[str(s)]["diff_pt"] for s in seeds]
            out[name] = {"seeds": list(seeds), "diffs_pt": d, "mean_pt": round(sum(d) / len(d), 3), "verdict": verdict(d)}

    # --- training records ---------------------------------------------------------------------------------
    train, loss = {}, {}
    run = None
    for l in lines("train_printouts.txt"):
        if l.startswith("== "):
            run = l[3:].removeprefix("qc_")
            train[run], loss[run] = {}, []
        elif l.startswith("step "):
            step, rec = l[5:].split(" ", 1)
            loss[run].append([int(step), ast.literal_eval(rec)["loss"]])
        elif l.startswith("final "):
            rec = ast.literal_eval(l[6:])
            train[run].update(steps=600, train_runtime_s=round(rec["train_runtime"], 1),
                              mean_train_loss=round(rec["train_loss"], 4))
        elif l.startswith("stopped at step "):
            train[run].update(steps=int(l.split()[3].split("/")[0]), stopped="VM deallocated by an org automation")
        elif "학습하는 가중치" in l:
            train[run]["fake_quant"] = "가짜 양자화를 넣은" in l
    for run, pts in loss.items():
        train[run]["last_logged"] = pts[-1]
    for l in lines("run_log.txt"):
        m = re.match(r"qc_([qx]\d\d)\.json (\{.*\})$", l)
        if m:
            rec = json.loads(m.group(2))
            train[m.group(1)].update(trained_weights=rec["trained_weights"], seed=rec["seed"])
            assert train[m.group(1)]["last_logged"][1] == rec["last_loss"]

    # --- VM time and cost ------------------------------------------------------------------------------------
    t = {k: dt.datetime.fromisoformat(v.replace("Z", "+00:00")) for k, v in VM_TIMELINE.items()}
    hours = ((t["deallocate_done"] - t["start"]) + (t["deallocated_after_copy"] - t["restart_to_copy_logs"])).total_seconds() / 3600

    return {
        "what": "QAT control: does INT4 QAT beat INT4 PTQ because of fake quantization or because of 600 more training steps?",
        "plan": "docs/qat_control.md (written before the run, 2026-11-17 05:31 UTC)",
        "test_questions": N,
        "provenance": ("Per-question eval files were on the VM's local NVMe and were lost when an org automation "
                       "deallocated the VM at 2026-11-17T15:50:12Z. Every number here comes from text the run printed "
                       "(files next to this one) or from the talk's committed eval JSONs. Paired counts exist only for "
                       "seeds 42-44; seed 45 has accuracy only; seed 45 QAT stopped at step 346/600; seed 46 never ran."),
        "p_same_answers_as_talk_ptq": p_same_as_talk,
        "models": dict(sorted(acc.items())),
        "comparisons": comparisons,
        "verdicts": {
            "rule": f"effect: every seed same sign and |mean| >= {EFFECT_PT}%p; no effect: |mean| <= {NO_EFFECT_PT}%p; else undecided",
            "planned_seeds_42_43_44": verdicts_planned,
            "all_finished_seeds (seeds 45-46 were added on 2026-11-17 09:15 UTC; only seed 45 X finished)": verdicts_all,
        },
        "train": dict(sorted(train.items())),
        "loss_every_20_steps": dict(sorted(loss.items())),
        "vm": {**VM_TIMELINE, "billed_hours": round(hours, 2), "price_per_hour_usd": PRICE_PER_H,
               "cost_usd": round(hours * PRICE_PER_H, 1)},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail if summary.json differs from a fresh build")
    args = ap.parse_args()
    text = json.dumps(build(), ensure_ascii=False, indent=1) + "\n"
    out = DIR / "summary.json"
    if args.check:
        if not out.exists() or out.read_text(encoding="utf-8") != text:
            print(f"stale: {out.relative_to(ROOT)} (run python scripts/qat_control_summary.py)")
            return 1
        print(f"ok: {out.relative_to(ROOT)} matches its records")
        return 0
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
