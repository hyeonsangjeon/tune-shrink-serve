"""Method 3: Mi:dm writes its own explanations for the TRAINING questions (no other model's output).

For every 2012-2020 training item, the model (default: the untuned Mi:dm 2.0 Base Instruct, MIT)
is asked to explain and then give '정답: N'. It samples ``--k`` answers; an item is kept only when
at least ``--min-correct`` of them end in the gold answer (so a lucky guess rarely passes), and
the first correct explanation is stored. The kept explanations become extra training targets
(``data.self_explain.path``); serving still answers with the number only.

The explanations restate CC BY-NC exam content, so the JSONL goes under quantization/artifacts/
(git-ignored). Only counts and lengths go to the results folder.

    python -m quantization.self_explain --config quantization/configs/sweep/s8_self_explain.yaml \
        --k 4 --min-correct 2
GPU only (vLLM).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_OUT = "quantization/artifacts/kormedmcqa_midm/self_explain/train_explanations_seed{seed}.jsonl"


def strip_final_line(text: str) -> str:
    """The explanation without its closing '정답: N' line (the training target adds it back)."""
    from .kormedmcqa import _FINAL_RE

    t = str(text).strip()
    hits = list(_FINAL_RE.finditer(t))
    if hits:
        t = t[:hits[-1].start()].rstrip()
    return t


def select(samples: Sequence[str], gold: int, min_correct: int, max_chars: int,
           finish: Optional[Sequence[Optional[str]]] = None) -> Dict[str, Any]:
    """Pure: decide keep/drop for one item from its sampled outputs. A sample counts as correct
    only when it ends with an explicit '정답: N' line equal to gold AND stopped on its own
    (finish_reason 'stop'); cut-off samples never count."""
    from .kormedmcqa import final_answer_hit

    finish = list(finish) if finish is not None else ["stop"] * len(samples)
    finals = [final_answer_hit(s) for s in samples]
    ok = [i for i, f in enumerate(finals) if f == gold and finish[i] == "stop"]
    rec: Dict[str, Any] = {"k": len(samples), "n_correct": len(ok), "finals": finals, "keep": False,
                           "explanation": None, "n_truncated": sum(1 for f in finish if f == "length"),
                           "n_no_final_line": sum(1 for f in finals if f is None)}
    for i in ok:
        e = strip_final_line(samples[i])
        if 20 <= len(e) <= max_chars:
            rec.update(keep=len(ok) >= min_correct, explanation=e)
            break
    return rec


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-dir", default=None, help="default: the config's base model")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--min-correct", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--max-chars", type=int, default=700)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None,
                    help="pilot: only the first N training items (writes *_n<N> files; check the keep rate)")
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from .data_korquad import load_config
    from . import v2_pipeline as V
    from . import kormedmcqa as K

    cfg = load_config(args.config)
    st = V.chat_settings(cfg)
    model_dir = args.model_dir or cfg["base_model"]["selected"]
    rev = None if args.model_dir else cfg["base_model"].get("revision")
    train = V.load_slices(cfg, train_seed=args.seed, split="selection_dev")["train"]   # train only
    if args.limit:
        train = train[:int(args.limit)]
    tok = V.prepare_tokenizer(AutoTokenizer.from_pretrained(model_dir, revision=rev), st)
    prompts = [V.render_prompt(tok, K.explain_prompt_example(e), st, True) for e in train]
    llm_kwargs = dict(model=model_dir, dtype="bfloat16", gpu_memory_utilization=0.85,
                      max_model_len=args.max_model_len, generation_config="vllm")
    if rev:
        llm_kwargs["revision"] = rev
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
                        stop_token_ids=V.stop_token_ids(tok, st), skip_special_tokens=True, seed=args.seed)
    t0 = time.time()
    outs = llm.generate(prompts, sp, use_tqdm=True)
    wall = time.time() - t0

    out = args.out or DEFAULT_OUT.format(seed=args.seed)
    if args.limit and not args.out:
        out = out[:-len(".jsonl")] + f"_n{args.limit}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    kept, by_role, ncorr, lens, gen_tok, n_trunc, n_nofinal = 0, Counter(), Counter(), [], [], 0, 0
    with open(out, "w", encoding="utf-8") as fh:
        for e, o in zip(train, outs):
            samples = [c.text for c in o.outputs]
            gen_tok += [len(c.token_ids) for c in o.outputs]
            r = select(samples, e.gold, args.min_correct, args.max_chars,
                       finish=[c.finish_reason for c in o.outputs])
            ncorr[r["n_correct"]] += 1
            n_trunc += r["n_truncated"]
            n_nofinal += r["n_no_final_line"]
            if r["keep"]:
                kept += 1
                by_role[e.role] += 1
                lens.append(len(r["explanation"]))
            fh.write(json.dumps({"id": e.id, "role": e.role, "year": e.year, "gold": e.gold,
                                 "k": r["k"], "n_correct": r["n_correct"], "keep": r["keep"],
                                 "explanation": r["explanation"]}, ensure_ascii=False) + "\n")
    stats = {"model_dir": model_dir, "revision": rev, "seed": args.seed, "k": args.k,
             "min_correct": args.min_correct, "temperature": args.temperature, "top_p": args.top_p,
             "max_tokens": args.max_tokens, "n_train": len(train), "n_kept": kept,
             "kept_by_role": dict(sorted(by_role.items())),
             "n_correct_histogram": {str(k): v for k, v in sorted(ncorr.items())},
             "samples_truncated": n_trunc, "samples_without_final_line": n_nofinal,
             "explanation_chars_mean": round(statistics.mean(lens), 1) if lens else None,
             "gen_tokens_mean": round(statistics.mean(gen_tok), 1) if gen_tok else None,
             "wall_seconds": round(wall, 1), "jsonl": out,
             "note": "explanation text is kept out of git (CC BY-NC exam content); counts only here"}
    res = os.path.join(cfg["paths"]["results_dir"],
                       f"self_explain_stats_seed{args.seed}{f'_n{args.limit}' if args.limit else ''}.json")
    os.makedirs(os.path.dirname(res), exist_ok=True)
    with open(res, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    print(f"[self_explain] kept {kept}/{len(train)} (k={args.k}, min_correct={args.min_correct}) "
          f"wall={wall:.0f}s -> {out}; stats {res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
