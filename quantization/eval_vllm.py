"""Served-path accuracy for KorMedMCQA: run A/B/C (or the untuned base) under vLLM.

This is the headline accuracy for the talk: the model is measured the way it is served
(vLLM, greedy, the model's own chat template with a FIXED ``date_string``). It also
records wall time, prompt/generated token counts and, when ``--usd-per-hour`` is given,
the cost per 1,000 questions on that GPU — tokenizers differ across models, so cost is
compared per question, not per token.

vLLM spawns workers, so this must run as ``__main__``. GPU only.

Usage:
  python -m quantization.eval_vllm --config quantization/configs/kormedmcqa_midm.yaml \
      --method A_bf16 --seed 42 [--split final_holdout] [--usd-per-hour 3.67]
  python -m quantization.eval_vllm --config ... --method base      # untuned baseline
Output: <results_dir>/eval_vllm_<method>_seed<S>[_<split>].json  (same schema as v2_run eval
plus engine/timing/cost fields; per-item records for paired McNemar via
``python -m quantization.kormedmcqa compare``).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time


def cost_per_1k(usd_per_hour, wall_s: float, n: int):
    if not usd_per_hour or n <= 0:
        return None
    return round(float(usd_per_hour) * (wall_s / 3600.0) / n * 1000.0, 4)


def parse_prices(usd_per_hour=None, price_args=()) -> dict:
    """``--price spot=3.537 --price ondemand=6.98`` (+ legacy ``--usd-per-hour``) -> {label: $/h}.
    Keep both Spot and on-demand: Spot alone makes self-hosting look cheaper than it is."""
    prices = {}
    if usd_per_hour is not None:
        prices["usd_per_hour"] = float(usd_per_hour)
    for p in price_args or ():
        if "=" not in p:
            raise ValueError(f"--price expects label=USD_PER_HOUR, got {p!r}")
        k, v = p.split("=", 1)
        prices[k.strip()] = float(v)
    return prices


def option_probs(step_logprobs, digit_ids) -> list:
    """P(option digit 1..5) at the first generated position from vLLM's top-k logprobs; a digit
    outside the top-k counts as 0."""
    import math

    lp = (step_logprobs or [None])[0] or {}
    return [round(math.exp(lp[t].logprob), 6) if t in lp else 0.0 for t in digit_ids]


def argmax_choice(probs) -> "int | None":
    if not probs or max(probs) <= 0:
        return None
    return max(range(len(probs)), key=lambda i: probs[i]) + 1


def _strip_mode(path: str) -> str:
    return re.sub(r"_(?:probs|explain)(?=\.json$)", "", os.path.abspath(path))


def guard_final_siblings(out_path: str, method: str, seed: int, split: str, eval_size=None,
                         force: bool = False) -> None:
    """Measure-once across decision modes: a final_holdout run in one mode (answer/probs/explain)
    is refused when the same result name already exists in another mode, unless --force. Other
    GPU tags (_h100 vs _a100) are separate tables and stay allowed."""
    if split != "final_holdout" or eval_size is not None or force:
        return
    base = _strip_mode(out_path)
    for cand in (base, base[:-len(".json")] + "_probs.json", base[:-len(".json")] + "_explain.json"):
        if os.path.abspath(cand) != os.path.abspath(out_path) and os.path.exists(cand):
            raise SystemExit(f"[eval_vllm] {cand} already measures final_holdout for {method} seed {seed} "
                             f"in another mode; final_holdout is measured once per model.")


def guard_final_holdout(out_path: str, split: str, eval_size=None, force: bool = False) -> None:
    """final_holdout (2022-2024 test) is measured ONCE per method. Refuse to overwrite an
    existing full-split result unless --force, so a re-run cannot silently replace the
    number the talk reports. Subsets (--eval-size) and selection_dev are never blocked."""
    if split != "final_holdout" or eval_size is not None or force:
        return
    if os.path.exists(out_path):
        raise SystemExit(f"[eval_vllm] {out_path} already exists: final_holdout is measured once "
                         f"per method. Pass --force only if that run is invalid, and say why.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True, choices=["base", "A_bf16", "B_int4_ptq", "C_int4_qat"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default=None, choices=["final_holdout", "selection_dev"],
                    help="default final_holdout in answer mode; other modes must name the split")
    ap.add_argument("--eval-size", type=int, default=None, help="stratified subset (dry run)")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--usd-per-hour", type=float, default=None, help="GPU VM price for cost/1k")
    ap.add_argument("--price", action="append", default=[],
                    help="label=USD/h, repeatable, e.g. --price spot=3.537 --price ondemand=6.98")
    ap.add_argument("--gpu-label", default=None, help="e.g. 'A100 80GB PCIe (NC24ads_A100_v4)'")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true",
                    help="allow re-measuring final_holdout when its result file already exists")
    ap.add_argument("--lora-dir", action="append", default=[],
                    help="selection_dev only: a PEFT adapter dir (e.g. <run>/checkpoint-213); repeatable. "
                         "The untuned base is loaded once and every adapter is scored on it.")
    ap.add_argument("--max-lora-rank", type=int, default=None, help="default: largest adapter r")
    ap.add_argument("--mode", default="answer", choices=["answer", "probs", "explain"],
                    help="answer (default, the talk's numbers): greedy text, first option number. "
                         "probs: one step, record P(option digit 1..5) and pick the largest (for "
                         "seed/adapter averaging). explain: write an explanation, then read the final "
                         "'정답: N' (method 3)")
    ap.add_argument("--explain-max-tokens", type=int, default=512)
    args = ap.parse_args(argv)
    if args.split is None:
        if args.mode != "answer":
            raise SystemExit("[eval_vllm] --mode probs/explain needs an explicit --split")
        args.split = "final_holdout"
    if args.lora_dir and args.split != "selection_dev":
        raise SystemExit("[eval_vllm] --lora-dir is for checkpoint selection on selection_dev only; "
                         "merge the chosen checkpoint and measure final_holdout once with --method A_bf16")
    if args.lora_dir and args.method != "base":
        raise SystemExit("[eval_vllm] --lora-dir applies adapters to the untuned base: use --method base")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import vllm

    from .data_korquad import load_config
    from . import v2_pipeline as V
    from . import kormedmcqa as K
    from .v2_run import artifact_dir, eval_filename

    prices = parse_prices(args.usd_per_hour, args.price)
    cfg = load_config(args.config)
    if args.eval_size is not None:
        cfg["data"]["eval_size"] = args.eval_size
    if V.task_name(cfg) != "kormedmcqa":
        raise SystemExit("eval_vllm.py scores KorMedMCQA; use v2_bench.py for KorQuAD serving")
    out = args.out or os.path.join(cfg["paths"]["results_dir"],
                                   eval_filename(args.method, args.seed, args.split, engine="vllm"))
    if args.eval_size is not None and not args.out:      # a subset never takes the full-run file name
        out = out[:-len(".json")] + f"_n{args.eval_size}.json"
    if args.mode != "answer" and not args.out:          # other decision rules never take the talk's file
        out = out[:-len(".json")] + f"_{args.mode}.json"
    guard_final_holdout(out, args.split, args.eval_size, args.force)
    guard_final_siblings(out, args.method, args.seed, args.split, args.eval_size, args.force)
    st = V.chat_settings(cfg)

    if args.method == "base":
        model_dir, rev, precision = cfg["base_model"]["selected"], cfg["base_model"].get("revision"), "bf16"
    else:
        model_dir, rev = artifact_dir(cfg, args.method, args.seed), None
        precision = "bf16" if args.method == "A_bf16" else "int4"
        assert os.path.isdir(model_dir), f"missing artifact: {model_dir}"

    data = V.load_slices(cfg, split=args.split)
    ex = data["eval"]
    tok = V.prepare_tokenizer(AutoTokenizer.from_pretrained(model_dir, revision=rev), st)
    if args.mode == "explain":
        prompts = [V.render_prompt(tok, K.explain_prompt_example(e), st, True) for e in ex]
    else:
        prompts = [V.render_prompt(tok, e, st, True) for e in ex]
    stop_ids = V.stop_token_ids(tok, st)
    digit_ids = []
    for d in range(1, 6):
        ids = tok(str(d), add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, f"option digit {d} is {len(ids)} tokens"
        digit_ids.append(int(ids[0]))

    llm_kwargs = dict(model=model_dir, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem_util,
                      max_model_len=args.max_model_len, enforce_eager=False,
                      generation_config="vllm")          # ignore the checkpoint's sampling defaults
    if rev:
        llm_kwargs["revision"] = rev
    if precision == "int4":
        llm_kwargs["quantization"] = "torchao"
    lora_reqs = []
    if args.lora_dir:
        from vllm.lora.request import LoRARequest

        ranks = []
        for i, d in enumerate(args.lora_dir, start=1):
            with open(os.path.join(d, "adapter_config.json"), encoding="utf-8") as fh:
                ranks.append(int(json.load(fh)["r"]))
            lora_reqs.append(LoRARequest(os.path.basename(os.path.normpath(d)), i, d))
        llm_kwargs.update(enable_lora=True, max_loras=1,
                          max_lora_rank=int(args.max_lora_rank or max(ranks)))
    t_load = time.time()
    llm = LLM(**llm_kwargs)
    load_s = time.time() - t_load

    if args.mode == "probs":
        sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20, skip_special_tokens=True)
    else:
        sp = SamplingParams(temperature=0.0, repetition_penalty=1.0, stop_token_ids=stop_ids,
                            max_tokens=int(args.explain_max_tokens if args.mode == "explain"
                                           else cfg["eval"]["max_new_tokens"]),
                            skip_special_tokens=True)

    def decode(outs):
        raw = [o.outputs[0].text for o in outs]
        probs = None
        if args.mode == "probs":
            probs = [option_probs(o.outputs[0].logprobs, digit_ids) for o in outs]
            preds = [argmax_choice(p) for p in probs]
        elif args.mode == "explain":
            # a cut-off explanation with no final '정답: N' line is unreadable, not a guess
            preds = [None if (o.outputs[0].finish_reason == "length" and K.final_answer_hit(r) is None)
                     else K.extract_final_choice(r) for o, r in zip(outs, raw)]
        else:
            preds = [K.extract_choice(r) for r in raw]
        sc = K.score(preds, ex, raw=raw)
        sc["n_truncated"] = sum(1 for o in outs if o.outputs[0].finish_reason == "length")
        if probs is not None:
            for item, p in zip(sc["items"], probs):
                item["probs"] = p
        gen_tok = [len(o.outputs[0].token_ids) for o in outs]
        return raw, sc, gen_tok

    if lora_reqs:                                         # one result file per adapter, same base
        for lr in lora_reqs:
            llm.generate(prompts[:8], sp, use_tqdm=False, lora_request=lr)
            t0 = time.time()
            outs = llm.generate(prompts, sp, use_tqdm=True, lora_request=lr)
            wall = time.time() - t0
            raw, sc, gen_tok = decode(outs)
            run = os.path.basename(os.path.dirname(os.path.normpath(lr.lora_path)))
            tag = (f"_n{args.eval_size}" if args.eval_size is not None else "") + \
                ("" if args.mode == "answer" else f"_{args.mode}")
            o_path = os.path.join(cfg["paths"]["results_dir"], "ckpt_dev",
                                  f"eval_vllm_lora_{run}_{lr.lora_name}_{args.split}{tag}.json")
            os.makedirs(os.path.dirname(o_path), exist_ok=True)
            with open(o_path, "w", encoding="utf-8") as fh:
                json.dump({"method": "A_bf16_lora_ckpt", "adapter": lr.lora_path, "engine": "vllm+lora",
                           "mode": args.mode, "vllm_version": getattr(vllm, "__version__", None),
                           "split": args.split, "accuracy": sc["accuracy"],
                           "accuracy_ci95": sc["accuracy_ci95"], "n_eval": sc["n"],
                           "n_unparsed": sc["n_unparsed"], "n_truncated": sc["n_truncated"],
                           "by_role": sc["by_role"],
                           "gen_tokens_mean": round(statistics.mean(gen_tok), 2) if gen_tok else None,
                           "load_seconds": round(load_s, 1), "wall_seconds": round(wall, 2),
                           "sample_predictions": [str(r)[:40] for r in raw[:3]],
                           "items": sc["items"]}, fh, ensure_ascii=False, indent=2)
            print(f"[eval_vllm] {lr.lora_path} ({args.mode}): dev acc={sc['accuracy']}% "
                  f"unparsed={sc['n_unparsed']} wall={wall:.1f}s -> {o_path}")
        return 0

    llm.generate(prompts[:8], sp, use_tqdm=False)        # warmup (CUDA graphs, KV pages)
    t0 = time.time()
    outs = llm.generate(prompts, sp, use_tqdm=True)
    wall = time.time() - t0

    raw, sc, gen_tok = decode(outs)
    prm_tok = [len(o.prompt_token_ids or []) for o in outs]
    n = sc["n"]
    res = {
        "method": args.method, "seed": int(args.seed), "base_model": cfg["base_model"]["selected"],
        "task": "kormedmcqa", "engine": "vllm", "vllm_version": getattr(vllm, "__version__", None),
        "split": args.split, "precision": precision, "model_dir": model_dir,
        "accuracy": sc["accuracy"], "accuracy_ci95": sc["accuracy_ci95"], "n_eval": n,
        "n_correct": sc["n_correct"], "n_unparsed": sc["n_unparsed"],
        "by_role": sc["by_role"], "by_year": sc["by_year"],
        "gpu_label": args.gpu_label, "load_seconds": round(load_s, 1),
        "wall_seconds": round(wall, 2), "questions_per_s": round(n / wall, 3) if wall > 0 else None,
        "prompt_tokens_mean": round(statistics.mean(prm_tok), 1) if prm_tok else None,
        "gen_tokens_mean": round(statistics.mean(gen_tok), 2) if gen_tok else None,
        "prices_usd_per_hour": prices,
        "usd_per_1k_questions": {k: cost_per_1k(v, wall, n) for k, v in prices.items()} or None,
        "template_kwargs": st["template_kwargs"], "stop_token_ids": stop_ids,
        "max_model_len": args.max_model_len, "sample_predictions": raw[:3], "items": sc["items"],
    }
    if args.mode != "answer":                           # answer-mode files keep their historical schema
        res["mode"] = args.mode
        res["n_truncated"] = sc["n_truncated"]
        res["sample_predictions"] = [str(r)[:40] for r in raw[:3]]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print(f"[eval_vllm] {args.method} seed={args.seed} split={args.split}: acc={res['accuracy']}% "
          f"ci95={res['accuracy_ci95']} n={n} unparsed={res['n_unparsed']} "
          f"wall={res['wall_seconds']}s q/s={res['questions_per_s']} "
          f"$/1k={res['usd_per_1k_questions']}")
    print("[eval_vllm] wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
