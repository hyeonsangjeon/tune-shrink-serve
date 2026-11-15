"""v2 CLI orchestrator — run the multi-seed 3-way pipeline on the GPU VM.

Each subcommand runs in its OWN process (fresh CUDA + import state), so INT4/QAT
loads never collide with a training import. Per-seed artifacts:
  artifacts/A_bf16_seed<S>/  B_int4_ptq_seed<S>/  C_int4_qat_seed<S>/
Per-(method,seed) eval:      results/eval_<method>_seed<S>.json
Aggregated table:            results/three_way_table.json  (mean +/- std over seeds)

Usage:
  python -m quantization.v2_run a    --seed 42 [--resume]
  python -m quantization.v2_run b    --seed 42
  python -m quantization.v2_run c    --seed 42
  python -m quantization.v2_run eval --method A_bf16 --seed 42
  python -m quantization.v2_run agg
Smoke overrides (fast end-to-end code check): --subset --max-steps --eval-size --epochs
"""
from __future__ import annotations

import argparse
import json
import os

from .data_korquad import DEFAULT_CONFIG, load_config
from . import v2_pipeline as V


def artifact_dir(cfg, method: str, seed: int) -> str:
    base = {"A_bf16": cfg["paths"]["method_a_dir"],
            "B_int4_ptq": cfg["paths"]["method_b_dir"],
            "C_int4_qat": cfg["paths"]["method_c_dir"]}[method]
    return f"{base}_seed{seed}"


def _apply_overrides(cfg, args):
    if args.subset is not None:
        cfg["data"]["train_subset"] = args.subset
    if args.eval_size is not None:
        cfg["data"]["eval_size"] = args.eval_size
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
        cfg["qat"]["max_steps"] = args.max_steps
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    # QAT memory knobs (H100 NVL 94GB: 4x4; 80GB cards: 2x8 keeps the effective batch at 16)
    if getattr(args, "qat_batch", None) is not None:
        cfg["qat"]["per_device_batch_size"] = args.qat_batch
    if getattr(args, "qat_accum", None) is not None:
        cfg["qat"]["grad_accum"] = args.qat_accum
    if getattr(args, "backend", None):
        cfg["train"]["backend"] = args.backend
    return cfg


def _write(cfg, name: str, obj) -> str:
    d = cfg["paths"]["results_dir"]
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    print("[write]", p)
    return p


def _load_for_eval(model_dir: str, precision: str, cfg=None, revision=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, revision=revision)
    if cfg is not None:
        V.prepare_tokenizer(tok, V.chat_settings(cfg))
    elif tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if precision == "bf16" else None  # int4 dirs carry their own config
    kwargs = dict(device_map="cuda")
    if revision:
        kwargs["revision"] = revision
    if dtype is not None:
        kwargs["dtype"] = dtype
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    except TypeError:
        kwargs.pop("dtype", None)
        if precision == "bf16":
            kwargs["torch_dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    return model, tok


def cmd_a(cfg, args):
    out = artifact_dir(cfg, "A_bf16", args.seed)
    log = V.train_method_a(cfg, args.seed, out, resume=args.resume)
    log.update(V.merged_vocab_check(cfg, out))
    if log["vocab_ok"] is False:
        print(f"[A] WARNING merged vocab_size={log['vocab_size_merged']} != base "
              f"{log['vocab_size_expected']} (the trainer resized embeddings; B/C/vLLM differ from base)")
    _write(cfg, f"A_train_seed{args.seed}.json", log)
    print(f"[A] seed={args.seed} loss={log['train_loss']:.4f} steps={log['global_step']} "
          f"secs={log['train_seconds']} -> {out}")


def build_env():
    """torch / torchao / GPU that wrote an INT4 artifact (pickled int4 dirs are tied to it)."""
    env = {"torch": None, "torchao": None, "device": None}
    try:
        import torch

        env["torch"] = torch.__version__
        env["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    try:
        import torchao

        env["torchao"] = getattr(torchao, "__version__", None)
    except Exception:
        pass
    return env


def cmd_b(cfg, args):
    a = artifact_dir(cfg, "A_bf16", args.seed)
    b = artifact_dir(cfg, "B_int4_ptq", args.seed)
    assert os.path.isdir(a), f"missing A merge: {a}"
    log = V.build_method_b(cfg, a, b)
    log.update(build_env())
    _write(cfg, f"B_build_seed{args.seed}.json", log)
    print(f"[B] seed={args.seed} size={log['size_gb']}GB -> {b}")


def cmd_c(cfg, args):
    a = artifact_dir(cfg, "A_bf16", args.seed)
    c = artifact_dir(cfg, "C_int4_qat", args.seed)
    if args.export_only:            # rebuild C int4 from <C>_bf16adapt on THIS machine's stack
        log = V.export_method_c(cfg, c + "_bf16adapt", c)
        log.update({"seed": int(args.seed)})
        log.update(build_env())
        _write(cfg, f"C_export_seed{args.seed}.json", log)
        print(f"[C export] seed={args.seed} size={log['size_gb']}GB -> {c}")
        return
    assert os.path.isdir(a), f"missing A merge: {a}"
    log = V.train_method_c(cfg, a, c, args.seed, resume=args.resume)
    _write(cfg, f"C_train_seed{args.seed}.json", log)
    print(f"[C] seed={args.seed} loss={log['train_loss']:.4f} size={log['size_gb']}GB -> {c}")


def eval_filename(method: str, seed: int, split: str = "final_holdout", engine: str = "hf") -> str:
    prefix = "eval_vllm_" if engine == "vllm" else "eval_"
    suffix = "" if split == "final_holdout" else f"_{split}"
    return f"{prefix}{method}_seed{seed}{suffix}.json"


def summary_line(res) -> str:
    if "accuracy" in res and "exact_match" not in res:
        return (f"acc={res['accuracy']}% ci95={res.get('accuracy_ci95')} n={res['n_eval']} "
                f"unparsed={res.get('n_unparsed')} vram={res.get('peak_vram_gb')}GB "
                f"tok/s={res.get('tok_per_s')}")
    return (f"EM={res['exact_match']} F1={res['f1']} ppl={res['perplexity']} "
            f"vram={res['peak_vram_gb']}GB tok/s={res['tok_per_s']}")


def cmd_eval(cfg, args):
    split = args.split
    if args.method == "base":           # untuned base model (the "before tuning" baseline)
        prec, mdir, rev = "bf16", cfg["base_model"]["selected"], cfg["base_model"].get("revision")
    else:
        prec = {"A_bf16": "bf16", "B_int4_ptq": "int4", "C_int4_qat": "int4"}[args.method]
        mdir, rev = artifact_dir(cfg, args.method, args.seed), None
        assert os.path.isdir(mdir), f"missing artifact: {mdir}"
    data = V.load_slices(cfg, split=split)   # P1-1: final_holdout is eval-only
    model, tok = _load_for_eval(mdir, prec, cfg=cfg, revision=rev)
    res = V.eval_model_chat(cfg, model, tok, data["eval"], method=args.method,
                            seed=args.seed, model_dir=mdir if os.path.isdir(mdir) else None,
                            precision=prec, split=split)
    _write(cfg, eval_filename(args.method, args.seed, split), res)
    print(f"[eval] {args.method} seed={args.seed} split={split}: {summary_line(res)}")
    print("       samples:", res["sample_predictions"])


def cmd_merge(cfg, args):
    """Merge the 2021-selected adapter checkpoint of THIS config's run into A_bf16_seed<S>
    (then eval_vllm as usual). Refuses an adapter from another run, and an existing merge."""
    import re

    assert args.adapter and os.path.isdir(args.adapter), f"missing adapter dir: {args.adapter}"
    out = artifact_dir(cfg, "A_bf16", args.seed)
    run_dir = os.path.realpath(out + "_run") + os.sep
    if not os.path.realpath(args.adapter).startswith(run_dir):
        raise SystemExit(f"[merge] {args.adapter} is not under {run_dir}: wrong CONFIG or SEED for this adapter")
    if os.path.exists(os.path.join(out, "config.json")) and not args.force:
        raise SystemExit(f"[merge] {out} already holds a merged model; pass --force to replace it")
    m = re.search(r"checkpoint-(\d+)", args.adapter)
    step = m.group(1) if m else "x"
    log = V.merge_adapter_checkpoint(cfg, args.adapter, out)
    log.update(V.merged_vocab_check(cfg, out))
    log["step"] = int(step) if step.isdigit() else None
    _write(cfg, f"A_merge_seed{args.seed}_ckpt{step}.json", log)
    print(f"[merge] {args.adapter} -> {out} vocab_ok={log['vocab_ok']}")


def cmd_selftest(cfg, args):
    import sys
    res = V.qat_scheme_selftest(gs=int(cfg["qat"]["group_size"]))
    _write(cfg, "qat_selftest.json", res)
    print(f"[selftest] ok={res['ok']} prepare_fires={res['prepare_fires']} "
          f"fq_layers={res['fake_quant_layers']} fake_err={res['fake_err_vs_orig']} "
          f"serve_err={res['serve_err_vs_orig']} ratio={res['fake_to_serve_ratio']} "
          f"same_family={res['same_int4_family']} convert_ok={res['convert_roundtrip_ok']}")
    if not res["ok"]:
        sys.exit(1)


def collect_eval_rows(cfg, results_dir: str, engine: str = "hf"):
    """Final-holdout eval rows for the configured base model/seeds (and eval_size when set)."""
    expected_seeds = set(cfg.get("seeds", []))
    es = cfg["data"].get("eval_size")
    expected_n = int(es) if es else None
    base = cfg["base_model"]["selected"]
    prefix = "eval_vllm_" if engine == "vllm" else "eval_"
    rows, skipped = [], []
    for fn in sorted(os.listdir(results_dir)):
        if not (fn.startswith(prefix) and fn.endswith(".json")):
            continue
        if engine == "hf" and fn.startswith("eval_vllm_"):
            continue
        r = json.load(open(os.path.join(results_dir, fn), encoding="utf-8"))
        ok = (r.get("seed") in expected_seeds and r.get("base_model") == base
              and r.get("split", "final_holdout") == "final_holdout"
              and r.get("method") in ("A_bf16", "B_int4_ptq", "C_int4_qat")
              and (expected_n is None or r.get("n_eval") == expected_n))
        (rows if ok else skipped).append(r if ok else fn)
    if expected_n is None and len({r["n_eval"] for r in rows}) > 1:
        raise SystemExit(f"[agg] rows disagree on n_eval: {sorted({r['n_eval'] for r in rows})}")
    return rows, skipped, (expected_n if expected_n is not None else (rows[0]["n_eval"] if rows else None))


def cmd_agg(cfg, args):
    d = cfg["paths"]["results_dir"]
    base = cfg["base_model"]["selected"]
    rows, skipped, n_eval = collect_eval_rows(cfg, d, engine=args.engine)
    if skipped:
        print("[agg] skipped (seed/n_eval/base/split mismatch):", skipped)
    agg = V.aggregate_seeds(rows)
    out_name = "three_way_table.json" if args.engine == "hf" else "three_way_table_vllm.json"
    table = {"base_model": base, "eval_size": n_eval,
             "seeds": sorted(set(cfg.get("seeds", []))), "per_seed": rows, "aggregate": agg}
    if V.task_name(cfg) == "kormedmcqa":    # KorQuAD output stays byte-compatible with history
        table["engine"] = args.engine
        table["per_seed"] = [{k: v for k, v in r.items() if k != "items"} for r in rows]
    _write(cfg, out_name, table)
    for m in ("A_bf16", "B_int4_ptq", "C_int4_qat"):
        if m not in agg:
            continue
        a = agg[m]
        if "accuracy" in a:
            print(f"[agg] {m:12} acc={a['accuracy']['mean']}±{a['accuracy']['std']} "
                  f"n={a['n_eval']} seeds={a['seeds']}")
        else:
            print(f"[agg] {m:12} F1={a['f1']['mean']}±{a['f1']['std']} "
                  f"EM={a['exact_match']['mean']}±{a['exact_match']['std']} "
                  f"ppl={a['perplexity']['mean']}±{a['perplexity']['std']} seeds={a['seeds']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("a", "b", "c", "eval", "agg", "selftest", "merge"):
        p = sub.add_parser(name)
        p.add_argument("--config", default=DEFAULT_CONFIG)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--resume", action="store_true")
        p.add_argument("--method", default="A_bf16",
                       help="A_bf16 | B_int4_ptq | C_int4_qat | base (eval only: untuned model)")
        p.add_argument("--split", default="final_holdout",
                       choices=["final_holdout", "selection_dev"])
        p.add_argument("--engine", default="hf", choices=["hf", "vllm"],
                       help="agg only: which eval files to aggregate")
        p.add_argument("--qat-batch", type=int, default=None, help="c only: per-device batch")
        p.add_argument("--qat-accum", type=int, default=None, help="c only: grad accumulation")
        p.add_argument("--backend", default=None, choices=["hf", "unsloth"],
                       help="a only: override train.backend")
        p.add_argument("--export-only", action="store_true",
                       help="c only: skip training; INT4-export <C>_bf16adapt with this machine's stack")
        p.add_argument("--adapter", default=None, help="merge only: <run>/checkpoint-N to merge")
        p.add_argument("--force", action="store_true", help="merge only: replace an existing merged model")
        p.add_argument("--subset", type=int, default=None)
        p.add_argument("--eval-size", type=int, default=None)
        p.add_argument("--max-steps", type=int, default=None)
        p.add_argument("--epochs", type=float, default=None)
    args = ap.parse_args()
    cfg = _apply_overrides(load_config(args.config), args)
    {"a": cmd_a, "b": cmd_b, "c": cmd_c, "eval": cmd_eval, "agg": cmd_agg,
     "selftest": cmd_selftest, "merge": cmd_merge}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
