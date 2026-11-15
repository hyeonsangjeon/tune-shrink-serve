"""P1-3: ONLINE serving benchmark for the 3-way artifacts.

``v2_bench.py`` measures *offline* ``LLM.generate`` batch throughput. This module
measures a running **OpenAI-compatible HTTP endpoint** (e.g. vLLM's
``api_server``) under real concurrency with **HTTP streaming**, so the reported
TTFT/TPOT/ITL are true online latencies — not an offline ``max_tokens=1`` proxy.

It is transport-only: it talks to any OpenAI-compatible ``/v1/chat/completions``
endpoint, so the measurement path is fully testable on CPU against a fake SSE
server (no GPU, no vLLM). The real numbers only exist once you point it at a
served artifact; until then the README keeps the online table ``planned``.

Metrics per concurrency level: request throughput, output-token throughput,
TTFT / TPOT / ITL / end-to-end p50·p95·p99, goodput (fraction meeting an SLA)
and failure rate — over at least a few hundred requests when you scale it up.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Union


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pct(vals: Sequence[float], q: float) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return round(s[k], 4)


@dataclass
class RequestResult:
    ok: bool = False
    status: int = 0
    ttft_s: Optional[float] = None
    e2e_s: Optional[float] = None
    out_tokens: int = 0
    prompt_tokens: Optional[int] = None
    itl_s: List[float] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def tpot_s(self) -> Optional[float]:
        # mean inter-token latency after the first token
        return round(statistics.mean(self.itl_s), 6) if self.itl_s else None


# --------------------------------------------------------------------------- #
# one streamed request (SSE)
# --------------------------------------------------------------------------- #
def stream_one(base_url: str, model: str, prompt: Union[str, List[Dict[str, str]]], *,
               max_tokens: int = 128, timeout: float = 120.0, api_key: str = "EMPTY",
               extra_body: Optional[Dict[str, Any]] = None) -> RequestResult:
    """``prompt`` is a user string or a full ``messages`` list; ``extra_body`` is merged into
    the request (vLLM extras such as ``chat_template_kwargs``, ``stop_token_ids``, ``ignore_eos``)."""
    import httpx

    url = base_url.rstrip("/") + "/v1/chat/completions"
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    payload.update(extra_body or {})
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    res = RequestResult()
    start = time.perf_counter()
    last = start
    usage_out: Optional[int] = None
    try:
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", url, json=payload, headers=headers) as resp:
                res.status = resp.status_code
                if resp.status_code != 200:
                    resp.read()
                    res.error = f"http {resp.status_code}"
                    return res
                for line in resp.iter_lines():
                    if not line:
                        continue
                    line = line[5:].strip() if line.startswith("data:") else line.strip()
                    if line == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line)
                    except ValueError:
                        continue
                    usage = chunk.get("usage")
                    if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                        usage_out = int(usage["completion_tokens"])
                        if usage.get("prompt_tokens") is not None:
                            res.prompt_tokens = int(usage["prompt_tokens"])
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = (choices[0] or {}).get("delta") or {}
                    piece = delta.get("content")
                    if not piece:
                        continue
                    now = time.perf_counter()
                    if res.ttft_s is None:
                        res.ttft_s = round(now - start, 6)
                    else:
                        res.itl_s.append(now - last)
                    last = now
                    res.out_tokens += 1
        res.e2e_s = round(time.perf_counter() - start, 6)
        if usage_out is not None:
            res.out_tokens = usage_out
        res.ok = res.ttft_s is not None
        if not res.ok:
            res.error = "no content tokens received"
    except Exception as exc:  # network/timeout/parse
        res.error = f"{type(exc).__name__}: {exc}"
    return res


# --------------------------------------------------------------------------- #
# concurrency sweep
# --------------------------------------------------------------------------- #
def _aggregate(results: List[RequestResult], wall_s: float, *,
               sla_ttft_s: float, sla_e2e_s: float) -> Dict[str, Any]:
    ok = [r for r in results if r.ok]
    failed = len(results) - len(ok)
    ttft = [r.ttft_s for r in ok if r.ttft_s is not None]
    e2e = [r.e2e_s for r in ok if r.e2e_s is not None]
    tpot = [r.tpot_s for r in ok if r.tpot_s is not None]
    itl = [x for r in ok for x in r.itl_s]
    out_tokens = sum(r.out_tokens for r in ok)
    ptoks = [r.prompt_tokens for r in ok if r.prompt_tokens is not None]
    good = sum(1 for r in ok
               if r.ttft_s is not None and r.e2e_s is not None
               and r.ttft_s <= sla_ttft_s and r.e2e_s <= sla_e2e_s)
    n = len(results)
    return {
        "requests": n,
        "completed": len(ok),
        "failed": failed,
        "failure_rate": round(failed / n, 4) if n else "not_measured",
        "wall_s": round(wall_s, 4),
        "request_throughput_rps": round(len(ok) / wall_s, 4) if wall_s > 0 else "not_measured",
        "output_token_throughput_tps": round(out_tokens / wall_s, 4) if wall_s > 0 else "not_measured",
        "output_tokens_total": out_tokens,
        "prompt_tokens_mean": round(statistics.mean(ptoks), 1) if ptoks else "not_reported",
        "ttft_s": {"p50": pct(ttft, 0.50), "p95": pct(ttft, 0.95), "p99": pct(ttft, 0.99),
                   "mean": round(statistics.mean(ttft), 6) if ttft else None},
        "tpot_s": {"p50": pct(tpot, 0.50), "p95": pct(tpot, 0.95), "p99": pct(tpot, 0.99),
                   "mean": round(statistics.mean(tpot), 6) if tpot else None},
        "itl_s": {"p50": pct(itl, 0.50), "p95": pct(itl, 0.95), "p99": pct(itl, 0.99)},
        "e2e_s": {"p50": pct(e2e, 0.50), "p95": pct(e2e, 0.95), "p99": pct(e2e, 0.99),
                  "mean": round(statistics.mean(e2e), 6) if e2e else None},
        "goodput": {"sla_ttft_s": sla_ttft_s, "sla_e2e_s": sla_e2e_s,
                    "met": good, "rate": round(good / len(ok), 4) if ok else "not_measured"},
    }


def run_online_benchmark(base_url: str, model: str, prompts: Sequence[str], *,
                         concurrency_levels: Sequence[int] = (1, 8, 32),
                         num_prompts: int = 0, max_tokens: int = 128,
                         timeout: float = 120.0, api_key: str = "EMPTY",
                         sla_ttft_s: float = 1.0, sla_e2e_s: float = 10.0,
                         artifact_sha256: Optional[str] = None,
                         vllm_version: Optional[str] = None,
                         extra_body: Optional[Dict[str, Any]] = None,
                         prompt_source: str = "default_prompts",
                         fresh_per_level: bool = False,
                         shuffle_seed: Optional[int] = None) -> Dict[str, Any]:
    """Drive an online concurrency sweep against an OpenAI-compatible endpoint.

    ``fresh_per_level``: every concurrency level gets its own slice of ``prompts`` (no prompt is
    sent twice in the whole sweep), so a prefix cache only ever holds what the requests share
    (the system block), never a whole earlier prompt. Needs ``num_prompts`` x levels prompts.
    ``shuffle_seed`` shuffles the prompts first, so each slice has a similar mix of questions;
    run cache on and off with the same seed to send the same slices to both."""
    prompts = list(prompts)
    if not prompts:
        raise ValueError("no prompts")
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(prompts)
    if fresh_per_level:
        if num_prompts <= 0:
            raise ValueError("fresh_per_level needs an explicit num_prompts")
        need = num_prompts * len(concurrency_levels)
        if need > len(prompts):
            raise ValueError(f"fresh_per_level needs {need} prompts, only {len(prompts)} given")
    per_level: List[Dict[str, Any]] = []
    for k, c in enumerate(concurrency_levels):
        total = num_prompts or max(c * 4, len(prompts))
        if fresh_per_level:
            start = k * total
            batch = prompts[start:start + total]
        else:
            start = 0
            batch = [prompts[i % len(prompts)] for i in range(total)]
        results: List[RequestResult] = []
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=c) as pool:
            futs = [pool.submit(stream_one, base_url, model, p, max_tokens=max_tokens,
                                timeout=timeout, api_key=api_key, extra_body=extra_body)
                    for p in batch]
            for f in as_completed(futs):
                results.append(f.result())
        wall = time.perf_counter() - t0
        agg = _aggregate(results, wall, sla_ttft_s=sla_ttft_s, sla_e2e_s=sla_e2e_s)
        agg["concurrency"] = c
        agg["prompt_slice"] = [start, start + total] if fresh_per_level else "first_n_repeated"
        per_level.append(agg)
    return {
        "benchmark": "v2_online_serving",
        "transport": "openai_chat_completions_streaming",
        "generated_at": _utc_now(),
        "status": "live",
        "engine": {"base_url": base_url, "model": model,
                   "vllm_version": vllm_version or "not_recorded",
                   "artifact_sha256": artifact_sha256 or "not_recorded"},
        "config": {"concurrency_levels": list(concurrency_levels),
                   "max_tokens": max_tokens, "sla_ttft_s": sla_ttft_s, "sla_e2e_s": sla_e2e_s,
                   "prompt_source": prompt_source, "extra_body": extra_body or {},
                   "fresh_per_level": fresh_per_level, "prompt_shuffle_seed": shuffle_seed},
        "note": ("Online HTTP-streaming latencies. Distinct from offline v2_bench.py "
                 "(LLM.generate batch throughput; its max_tokens=1 row is a TTFT proxy, "
                 "not an online TTFT)."),
        "results": per_level,
    }


# --------------------------------------------------------------------------- #
# prompts + CLI
# --------------------------------------------------------------------------- #
def default_prompts(n: int = 8) -> List[str]:
    """Small self-contained Korean QA prompt set (no dataset download required)."""
    base = [
        "다음 문맥에서 질문에 답하세요.\n문맥: 광주는 대한민국 남서부의 광역시이다.\n질문: 광주는 어느 지역에 있는가?",
        "다음 문맥에서 질문에 답하세요.\n문맥: 세종대왕은 훈민정음을 창제하였다.\n질문: 훈민정음을 창제한 사람은?",
        "다음 문맥에서 질문에 답하세요.\n문맥: 2009년에 해당 특별법이 제정되었다.\n질문: 특별법이 제정된 연도는?",
    ]
    return [base[i % len(base)] for i in range(max(1, n))]


def task_messages(cfg_path: str, split: str = "selection_dev") -> tuple:
    """Chat messages exactly as eval_vllm renders them (system prompt + MCQ user turn), plus
    the request extras that keep the served prompt identical (fixed ``date_string``)."""
    from quantization.data_korquad import load_config
    from quantization import v2_pipeline as V

    cfg = load_config(cfg_path)
    st = V.chat_settings(cfg)
    ex = V.load_slices(cfg, split=split)["eval"]
    msgs = [[{"role": "system", "content": st["system"]},
             {"role": "user", "content": V.user_content_for(e, st)}] for e in ex]
    extra = {"chat_template_kwargs": dict(st["template_kwargs"])} if st["template_kwargs"] else {}
    return msgs, extra


def common_prefix_len(seqs: Sequence[Sequence[Any]]) -> int:
    """Length of the prefix every sequence shares (what a prefix cache can reuse across them)."""
    seqs = [list(s) for s in seqs]
    if not seqs:
        return 0
    n, p = min(len(s) for s in seqs), 0
    while p < n and all(s[p] == seqs[0][p] for s in seqs):
        p += 1
    return p


def shared_prefix_report(cfg_path: str, split: str, tokenizer_dir: str) -> Dict[str, Any]:
    """Token counts only (no text): how much of every served prompt is the same leading block."""
    from transformers import AutoTokenizer

    msgs, extra = task_messages(cfg_path, split)
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    kw = extra.get("chat_template_kwargs", {})
    ids = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=True, **kw) for m in msgs]
    ids = [i["input_ids"] if isinstance(i, dict) else i for i in ids]
    sys_ids = tok.apply_chat_template([msgs[0][0]], tokenize=True, **kw)
    sys_ids = sys_ids["input_ids"] if isinstance(sys_ids, dict) else sys_ids
    return {"benchmark": "shared_prompt_prefix", "generated_at": _utc_now(),
            "prompt_source": f"{cfg_path}:{split}", "tokenizer": tokenizer_dir,
            "n_prompts": len(ids), "prompt_tokens_mean": round(statistics.mean(map(len, ids)), 1),
            "shared_prefix_tokens": common_prefix_len(ids), "system_turn_tokens": len(sys_ids),
            "note": "Every served prompt starts with the same shared_prefix_tokens (KT system block, date, task "
                    "instruction, user header); with prefix caching on and fresh questions, this is what is reused."}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="quantization.serving.v2_bench_serve",
        description="Online serving benchmark against an OpenAI-compatible endpoint (P1-3).")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000", help="server base URL")
    ap.add_argument("--model", required=True, help="model name/path as served")
    ap.add_argument("--concurrency", default="1,8,32", help="comma list, e.g. 1,8,32")
    ap.add_argument("--num-prompts", type=int, default=0, help="requests per level (0=auto)")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--sla-ttft-s", type=float, default=1.0)
    ap.add_argument("--sla-e2e-s", type=float, default=10.0)
    ap.add_argument("--artifact-sha256", default=None, help="serving artifact hash for provenance")
    ap.add_argument("--output", required=True, help="raw results JSON (e.g. runs/<id>/serving/raw/online.json)")
    ap.add_argument("--prompts-config", default=None,
                    help="task config (e.g. quantization/configs/kormedmcqa_midm.yaml): send the "
                         "real eval prompts instead of the built-in toy prompts")
    ap.add_argument("--prompts-split", default="selection_dev")
    ap.add_argument("--stop-token-ids", default=None, help="comma list, e.g. 131301,2 (vLLM extra)")
    ap.add_argument("--ignore-eos", action="store_true", help="decode exactly --max-tokens (vLLM extra)")
    ap.add_argument("--fresh-per-level", action="store_true",
                    help="send different prompts at every concurrency level (no prompt sent twice)")
    ap.add_argument("--shuffle-seed", type=int, default=None,
                    help="shuffle the prompts first (same seed for cache on/off = same prompts)")
    ap.add_argument("--shared-prefix-report", action="store_true",
                    help="no requests: count the tokens every prompt shares (tokenizer = --model dir), write --output")
    args = ap.parse_args(argv)

    if args.shared_prefix_report:
        if not args.prompts_config:
            ap.error("--shared-prefix-report needs --prompts-config")
        rep = shared_prefix_report(args.prompts_config, args.prompts_split, args.model)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        print(f"[bench-serve] shared prefix {rep['shared_prefix_tokens']} of ~{rep['prompt_tokens_mean']} tokens")
        return 0

    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    extra: Dict[str, Any] = {}
    if args.prompts_config:
        prompts, extra = task_messages(args.prompts_config, args.prompts_split)
        source = f"{args.prompts_config}:{args.prompts_split}"
    else:
        prompts, source = default_prompts(max(levels) * 2), "default_prompts"
    if args.stop_token_ids:
        extra["stop_token_ids"] = [int(x) for x in args.stop_token_ids.split(",") if x.strip()]
    if args.ignore_eos:
        extra["ignore_eos"] = True

    vllm_version = None
    try:  # best-effort provenance from the live server
        import httpx
        r = httpx.get(args.base_url.rstrip("/") + "/version", timeout=5.0)
        if r.status_code == 200:
            vllm_version = r.json().get("version")
    except Exception:
        pass

    doc = run_online_benchmark(
        args.base_url, args.model, prompts, concurrency_levels=levels,
        num_prompts=args.num_prompts, max_tokens=args.max_tokens, timeout=args.timeout,
        sla_ttft_s=args.sla_ttft_s, sla_e2e_s=args.sla_e2e_s,
        artifact_sha256=args.artifact_sha256, vllm_version=vllm_version,
        extra_body=extra or None, prompt_source=source,
        fresh_per_level=args.fresh_per_level, shuffle_seed=args.shuffle_seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"[bench-serve] {args.output}")
    for lvl in doc["results"]:
        print(f"  c={lvl['concurrency']:>3}  rps={lvl['request_throughput_rps']}  "
              f"tok/s={lvl['output_token_throughput_tps']}  "
              f"ttft_p50={lvl['ttft_s']['p50']}  ttft_p99={lvl['ttft_s']['p99']}  "
              f"fail={lvl['failure_rate']}  goodput={lvl['goodput']['rate']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
