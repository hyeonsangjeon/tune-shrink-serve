"""Managed-API accuracy and token cost for KorMedMCQA (agenda ④: self-hosted GPU vs managed).

Sends the same questions, with the same system and user text the self-hosted eval uses
(``eval_vllm.py``), to a model deployment behind an OpenAI-compatible Chat Completions
endpoint (for example Azure AI Foundry ``/openai/v1``). Answers are scored with the same
extractor, so a result pairs with ``eval_vllm_*.json`` by question id for McNemar.

Cost here is COMPUTED: the usage tokens the endpoint reports x the list price passed on the
command line (USD per 1M tokens, with its source and date). It is not an invoice.

Auth is Microsoft Entra ID through the az CLI (``az account get-access-token``). No API key is
read, stored, or printed, and the endpoint address comes from the environment, never the repo:

  export MANAGED_BASE_URL=https://<resource>.services.ai.azure.com/openai/v1
  export MANAGED_AZ_SUBSCRIPTION=<subscription name or id>     # optional: picks the tenant
  # pilot on 2021 questions (never touches the 2022-2024 result files)
  python -m quantization.eval_managed --config quantization/configs/kormedmcqa_midm.yaml \
      --deployment gpt-5.4-nano --split selection_dev --eval-size 30
  # the measured run, 2022-2024, once per deployment (prices in USD per 1M tokens, with source and date)
  python -m quantization.eval_managed --config ... --deployment <deployment> --concurrency 32 \
      --region <region> --price-in <in> --price-out <out> --price-cached-in <cached> \
      --price-source "Azure Retail Prices API (...), read <date>"

Output: <results_dir>/managed/eval_managed_<deployment>[_<split>][_n<size>].json
The network code is standard library (plus the az CLI for the token). Loading the questions uses the
repo loader, which needs pyyaml and datasets; datasets 4.3.0 fails on Python 3.14, so use 3.12.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence

TOKEN_RESOURCE = "https://cognitiveservices.azure.com"
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
COST_NOTE = "computed: reported usage tokens x list price per 1M tokens; not an invoice"


# --------------------------------------------------------------------------- #
# Pure helpers (tested)
# --------------------------------------------------------------------------- #
def build_body(deployment: str, messages: List[Dict[str, str]], max_completion_tokens: int,
               reasoning_effort: Optional[str] = None, temperature: Optional[float] = None) -> Dict[str, Any]:
    """Chat Completions body. ``temperature`` is sent only when asked: reasoning models reject it."""
    body: Dict[str, Any] = {"model": deployment, "messages": messages,
                            "max_completion_tokens": int(max_completion_tokens)}
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    if temperature is not None:
        body["temperature"] = float(temperature)
    return body


def usage_of(resp: Dict[str, Any]) -> Dict[str, Any]:
    """Token usage as reported. ``reasoning`` is None when the endpoint does not report it separately
    (some models put reasoning inside completion_tokens without a breakdown)."""
    u = resp.get("usage") or {}
    pd = u.get("prompt_tokens_details") or {}
    cd = u.get("completion_tokens_details") or {}
    rt = cd.get("reasoning_tokens") if isinstance(cd, dict) else None
    return {"prompt": int(u.get("prompt_tokens") or 0), "completion": int(u.get("completion_tokens") or 0),
            "reasoning": None if rt is None else int(rt), "cached": int(pd.get("cached_tokens") or 0)}


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    v = sorted(values)
    if not v:
        return None
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return round(v[lo] + (v[hi] - v[lo]) * (k - lo), 3)


def token_totals(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Totals and per-question means. Blocked requests count as 0 tokens. ``reasoning_*`` is None when no
    response reported reasoning tokens separately."""
    n = len(records) or 1
    out: Dict[str, Any] = {}
    for k in ("prompt", "completion", "reasoning", "cached"):
        vals = [r.get("usage", {}).get(k) for r in records]
        if k == "reasoning" and all(v is None for v in vals):
            out["reasoning_total"], out["reasoning_mean"] = None, None
            continue
        tot = sum(int(v or 0) for v in vals)
        out[f"{k}_total"], out[f"{k}_mean"] = tot, round(tot / n, 2)
    return {**{k: v for k, v in out.items() if k.endswith("_total")},
            **{k: v for k, v in out.items() if k.endswith("_mean")}}


def cost_per_1k(totals: Dict[str, Any], n: int, price_in: Optional[float], price_out: Optional[float],
                price_cached_in: Optional[float] = None) -> Optional[float]:
    """USD per 1,000 questions from token totals and USD-per-1M prices. Cached prompt tokens use
    ``price_cached_in`` when given, otherwise the normal input price (the conservative choice)."""
    if price_in is None or price_out is None or n <= 0:
        return None
    cached = totals["cached_total"]
    cin = price_in if price_cached_in is None else price_cached_in
    usd = ((totals["prompt_total"] - cached) * price_in + cached * cin
           + totals["completion_total"] * price_out) / 1e6
    return round(usd / n * 1000.0, 4)


def error_reason(body: str) -> str:
    """Short reason from an error body: ``error.code`` (e.g. content_safety_violation), else a
    choice's ``finish_reason`` (e.g. content_filter), else the first 120 characters."""
    try:
        d = json.loads(body)
    except ValueError:
        return body[:120]
    if not isinstance(d, dict):
        return body[:120]
    if isinstance(d.get("error"), dict) and d["error"].get("code"):
        return str(d["error"]["code"])
    for ch in d.get("choices") or []:
        if ch.get("finish_reason"):
            return f"finish_reason={ch['finish_reason']}"
    return body[:120]


def safe_name(deployment: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", deployment)


def out_filename(deployment: str, split: str, eval_size: Optional[int]) -> str:
    name = f"eval_managed_{safe_name(deployment)}"
    if split != "final_holdout":
        name += f"_{split}"
    if eval_size is not None:
        name += f"_n{eval_size}"
    return name + ".json"


def run_all(messages: Sequence[List[Dict[str, str]]], send: Callable[[List[Dict[str, str]]], Dict[str, Any]],
            concurrency: int) -> List[Dict[str, Any]]:
    """Send every message list through ``send`` (returns one record) with a thread pool; keep order."""
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        return list(pool.map(send, messages))


# --------------------------------------------------------------------------- #
# Network (Entra ID token via az CLI, urllib)
# --------------------------------------------------------------------------- #
class AzToken:
    """Bearer token from ``az account get-access-token``, refreshed 5 minutes before expiry.
    The token stays in memory; it is never logged or written."""

    def __init__(self, subscription: Optional[str] = None, resource: str = TOKEN_RESOURCE):
        self.subscription, self.resource = subscription, resource
        self._tok: Optional[str] = None
        self._exp = 0.0
        self._lock = threading.Lock()

    def get(self) -> str:
        with self._lock:
            if self._tok is None or time.time() > self._exp - 300:
                cmd = ["az", "account", "get-access-token", "--resource", self.resource,
                       "--query", "{t:accessToken,e:expires_on}", "-o", "json"]
                if self.subscription:
                    cmd += ["--subscription", self.subscription]
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if p.returncode:
                    raise SystemExit("[eval_managed] az could not get a token (run `az login`): "
                                     + p.stderr.strip()[:300])
                d = json.loads(p.stdout)
                self._tok, self._exp = d["t"], float(d["e"])
            return self._tok


def retry_wait(headers: Any, attempt: int, cap: float = 60.0) -> float:
    """Seconds to wait before the next try: ``retry-after-ms`` / ``retry-after`` when they parse, else
    exponential backoff; never more than ``cap``."""
    for key, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        try:
            v = float((headers or {}).get(key) or 0) * scale
        except (TypeError, ValueError):
            v = 0.0
        if v > 0:
            return min(cap, v)
    return min(cap, 2.0 ** attempt)


def make_sender(base_url: str, token: AzToken, body_for: Callable[[List[Dict[str, str]]], Dict[str, Any]],
                timeout: float = 120.0, max_tries: int = 8) -> Callable[[List[Dict[str, str]]], Dict[str, Any]]:
    """One request with retries. Any failure becomes an error record, never an exception, so one bad
    request cannot abort a run whose other answers are already paid for."""
    url = base_url.rstrip("/") + "/chat/completions"

    def send(messages: List[Dict[str, str]]) -> Dict[str, Any]:
        data = json.dumps(body_for(messages)).encode("utf-8")
        err, tries = None, 0
        for attempt in range(max_tries):
            tries = attempt + 1
            t0 = time.time()
            retry, wait = False, 0.0
            try:
                req = urllib.request.Request(url, data=data, headers={
                    "Authorization": "Bearer " + token.get(), "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    resp = json.load(r)
                ch = (resp.get("choices") or [{}])[0]
                return {"text": (ch.get("message") or {}).get("content") or "",
                        "finish_reason": ch.get("finish_reason"), "model": resp.get("model"),
                        "usage": usage_of(resp), "latency_s": round(time.time() - t0, 3),
                        "tries": tries, "error": None}
            except urllib.error.HTTPError as e:
                try:
                    body = e.read().decode("utf-8", "replace")
                except Exception:  # noqa: BLE001
                    body = ""
                err = f"HTTP {e.code}: {error_reason(body)}"
                retry, wait = e.code in RETRY_STATUS, retry_wait(e.headers, attempt)
            except SystemExit:
                raise                                   # no az token: stop the run, nothing to record
            except Exception as e:  # noqa: BLE001  URLError, timeouts, broken JSON, http.client errors
                err = f"{type(e).__name__}: {e}"[:300]
                retry, wait = True, retry_wait(None, attempt)
            if not retry or attempt == max_tries - 1:
                break
            time.sleep(wait)
        return {"text": "", "finish_reason": None, "model": None,
                "usage": {"prompt": 0, "completion": 0, "reasoning": None, "cached": 0},
                "latency_s": None, "tries": tries, "error": err}

    return send


# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--deployment", required=True, help="deployment name sent as `model`")
    ap.add_argument("--split", default="final_holdout", choices=["final_holdout", "selection_dev"])
    ap.add_argument("--eval-size", type=int, default=None, help="stratified subset (pilot)")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-completion-tokens", type=int, default=2048,
                    help="reasoning tokens count against this limit; record, don't starve")
    ap.add_argument("--reasoning-effort", default=None, help="e.g. none, minimal, low (model dependent)")
    ap.add_argument("--temperature", type=float, default=None, help="only for models that accept it")
    ap.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens")
    ap.add_argument("--price-cached-in", type=float, default=None, help="USD per 1M cached input tokens")
    ap.add_argument("--price-out", type=float, default=None, help="USD per 1M output tokens")
    ap.add_argument("--price-source", default=None, help="where and when the prices were read")
    ap.add_argument("--provider", default="Azure AI Foundry")
    ap.add_argument("--deployment-type", default="GlobalStandard")
    ap.add_argument("--region", default=None, help="label only, e.g. eastus2")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true",
                    help="allow re-measuring final_holdout when its result file already exists")
    args = ap.parse_args(argv)

    from .data_korquad import load_config
    from . import v2_pipeline as V
    from . import kormedmcqa as K
    from .eval_vllm import guard_final_holdout

    base_url = os.environ.get("MANAGED_BASE_URL")
    if not base_url:
        raise SystemExit("[eval_managed] set MANAGED_BASE_URL (OpenAI-compatible /v1 base URL)")
    cfg = load_config(args.config)
    if V.task_name(cfg) != "kormedmcqa":
        raise SystemExit("eval_managed.py scores KorMedMCQA")
    if args.eval_size is not None:
        cfg["data"]["eval_size"] = args.eval_size
    out = args.out or os.path.join(cfg["paths"]["results_dir"], "managed",
                                   out_filename(args.deployment, args.split, args.eval_size))
    guard_final_holdout(out, args.split, args.eval_size, args.force)

    st = V.chat_settings(cfg)
    ex = V.load_slices(cfg, split=args.split)["eval"]
    msgs = [[{"role": "system", "content": st["system"]},
             {"role": "user", "content": V.user_content_for(e, st)}] for e in ex]

    token = AzToken(os.environ.get("MANAGED_AZ_SUBSCRIPTION"))
    send = make_sender(base_url, token, lambda m: build_body(
        args.deployment, m, args.max_completion_tokens, args.reasoning_effort, args.temperature))
    started = dt.datetime.now(dt.timezone.utc)
    t0 = time.time()
    recs = run_all(msgs, send, args.concurrency)
    wall = time.time() - t0

    raw = [r["text"] for r in recs]
    preds = [K.extract_choice(t) if not r["error"] else None for t, r in zip(raw, recs)]
    sc = K.score(preds, ex, raw=raw)
    for item, r in zip(sc["items"], recs):
        item.update(usage=r["usage"], latency_s=r["latency_s"], finish_reason=r["finish_reason"],
                    error=(r["error"] or None) and r["error"][:120])
    n = sc["n"]
    tot = token_totals(recs)
    lat = [r["latency_s"] for r in recs if r["latency_s"] is not None]
    finish: Dict[str, int] = {}
    for r in recs:
        finish[str(r["finish_reason"])] = finish.get(str(r["finish_reason"]), 0) + 1
    res = {
        "method": f"managed:{args.deployment}", "engine": "managed_api", "task": "kormedmcqa",
        "provider": args.provider, "deployment": args.deployment, "deployment_type": args.deployment_type,
        "region": args.region, "models_reported": sorted({r["model"] for r in recs if r["model"]}),
        "split": args.split, "measured_at_utc": started.isoformat(timespec="seconds"),
        "accuracy": sc["accuracy"], "accuracy_ci95": sc["accuracy_ci95"], "n_eval": n,
        "n_correct": sc["n_correct"], "n_unparsed": sc["n_unparsed"],
        "n_errors": sum(1 for r in recs if r["error"]), "finish_reasons": finish,
        "by_role": sc["by_role"], "by_year": sc["by_year"],
        "request": {"max_completion_tokens": args.max_completion_tokens,
                    "reasoning_effort": args.reasoning_effort, "temperature": args.temperature,
                    "system": st["system"], "user_template": st.get("user_template")},
        "concurrency": args.concurrency, "wall_seconds": round(wall, 2),
        "questions_per_s": round(n / wall, 3) if wall > 0 else None,
        "latency_s": {"p50": percentile(lat, 0.5), "p95": percentile(lat, 0.95), "max": max(lat) if lat else None},
        "tokens": tot, "token_source": "usage field returned by the endpoint",
        "prices_usd_per_1m": {"input": args.price_in, "cached_input": args.price_cached_in,
                              "output": args.price_out},
        "price_source": args.price_source,
        "usd_per_1k_questions": cost_per_1k(tot, n, args.price_in, args.price_out, args.price_cached_in),
        "cost_note": COST_NOTE, "sample_predictions": [str(t)[:40] for t in raw[:3]], "items": sc["items"],
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print(f"[eval_managed] {args.deployment} {args.split} n={n}: acc={res['accuracy']}% "
          f"ci95={res['accuracy_ci95']} unparsed={res['n_unparsed']} errors={res['n_errors']} "
          f"prompt/q={tot['prompt_mean']} out/q={tot['completion_mean']} (reasoning {tot['reasoning_mean']}) "
          f"p50={res['latency_s']['p50']}s wall={res['wall_seconds']}s $/1k={res['usd_per_1k_questions']}")
    print("[eval_managed] wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
