"""CPU-only checks for quantization/eval_managed.py (no network, no az)."""
from quantization import eval_managed as M
from quantization import kormedmcqa as K


def test_body_sends_temperature_and_effort_only_when_asked():
    msgs = [{"role": "user", "content": "q"}]
    b = M.build_body("dep", msgs, 64)
    assert b == {"model": "dep", "messages": msgs, "max_completion_tokens": 64}
    b = M.build_body("dep", msgs, 64, reasoning_effort="minimal", temperature=0)
    assert b["reasoning_effort"] == "minimal" and b["temperature"] == 0.0


def test_usage_reads_reasoning_and_cached_tokens():
    resp = {"usage": {"prompt_tokens": 300, "completion_tokens": 40,
                      "prompt_tokens_details": {"cached_tokens": 128},
                      "completion_tokens_details": {"reasoning_tokens": 38}}}
    assert M.usage_of(resp) == {"prompt": 300, "completion": 40, "reasoning": 38, "cached": 128}
    # no breakdown reported: reasoning is unknown, not zero
    assert M.usage_of({}) == {"prompt": 0, "completion": 0, "reasoning": None, "cached": 0}
    tot = M.token_totals([{"usage": M.usage_of({"usage": {"prompt_tokens": 5, "completion_tokens": 9}})}])
    assert tot["reasoning_total"] is None and tot["completion_total"] == 9


def test_cost_per_1k_uses_cached_price_only_when_given():
    recs = [{"usage": {"prompt": 1000, "completion": 10, "reasoning": 0, "cached": 400}}] * 2
    tot = M.token_totals(recs)
    assert tot["prompt_total"] == 2000 and tot["completion_mean"] == 10
    # no cached price: all 2,000 prompt tokens at $1/1M + 20 output tokens at $10/1M
    assert M.cost_per_1k(tot, 2, 1.0, 10.0) == round((2000 * 1.0 + 20 * 10.0) / 1e6 / 2 * 1000, 4)
    # cached price: 800 of them at $0.1/1M
    assert M.cost_per_1k(tot, 2, 1.0, 10.0, 0.1) == round((1200 + 80 + 200) / 1e6 / 2 * 1000, 4)
    assert M.cost_per_1k(tot, 2, None, 10.0) is None


def test_file_names_never_collide_with_the_final_result():
    assert M.out_filename("gpt-5.4-nano", "final_holdout", None) == "eval_managed_gpt-5.4-nano.json"
    assert M.out_filename("gpt-5.4-nano", "selection_dev", 30) == "eval_managed_gpt-5.4-nano_selection_dev_n30.json"
    assert M.out_filename("a/b c", "final_holdout", 5) == "eval_managed_a-b-c_n5.json"


def test_run_all_keeps_order_and_errors_score_as_wrong():
    answers = {"q1": "3", "q2": "정답은 ②입니다", "q3": None}

    def send(m):
        a = answers[m[0]["content"]]
        if a is None:
            return {"text": "", "finish_reason": None, "model": None, "latency_s": None,
                    "usage": {"prompt": 0, "completion": 0, "reasoning": 0, "cached": 0}, "error": "HTTP 400"}
        return {"text": a, "finish_reason": "stop", "model": "m", "latency_s": 0.1,
                "usage": {"prompt": 10, "completion": 1, "reasoning": 0, "cached": 0}, "error": None}

    recs = M.run_all([[{"role": "user", "content": k}] for k in ("q1", "q2", "q3")], send, concurrency=3)
    assert [r["text"] for r in recs] == ["3", "정답은 ②입니다", ""]
    preds = [K.extract_choice(r["text"]) if not r["error"] else None for r in recs]
    assert preds == [3, 2, None]


def test_error_reason_is_short():
    assert M.error_reason('{"error":{"code":"content_safety_violation","message":"blocked"}}') == \
        "content_safety_violation"
    assert M.error_reason('{"choices":[{"index":0,"finish_reason":"content_filter"}]}') == \
        "finish_reason=content_filter"
    assert M.error_reason("not json " * 40) == ("not json " * 40)[:120]


def test_percentile():
    assert M.percentile([], 0.5) is None
    assert M.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5


def test_retry_wait_parses_headers_and_caps():
    assert M.retry_wait({"retry-after": "3"}, 0) == 3.0
    assert M.retry_wait({"retry-after-ms": "1500"}, 0) == 1.5
    assert M.retry_wait({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}, 2) == 4.0   # unparsable -> backoff
    assert M.retry_wait({"retry-after": "900"}, 0) == 60.0
    assert M.retry_wait(None, 10) == 60.0


def test_send_turns_any_failure_into_an_error_record(monkeypatch):
    class Tok:
        def get(self):
            return "t"

    def boom(*a, **k):
        raise ValueError("bad json")

    monkeypatch.setattr(M.urllib.request, "urlopen", boom)
    monkeypatch.setattr(M.time, "sleep", lambda s: None)
    send = M.make_sender("https://example.invalid/v1", Tok(), lambda m: {"model": "x"}, max_tries=3)
    rec = send([{"role": "user", "content": "q"}])
    assert rec["error"].startswith("ValueError") and rec["tries"] == 3 and rec["text"] == ""
