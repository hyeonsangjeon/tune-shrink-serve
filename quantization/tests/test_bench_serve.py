"""P1-3 tests — online serving benchmark measurement path against a fake SSE server.

No GPU / no vLLM: a tiny in-process HTTP server emits OpenAI-compatible streaming
chunks, so the TTFT/TPOT/throughput/goodput/percentile logic is fully exercised.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from quantization.serving import v2_bench_serve as bs
from quantization.serving import client as serving_client

_TOKENS = ["광주", "는", " 남서부", "에", " 있다", "."]


class _FakeOpenAIHandler(BaseHTTPRequestHandler):
    seen: list = []  # user turns received, in arrival order

    def log_message(self, *a):  # silence
        pass

    def do_GET(self):  # /health, /version
        if self.path == "/health":
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
        elif self.path == "/version":
            body = json.dumps({"version": "fake-0.0"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers(); self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            type(self).last_payload = json.loads(body or b"{}")
        except ValueError:
            type(self).last_payload = None
        if type(self).last_payload:
            msgs = type(self).last_payload.get("messages") or [{}]
            type(self).seen.append(msgs[-1].get("content"))
        if not self.path.endswith("/chat/completions"):
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def send(obj):
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()

        time.sleep(0.02)  # simulated prefill -> TTFT > 0
        for i, tok in enumerate(_TOKENS):
            send({"choices": [{"delta": {"content": tok}, "index": 0}]})
            time.sleep(0.005)  # inter-token latency
        send({"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
              "usage": {"completion_tokens": len(_TOKENS), "prompt_tokens": 7}})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


@pytest.fixture(scope="module")
def fake_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenAIHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    host, port = srv.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        srv.shutdown()


def test_stream_one_measures_ttft_and_tokens(fake_server):
    r = bs.stream_one(fake_server, "fake-model", "질문?", max_tokens=32)
    assert r.ok and r.status == 200
    assert r.ttft_s is not None and r.ttft_s > 0
    assert r.out_tokens == len(_TOKENS)          # from usage
    assert len(r.itl_s) == len(_TOKENS) - 1      # inter-token gaps
    assert r.e2e_s >= r.ttft_s
    assert r.tpot_s is not None and r.tpot_s > 0


def test_online_sweep_aggregates(fake_server):
    doc = bs.run_online_benchmark(
        fake_server, "fake-model", bs.default_prompts(4),
        concurrency_levels=[1, 4], num_prompts=8,
        sla_ttft_s=5.0, sla_e2e_s=10.0)
    assert doc["status"] == "live"
    assert [lvl["concurrency"] for lvl in doc["results"]] == [1, 4]
    for lvl in doc["results"]:
        assert lvl["requests"] == 8
        assert lvl["completed"] == 8 and lvl["failed"] == 0
        assert lvl["failure_rate"] == 0.0
        assert isinstance(lvl["request_throughput_rps"], (int, float)) and lvl["request_throughput_rps"] > 0
        assert lvl["output_token_throughput_tps"] > 0
        for p in ("p50", "p95", "p99"):
            assert lvl["ttft_s"][p] is not None and lvl["ttft_s"][p] > 0
            assert lvl["e2e_s"][p] is not None
        assert lvl["goodput"]["rate"] == 1.0     # generous SLA -> all good


def test_concurrency_speeds_up_wall_time(fake_server):
    """Higher concurrency should not serialize: 8-way wall < 8x single-request time."""
    doc = bs.run_online_benchmark(
        fake_server, "fake-model", bs.default_prompts(2),
        concurrency_levels=[8], num_prompts=8, sla_ttft_s=5.0, sla_e2e_s=10.0)
    lvl = doc["results"][0]
    per_req_e2e = lvl["e2e_s"]["p50"]
    assert lvl["wall_s"] < per_req_e2e * 8       # real parallelism


def test_failure_counted_on_bad_endpoint():
    r = bs.stream_one("http://127.0.0.1:9", "m", "q", timeout=1.0)  # nothing listening
    assert r.ok is False and r.error


def test_percentile_helper():
    assert bs.pct([], 0.5) is None
    assert bs.pct([1.0], 0.99) == 1.0
    assert bs.pct([1, 2, 3, 4], 0.5) in (2.0, 3.0)


def test_tight_sla_lowers_goodput(fake_server):
    # Impossible TTFT SLA -> goodput 0 even though all requests succeed.
    doc = bs.run_online_benchmark(
        fake_server, "fake-model", bs.default_prompts(2),
        concurrency_levels=[2], num_prompts=4,
        sla_ttft_s=1e-9, sla_e2e_s=1e-9)
    lvl = doc["results"][0]
    assert lvl["completed"] == 4
    assert lvl["goodput"]["rate"] == 0.0


def test_client_ask_streams(fake_server):
    ans = serving_client.ask(fake_server, "fake-model", "광주는?", context="광주는 남서부에 있다.")
    assert ans == "".join(_TOKENS)


def test_stream_one_sends_messages_and_vllm_extras(fake_server):
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}]
    extra = {"chat_template_kwargs": {"date_string": "21 Nov 2026"}, "stop_token_ids": [131301, 2],
             "ignore_eos": True}
    r = bs.stream_one(fake_server, "fake-model", msgs, max_tokens=16, extra_body=extra)
    assert r.ok
    sent = _FakeOpenAIHandler.last_payload
    assert sent["messages"] == msgs and sent["max_tokens"] == 16
    assert sent["chat_template_kwargs"] == {"date_string": "21 Nov 2026"}
    assert sent["stop_token_ids"] == [131301, 2] and sent["ignore_eos"] is True


def test_task_messages_match_eval_prompts(monkeypatch):
    import os
    from quantization import kormedmcqa as K
    from quantization import v2_pipeline as V

    ex = [K.MCQExample(id="nurse-2021-1-1", role="nurse", year=2021, question="Q?",
                       choices=["a", "b", "c", "d", "e"], gold=3)]
    monkeypatch.setattr(V, "load_slices", lambda cfg, split="final_holdout", **k: {"eval": ex})
    cfg = os.path.join(os.path.dirname(__file__), "..", "configs", "kormedmcqa_midm.yaml")
    msgs, extra = bs.task_messages(cfg, "selection_dev")
    st = V.chat_settings(V.load_config(cfg))
    assert msgs == [[{"role": "system", "content": st["system"]},
                     {"role": "user", "content": V.user_content_for(ex[0], st)}]]
    assert extra == {"chat_template_kwargs": {"date_string": "21 Nov 2026"}}


def test_fresh_per_level_never_resends_a_prompt(fake_server):
    prompts = [f"q{i}" for i in range(12)]
    _FakeOpenAIHandler.seen = []
    doc = bs.run_online_benchmark(fake_server, "fake-model", prompts, concurrency_levels=[1, 2, 4],
                                  num_prompts=4, fresh_per_level=True, shuffle_seed=0)
    seen = list(_FakeOpenAIHandler.seen)
    assert len(seen) == 12 and len(set(seen)) == 12          # every prompt exactly once
    assert [lvl["prompt_slice"] for lvl in doc["results"]] == [[0, 4], [4, 8], [8, 12]]
    assert doc["config"]["fresh_per_level"] is True and doc["config"]["prompt_shuffle_seed"] == 0
    assert all(lvl["prompt_tokens_mean"] == 7 for lvl in doc["results"])


def test_same_shuffle_seed_sends_same_slices(fake_server):
    prompts = [f"q{i}" for i in range(8)]
    runs = []
    for _ in range(2):   # e.g. prefix caching on, then off
        _FakeOpenAIHandler.seen = []
        bs.run_online_benchmark(fake_server, "fake-model", prompts, concurrency_levels=[1, 1],
                                num_prompts=4, fresh_per_level=True, shuffle_seed=3)
        runs.append(list(_FakeOpenAIHandler.seen))
    assert runs[0] == runs[1]                                  # concurrency 1: same order too
    assert sorted(runs[0]) == sorted(prompts)


def test_fresh_per_level_refuses_too_few_prompts(fake_server):
    with pytest.raises(ValueError, match="needs 12 prompts"):
        bs.run_online_benchmark(fake_server, "fake-model", ["a"] * 10, concurrency_levels=[1, 2, 4],
                                num_prompts=4, fresh_per_level=True)
    with pytest.raises(ValueError, match="explicit num_prompts"):
        bs.run_online_benchmark(fake_server, "fake-model", ["a"] * 10, concurrency_levels=[1],
                                fresh_per_level=True)


def test_default_still_repeats_first_prompts(fake_server):
    _FakeOpenAIHandler.seen = []
    doc = bs.run_online_benchmark(fake_server, "fake-model", ["x", "y", "z"], concurrency_levels=[1, 1],
                                  num_prompts=2)
    assert _FakeOpenAIHandler.seen == ["x", "y", "x", "y"]
    assert doc["results"][0]["prompt_slice"] == "first_n_repeated"


def test_common_prefix_len():
    assert bs.common_prefix_len([[1, 2, 3, 9], [1, 2, 3, 4, 5], [1, 2, 7]]) == 2
    assert bs.common_prefix_len([[1, 2], [1, 2]]) == 2
    assert bs.common_prefix_len([[5], [6]]) == 0
    assert bs.common_prefix_len([]) == 0
