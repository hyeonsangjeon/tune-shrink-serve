"""CPU tests for the task-agnostic chat settings and the KorMedMCQA wiring in
v2_pipeline / v2_run (fake tokenizer; no torch / transformers / network)."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from quantization import kormedmcqa as K  # noqa: E402
from quantization import v2_pipeline as V  # noqa: E402
from quantization import v2_run as R  # noqa: E402
from quantization.data_korquad import QAExample, load_config  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MIDM_CFG = os.path.join(HERE, "..", "configs", "kormedmcqa_midm.yaml")
KORQUAD_CFG = os.path.join(HERE, "..", "..", "archive", "qwen3_korquad", "config.yaml")


class FakeTok:
    """Records apply_chat_template kwargs; vocab knows a few Mi:dm specials."""
    vocab = {"<|end_of_text|>": 2, "<|eot_id|>": 131301, "<|finetune_right_pad_id|>": 131363,
             "<unk>": 3}
    unk_token, unk_token_id = "<unk>", 3
    eos_token, eos_token_id = "<|end_of_text|>", 2

    def __init__(self, accepts_thinking=True):
        self.pad_token = None
        self.calls = []
        self.accepts_thinking = accepts_thinking

    def convert_tokens_to_ids(self, t):
        return self.vocab.get(t, self.unk_token_id)

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        if "enable_thinking" in kw and not self.accepts_thinking:
            raise TypeError("unexpected kwarg enable_thinking")
        self.calls.append({"msgs": msgs, "kw": kw, "agp": add_generation_prompt})
        return "|".join(f"{m['role']}:{m['content']}" for m in msgs) + ("|assistant:" if add_generation_prompt else "")


def _mcq():
    return K.MCQExample(id="doctor-2022-1-1", role="doctor", year=2022, question="Q?",
                        choices=["a", "b", "c", "d", "e"], gold=2)


def test_midm_config_loads_and_routes_to_kormedmcqa():
    cfg = load_config(MIDM_CFG)
    assert V.task_name(cfg) == "kormedmcqa"
    st = V.chat_settings(cfg)
    assert st["template_kwargs"] == {"date_string": "21 Nov 2026"}
    assert st["answer_end_token"] == "<|eot_id|>" and st["pad_token"] == "<|finetune_right_pad_id|>"
    assert "{question}" in st["user_template"] and "{choices}" in st["user_template"]
    assert cfg["paths"]["results_dir"].endswith("kormedmcqa_midm")      # never the historical dir
    assert R.artifact_dir(cfg, "C_int4_qat", 42).endswith("kormedmcqa_midm/C_int4_qat_seed42")


def test_korquad_config_keeps_legacy_defaults():
    cfg = load_config(KORQUAD_CFG)
    st = V.chat_settings(cfg)
    assert V.task_name(cfg) == "korquad"
    assert st["template_kwargs"] == {} and st["answer_end_token"] is None and st["pad_token"] is None


def test_render_prompt_passes_fixed_date_and_mcq_user_content():
    cfg = load_config(MIDM_CFG)
    st = V.chat_settings(cfg)
    tok = FakeTok()
    out = V.render_prompt(tok, _mcq(), st)
    call = tok.calls[-1]
    assert call["kw"]["date_string"] == "21 Nov 2026"
    assert call["msgs"][0]["role"] == "system" and "번호 하나만" in call["msgs"][0]["content"]
    assert "의사 국가시험" in call["msgs"][1]["content"] and "2. b" in call["msgs"][1]["content"]
    assert out.endswith("|assistant:")


def test_render_prompt_matches_legacy_korquad_builder():
    tok = FakeTok()
    ex = QAExample(id="1", prompt="", answer="세종", answers=["세종"], context="문맥", question="질문")
    st = V._default_settings("SYS", False)
    assert V.render_prompt(tok, ex, st) == V.build_chat_prompt(tok, "SYS", "문맥", "질문", None, False, True)


def test_render_prompt_falls_back_without_enable_thinking():
    tok = FakeTok(accepts_thinking=False)
    st = V.chat_settings(load_config(MIDM_CFG))
    V.render_prompt(tok, _mcq(), st)
    assert "enable_thinking" not in tok.calls[-1]["kw"] and tok.calls[-1]["kw"]["date_string"]


def test_token_resolution_pad_answer_end_and_stops():
    st = V.chat_settings(load_config(MIDM_CFG))
    tok = V.prepare_tokenizer(FakeTok(), st)
    assert tok.pad_token == "<|finetune_right_pad_id|>"
    assert V.answer_end_id(tok, st) == 131301
    assert V.stop_token_ids(tok, st) == [131301, 2]
    assert V.resolve_token_id(tok, "<|not_a_token|>") is None


def test_token_resolution_falls_back_when_vocab_lacks_specials():
    class QwenLike(FakeTok):
        vocab = {"<|im_end|>": 151645, "<unk>": 3}
        eos_token, eos_token_id = "<|im_end|>", 151645

    st = V.chat_settings(load_config(MIDM_CFG))
    tok = V.prepare_tokenizer(QwenLike(), st)
    assert tok.pad_token == "<|im_end|>"                     # legacy pad = eos
    assert V.answer_end_id(tok, st) == 151645
    assert V.stop_token_ids(tok, st) == [151645]


def test_tokenize_completion_only_labels_only_answer_plus_eot():
    pytest.importorskip("datasets")

    class TT(FakeTok):
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) for c in text]}

    st = V.chat_settings(load_config(MIDM_CFG))
    tok = V.prepare_tokenizer(TT(), st)
    ds = V.tokenize_completion_only(tok, [_mcq()], st["system"], 4096, False, settings=st)
    ids, labels = ds[0]["input_ids"], ds[0]["labels"]
    assert ids[-2:] == [ord("2"), 131301]
    assert labels[-2:] == [ord("2"), 131301] and all(l == -100 for l in labels[:-2])


def test_aggregate_seeds_handles_accuracy_rows():
    rows = [{"method": "A_bf16", "seed": 42, "base_model": "m", "n_eval": 3009, "accuracy": 60.0,
             "n_unparsed": 1, "size_gb": 21.5, "tok_per_s": 10.0, "precision": "bf16"},
            {"method": "A_bf16", "seed": 43, "base_model": "m", "n_eval": 3009, "accuracy": 62.0,
             "n_unparsed": 3, "size_gb": 21.5, "tok_per_s": 12.0, "precision": "bf16"}]
    agg = V.aggregate_seeds(rows)["A_bf16"]
    assert agg["accuracy"] == {"mean": 61.0, "std": 1.0, "n": 2}
    assert "f1" not in agg and agg["seeds"] == [42, 43]


def test_collect_eval_rows_filters_split_engine_and_method(tmp_path):
    cfg = load_config(MIDM_CFG)
    base = cfg["base_model"]["selected"]

    def w(name, **kw):
        row = {"method": "A_bf16", "seed": 42, "base_model": base, "n_eval": 3009,
               "accuracy": 60.0, "split": "final_holdout"}
        row.update(kw)
        (tmp_path / name).write_text(json.dumps(row))

    w(R.eval_filename("A_bf16", 42))
    w(R.eval_filename("B_int4_ptq", 42), method="B_int4_ptq")
    w(R.eval_filename("A_bf16", 42, "selection_dev"), split="selection_dev", n_eval=1059)
    w(R.eval_filename("base", 42), method="base")
    w(R.eval_filename("C_int4_qat", 42, engine="vllm"), method="C_int4_qat")
    rows, skipped, n = R.collect_eval_rows(cfg, str(tmp_path), engine="hf")
    assert sorted(r["method"] for r in rows) == ["A_bf16", "B_int4_ptq"] and n == 3009
    vrows, _, _ = R.collect_eval_rows(cfg, str(tmp_path), engine="vllm")
    assert [r["method"] for r in vrows] == ["C_int4_qat"]


def _cfg_in(tmp_path):
    cfg = load_config(MIDM_CFG)
    cfg["paths"] = {k: str(tmp_path / os.path.basename(v)) if k.startswith("method_") else v
                    for k, v in cfg["paths"].items()}
    cfg["paths"]["results_dir"] = str(tmp_path / "results")
    return cfg


def test_cmd_c_export_only_skips_training_and_records_env(tmp_path, monkeypatch):
    from types import SimpleNamespace

    cfg = _cfg_in(tmp_path)
    calls = []

    def fake_export(c, src, dst):
        calls.append((src, dst))
        return {"method": "C_int4_qat", "src_bf16": src, "int4_dir": dst, "group_size": 128,
                "size_gb": 7.5}

    def no_train(*a, **k):
        raise AssertionError("export-only must not train")

    monkeypatch.setattr(V, "export_method_c", fake_export)
    monkeypatch.setattr(V, "train_method_c", no_train)
    R.cmd_c(cfg, SimpleNamespace(seed=42, export_only=True, resume=False))
    c = R.artifact_dir(cfg, "C_int4_qat", 42)
    assert calls == [(c + "_bf16adapt", c)]
    out = json.load(open(os.path.join(cfg["paths"]["results_dir"], "C_export_seed42.json")))
    assert out["seed"] == 42 and out["size_gb"] == 7.5
    assert {"torch", "torchao", "device"} <= set(out)


def test_cmd_b_records_build_env(tmp_path, monkeypatch):
    from types import SimpleNamespace

    cfg = _cfg_in(tmp_path)
    os.makedirs(R.artifact_dir(cfg, "A_bf16", 42))
    monkeypatch.setattr(V, "build_method_b", lambda c, a, b: {"method": "B_int4_ptq", "src_A": a,
                                                               "int4_dir": b, "size_gb": 7.4})
    R.cmd_b(cfg, SimpleNamespace(seed=42))
    out = json.load(open(os.path.join(cfg["paths"]["results_dir"], "B_build_seed42.json")))
    assert out["size_gb"] == 7.4 and {"torch", "torchao", "device"} <= set(out)


def test_eval_filename_scheme():
    assert R.eval_filename("A_bf16", 42) == "eval_A_bf16_seed42.json"       # historical name
    assert R.eval_filename("A_bf16", 42, "selection_dev") == "eval_A_bf16_seed42_selection_dev.json"
    assert R.eval_filename("base", 42, engine="vllm") == "eval_vllm_base_seed42.json"


def test_summary_line_for_both_tasks():
    assert "acc=61.0%" in R.summary_line({"accuracy": 61.0, "n_eval": 10})
    assert "F1=90" in R.summary_line({"exact_match": 80, "f1": 90, "perplexity": 9,
                                      "peak_vram_gb": 1, "tok_per_s": 2})


def test_cost_per_1k_questions():
    from quantization.eval_vllm import cost_per_1k

    assert cost_per_1k(3.6, 360.0, 3000) == pytest.approx(0.12)     # $3.6/h, 6 min, 3k q
    assert cost_per_1k(None, 10.0, 10) is None and cost_per_1k(1.0, 10.0, 0) is None


def test_parse_prices_keeps_spot_and_ondemand():
    from quantization.eval_vllm import parse_prices

    assert parse_prices(None, ["spot=3.537", "ondemand=6.98"]) == {"spot": 3.537, "ondemand": 6.98}
    assert parse_prices(2.0, []) == {"usd_per_hour": 2.0}
    assert parse_prices() == {}
    with pytest.raises(ValueError):
        parse_prices(None, ["3.5"])


def test_final_holdout_is_measured_once(tmp_path):
    from quantization.eval_vllm import guard_final_holdout

    out = tmp_path / "eval_vllm_A_bf16_seed42.json"
    guard_final_holdout(str(out), "final_holdout")                 # first run: allowed
    out.write_text("{}")
    with pytest.raises(SystemExit):
        guard_final_holdout(str(out), "final_holdout")             # second run: blocked
    guard_final_holdout(str(out), "final_holdout", force=True)     # explicit re-measure
    guard_final_holdout(str(out), "selection_dev")                 # dev is never blocked
    guard_final_holdout(str(out), "final_holdout", eval_size=50)   # subsets are never blocked


def test_pinned_revisions_and_vocab_in_midm_config():
    cfg = load_config(MIDM_CFG)
    assert len(cfg["base_model"]["revision"]) == 40 and len(cfg["data"]["revision"]) == 40
    assert cfg["base_model"]["vocab_size"] == 131384


def test_merged_vocab_check(tmp_path):
    cfg = load_config(MIDM_CFG)
    d = tmp_path / "A_bf16_seed42"
    d.mkdir()
    assert V.merged_vocab_check(cfg, str(d))["vocab_ok"] is None    # no config.json yet
    (d / "config.json").write_text(json.dumps({"vocab_size": 131384}))
    assert V.merged_vocab_check(cfg, str(d))["vocab_ok"] is True
    (d / "config.json").write_text(json.dumps({"vocab_size": 131385}))   # pad token added
    r = V.merged_vocab_check(cfg, str(d))
    assert r["vocab_ok"] is False and r["vocab_size_merged"] == 131385


def test_unsloth_config_extends_base_but_writes_elsewhere():
    base = load_config(MIDM_CFG)
    uns = load_config(os.path.join(HERE, "..", "configs", "kormedmcqa_midm_unsloth.yaml"))
    assert uns["train"]["backend"] == "unsloth" and uns["train"]["unsloth_external_tokenizer"] is True
    # same recipe ...
    for k in ("base_model", "data", "lora", "qat", "ptq", "eval"):
        assert uns[k] == base[k], k
    assert {k: v for k, v in uns["train"].items() if k != "unsloth_external_tokenizer"} == \
        {**base["train"], "backend": "unsloth"}
    # ... but never the hf-backend output folders
    for k in ("results_dir", "method_a_dir", "method_b_dir", "method_c_dir"):
        assert uns["paths"][k] != base["paths"][k], k
    assert R.artifact_dir(uns, "A_bf16", 42).endswith("A_bf16_unsloth_seed42")


def test_config_extends_deep_merges_and_rejects_loops(tmp_path):
    from quantization.data_korquad import _read_config

    (tmp_path / "base.yaml").write_text("a: {x: 1, y: 2}\nb: [1, 2]\ncompute: {mode: gpu}\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\na: {y: 3}\nb: [9]\n")
    assert _read_config(str(tmp_path / "child.yaml")) == {"a": {"x": 1, "y": 3}, "b": [9], "compute": {"mode": "gpu"}}
    (tmp_path / "loop.yaml").write_text("extends: loop.yaml\n")
    with pytest.raises(ValueError):
        _read_config(str(tmp_path / "loop.yaml"))
