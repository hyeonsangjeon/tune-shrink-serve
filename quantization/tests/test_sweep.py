"""CPU tests for the recipe-sweep helpers (no GPU, no vLLM, no datasets)."""
import json
import types

from quantization import kormedmcqa as K
from quantization import self_explain as SE
from quantization import sweep_report as SR
from quantization import eval_vllm as EV
from quantization import v2_pipeline as V


def test_self_explain_select_needs_min_correct_and_strips_answer_line():
    good = "기전상 3번이 맞다. 다른 보기는 해당하지 않는다.\n정답: 3"
    r = SE.select([good, "추론...\n정답: 2", good], gold=3, min_correct=2, max_chars=700)
    assert r["keep"] and r["n_correct"] == 2 and not r["explanation"].endswith("3")
    assert r["explanation"].startswith("기전상")
    r = SE.select([good, "추론...\n정답: 2"], gold=3, min_correct=2, max_chars=700)
    assert not r["keep"] and r["n_correct"] == 1          # one lucky hit is not enough
    assert SE.strip_final_line("풀이\n정답: 5") == "풀이"


def test_pick_checkpoint_prefers_earlier_within_tie():
    rows = [{"step": 213, "accuracy": 71.0}, {"step": 426, "accuracy": 71.4}, {"step": 639, "accuracy": 70.0}]
    assert SR.pick_checkpoint(rows)["step"] == 213
    rows[0]["accuracy"] = 70.8
    assert SR.pick_checkpoint(rows)["step"] == 426
    assert SR.pick_checkpoint([]) is None


def test_gate_needs_both_size_and_significance():
    assert SR.gate({"acc_a": 70.0, "acc_b": 72.5, "mcnemar_p": 0.01})
    assert not SR.gate({"acc_a": 70.0, "acc_b": 72.5, "mcnemar_p": 0.2})
    assert not SR.gate({"acc_a": 70.0, "acc_b": 71.0, "mcnemar_p": 0.001})


def test_ensemble_averages_probabilities(tmp_path):
    a = {"items": [{"id": "q1", "gold": 1, "probs": [0.6, 0.4, 0, 0, 0]},
                   {"id": "q2", "gold": 2, "probs": [0.55, 0.45, 0, 0, 0]}]}
    b = {"items": [{"id": "q1", "gold": 1, "probs": [0.3, 0.7, 0, 0, 0]},
                   {"id": "q2", "gold": 2, "probs": [0.1, 0.9, 0, 0, 0]}]}
    fa, fb = tmp_path / "a.json", tmp_path / "b.json"
    fa.write_text(json.dumps(a)); fb.write_text(json.dumps(b))
    r = SR.ensemble([str(fa), str(fb)])
    assert r["n"] == 2 and r["accuracy"] == 50.0        # q1 -> option 2 (wrong), q2 -> option 2 (right)


def test_option_probs_and_argmax():
    lp = types.SimpleNamespace
    step = [{11: lp(logprob=-0.1), 12: lp(logprob=-2.5), 99: lp(logprob=-3.0)}]
    p = EV.option_probs(step, [11, 12, 13, 14, 15])
    assert p[0] > p[1] > 0 and p[2:] == [0.0, 0.0, 0.0]
    assert EV.argmax_choice(p) == 1 and EV.argmax_choice([0.0] * 5) is None
    assert EV.option_probs(None, [1, 2, 3, 4, 5]) == [0.0] * 5


class _Tok:
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        return " | ".join(f"{m['role']}:{m['content']}" for m in msgs)


def test_render_prompt_uses_the_explain_system_prompt():
    st = {"system": "SYS", "template_kwargs": {}, "user_template": None, "enable_thinking": False}
    ex = K.MCQExample(id="nurse-2019-1-1", role="nurse", year=2019, question="Q?", choices=list("가나다라마"), gold=1)
    assert V.render_prompt(_Tok(), ex, st).startswith("system:SYS")
    assert V.render_prompt(_Tok(), K.explain_prompt_example(ex), st).startswith("system:" + K.EXPLAIN_SYSTEM)


def test_train_examples_for_run_today_vs_sweep(tmp_path):
    items = [K.MCQExample(id=f"doctor-2020-1-{i}", role="doctor", year=2020, question="Q?",
                          choices=list("가나다라마"), gold=1 + i % 5) for i in range(6)]
    base = {"data": {"task": "kormedmcqa"}, "train": {"epochs": 2}}
    ex, ep, info = V.train_examples_for_run(base, {"train": items}, 42)
    assert ex is items and ep == 2.0 and info is None            # today's run is untouched
    jl = tmp_path / "e.jsonl"
    jl.write_text("\n".join(json.dumps({"id": e.id, "keep": i < 3, "explanation": "해설"})
                            for i, e in enumerate(items)), encoding="utf-8")
    cfg = {"data": {"task": "kormedmcqa", "self_explain": {"path": str(jl)}}, "train": {"epochs": 2}}
    ex, ep, info = V.train_examples_for_run(cfg, {"train": items}, 42)
    assert ep == 1.0 and info["sequential"] and info["n_explain_items"] == 3 and len(ex) == 2 * (6 + 3)


def _items(role, n, start=0):
    return [K.MCQExample(id=f"{role}-2020-1-{start + i}", role=role, year=2020, question="Q?",
                         choices=list("가나다라마"), gold=1 + (start + i) % 5) for i in range(n)]


def test_role_weights_keep_every_block_balanced():
    items = _items("doctor", 300) + _items("nurse", 100, 300) + _items("dentist", 50, 400)
    w = {"doctor": 0.71, "nurse": 1.28, "dentist": 1.80}
    out = K.expand_train(items, {"option_permutations": 4}, 4, w, 42)
    n = {"doctor": round(4 * 0.71), "nurse": round(4 * 1.28), "dentist": round(4 * 1.80)}
    total = 300 * n["doctor"] + 100 * n["nurse"] + 50 * n["dentist"]
    assert len(out) == total
    # rebuild the block boundaries: each block is the same size within a few views
    sizes = [total // 4] * 4
    for b in range(4):
        blk = out[sum(sizes[:b]): sum(sizes[:b + 1])]
        share = sum(1 for v in blk if v.role == "doctor") / len(blk)
        assert abs(share - 300 * n["doctor"] / total) < 0.08, (b, share)   # no doctor-free block


def test_unweighted_layout_is_unchanged_by_the_block_offset():
    items = _items("nurse", 40)
    a = K.expand_train(items, {"option_permutations": 4}, 4, None, 42)
    b = K.expand_train(items, {"option_permutations": 4}, 4, {"nurse": 1.0}, 42)
    assert [v.id for v in a] == [v.id for v in b]


def test_final_answer_line_variants_and_explain_prompt():
    assert K.final_answer_hit("풀이\n정답: 3.") == 3
    assert K.final_answer_hit("**정답**: 4") == 4
    assert K.final_answer_hit("정답: ③.") == 3
    assert K.final_answer_hit("3") is None and K.extract_final_choice("3") == 3
    tmpl = "다음은 {role_ko} 국가시험 문제입니다. 보기 중 정답의 번호 하나만 숫자로 답하세요.\n\n{question}\n{choices}"
    u = K.ExplainExample(base=_items("nurse", 1)[0], explanation="").user_content(tmpl)
    assert "하나만 숫자로" not in u and u.endswith(K.EXPLAIN_SUFFIX)


def test_self_explain_ignores_cut_off_samples():
    good = "근거 문장이 충분히 길게 이어진다. 다른 보기는 해당하지 않는다.\n정답: 2"
    r = SE.select([good, good], gold=2, min_correct=2, max_chars=700, finish=["stop", "length"])
    assert r["n_correct"] == 1 and not r["keep"] and r["n_truncated"] == 1
    r = SE.select(["2번이 답인 것 같은데 설명이 끊겼", good, good], gold=2, min_correct=2, max_chars=700)
    assert r["n_correct"] == 2 and r["keep"] and r["n_no_final_line"] == 1


def test_measure_once_across_modes(tmp_path):
    import pytest

    (tmp_path / "eval_vllm_A_bf16_seed42_h100.json").write_text("{}")
    with pytest.raises(SystemExit):
        EV.guard_final_siblings(str(tmp_path / "eval_vllm_A_bf16_seed42_h100_probs.json"), "A_bf16", 42, "final_holdout")
    EV.guard_final_siblings(str(tmp_path / "eval_vllm_A_bf16_seed42_a100.json"), "A_bf16", 42, "final_holdout")
    EV.guard_final_siblings(str(tmp_path / "eval_vllm_A_bf16_seed42_h100_probs.json"), "A_bf16", 42, "selection_dev")


def test_sweep_report_counts_looks_and_skips_partial_runs(tmp_path):
    def run(acc, ids, adapter=None):
        return {"accuracy": acc, "n_eval": len(ids), "adapter": adapter,
                "items": [{"id": i, "correct": int(c)} for i, c in ids]}
    full = [(f"q{i}", i % 3 != 0) for i in range(30)]
    better = [(f"q{i}", True) for i in range(30)]
    ref = tmp_path / "ref.json"; ref.write_text(json.dumps(run(66.7, full)))
    arm = tmp_path / "sweep" / "s1"; (arm / "ckpt_dev").mkdir(parents=True)
    (arm / "ckpt_dev" / "eval_vllm_lora_x_checkpoint-213_selection_dev.json").write_text(
        json.dumps(run(100.0, better, "a/A_bf16_seed42_run/checkpoint-213")))
    (arm / "ckpt_dev" / "eval_vllm_lora_x_checkpoint-5_selection_dev_n20.json").write_text(
        json.dumps(run(50.0, full[:20], "a/A_bf16_seed42_run/checkpoint-5")))
    (arm / "ckpt_dev" / "eval_vllm_lora_x_checkpoint-213_selection_dev_probs.json").write_text(
        json.dumps(run(100.0, better, "a/A_bf16_seed42_run/checkpoint-213")))
    (arm / "eval_vllm_A_bf16_seed42_ckpt213_selection_dev.json").write_text(json.dumps(run(100.0, better)))
    s = SR.summarize(str(tmp_path), str(ref))
    a = s["arms"]["s1"]
    assert a["picked"]["step"] == 213 and a["picked"]["vs_reference"]["passes_gate"]
    assert a["merged"][0]["merged_from_step"] == 213
    assert s["dev_scorings_looked_at"] == 4                   # 3 ckpt_dev files (all modes) + 1 merged
    assert not any("checkpoint-5" in c["file"] for c in a["checkpoints"])


def test_stage_sh_refuses_sweep_stages_without_a_sweep_config(tmp_path):
    import os
    import shutil
    import subprocess

    import pytest

    if not shutil.which("bash"):
        pytest.skip("bash not found")
    env = {k: v for k, v in os.environ.items() if k not in ("CONFIG", "SEED", "TAG")}
    env["WORK"] = str(tmp_path)
    r = subprocess.run(["bash", "scripts/gpu/stage.sh", "sweep_a"], env=env, capture_output=True, text=True)
    assert r.returncode == 64
    env.update(CONFIG="quantization/configs/sweep/s0_seed43.yaml", SEED="42")
    r = subprocess.run(["bash", "scripts/gpu/stage.sh", "sweep_a"], env=env, capture_output=True, text=True)
    assert r.returncode == 64
    log = (tmp_path / "logs" / "sweep_a.log").read_text()
    assert "needs CONFIG" in log and "is seed 43 but SEED=42" in log
