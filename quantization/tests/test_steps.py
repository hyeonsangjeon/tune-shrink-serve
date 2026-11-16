"""steps/ (the readable step-by-step scripts) must stay in sync with the run that made the numbers.

CPU only: the scripts import torch/transformers/torchao/vllm inside functions, so loading them
here checks that, and their settings are compared with the YAML config the talk run used.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STEPS = os.path.join(ROOT, "steps")
CFG_PATH = os.path.join(ROOT, "quantization", "configs", "kormedmcqa_midm.yaml")
PY_STEPS = ["1_lora_train", "2_int4_ptq", "3_int4_qat", "4_evaluate", "5_ask"]


def _load(name: str):
    if STEPS not in sys.path:
        sys.path.insert(0, STEPS)        # the scripts do `import common`
    spec = importlib.util.spec_from_file_location(f"steps_{name}", os.path.join(STEPS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cfg():
    with open(CFG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def mods():
    return {n: _load(n) for n in PY_STEPS + ["common"]}


def test_common_matches_config(cfg, mods):
    c = mods["common"]
    assert c.BASE_MODEL == cfg["base_model"]["selected"]
    assert c.BASE_REVISION == cfg["base_model"]["revision"]
    assert c.MAX_SEQ_LEN == cfg["data"]["max_seq_len"]
    assert [c.SEED] == cfg["seeds"]
    assert os.path.samefile(c.CONFIG, CFG_PATH)


def test_lora_settings_match_config(cfg, mods):
    m, lora, tr = mods["1_lora_train"], cfg["lora"], cfg["train"]
    assert (m.LORA_R, m.LORA_ALPHA, m.LORA_DROPOUT) == (lora["r"], lora["alpha"], lora["dropout"])
    assert m.LORA_TARGETS == lora["target_modules"]
    assert not lora.get("use_rslora", False)
    assert m.EPOCHS == tr["epochs"]
    assert (m.BATCH_SIZE, m.GRAD_ACCUM) == (tr["per_device_batch_size"], tr["grad_accum"])
    assert m.LEARNING_RATE == float(tr["learning_rate"])
    assert (m.WARMUP_RATIO, m.WEIGHT_DECAY, m.MAX_GRAD_NORM) == (
        tr["warmup_ratio"], tr["weight_decay"], tr["max_grad_norm"])
    assert m.SAVE_STEPS == tr["save_steps"]
    assert tr["gradient_checkpointing"] is True and tr["load_in_4bit"] is False
    assert tr.get("max_steps") is None


def test_quantization_settings_match_config(cfg, mods):
    ptq, qat, q = mods["2_int4_ptq"], mods["3_int4_qat"], cfg["qat"]
    assert ptq.GROUP_SIZE == cfg["ptq"]["group_size"] == qat.GROUP_SIZE == q["group_size"]
    assert qat.MAX_STEPS == q["max_steps"]
    assert qat.LEARNING_RATE == float(q["learning_rate"])
    assert (qat.BATCH_SIZE, qat.GRAD_ACCUM) == (q["per_device_batch_size"], q["grad_accum"])
    assert (qat.OPTIM, qat.SAVE_STEPS) == (q["optim"], q["save_steps"])


def test_eval_and_ask_match_config(cfg, mods):
    assert mods["4_evaluate"].MAX_NEW_TOKENS == cfg["eval"]["max_new_tokens"]
    ask = mods["5_ask"]
    assert ask.build_request.__defaults__ == (cfg["eval"]["max_new_tokens"],)


def test_ask_request_is_the_eval_prompt(mods):
    from quantization import kormedmcqa as K
    from quantization import v2_pipeline as V

    common = mods["common"]
    _, st = common.load_settings()
    ex = K.MCQExample(id="nurse-2022-1-1", role="nurse", year=2022, question="Q?",
                      choices=["a", "b", "c", "d", "e"], gold=3)
    body = mods["5_ask"].build_request(ex, st, [131301, 2])
    assert body["messages"] == [{"role": "system", "content": st["system"]},
                                {"role": "user", "content": V.user_content_for(ex, st)}]
    assert body["chat_template_kwargs"] == {"date_string": "21 Nov 2026"}
    assert body["temperature"] == 0.0 and body["stop_token_ids"] == [131301, 2]


def test_outputs_stay_in_the_ignored_folder(mods):
    c = mods["common"]
    for d in (c.LORA_DIR, c.PTQ_DIR, c.QAT_DIR, c.QAT_BF16_DIR):
        assert os.path.dirname(d) == c.OUT
    with open(os.path.join(ROOT, ".gitignore"), encoding="utf-8") as fh:
        assert "outputs/" in fh.read().split("\n")      # model weights never reach git


@pytest.mark.parametrize("name", PY_STEPS)
def test_help_runs_without_gpu_libraries(mods, name, capsys):
    with pytest.raises(SystemExit) as e:
        mods[name].main(["--help"])
    assert e.value.code == 0
    assert "usage" in capsys.readouterr().out


def test_missing_model_folder_is_a_clear_error(mods, tmp_path):
    for name in ("2_int4_ptq", "3_int4_qat"):
        with pytest.raises(SystemExit, match="먼저 1단계"):
            mods[name].main(["--src", str(tmp_path / "nope")])


def test_is_int4_dir(mods, tmp_path):
    c = mods["common"]
    (tmp_path / "config.json").write_text('{"quantization_config": {"quant_method": "torchao"}}')
    assert c.is_int4_dir(str(tmp_path))
    (tmp_path / "config.json").write_text('{"architectures": ["LlamaForCausalLM"]}')
    assert not c.is_int4_dir(str(tmp_path))
    assert not c.is_int4_dir("K-intelligence/Midm-2.0-Base-Instruct")


def test_serve_script(tmp_path):
    script = os.path.join(STEPS, "5_serve.sh")
    assert subprocess.run(["bash", "-n", script]).returncode == 0
    r = subprocess.run(["bash", script, str(tmp_path / "nope")], capture_output=True, text=True)
    assert r.returncode == 2 and "모델 폴더가 없습니다" in r.stderr
