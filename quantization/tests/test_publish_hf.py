"""CPU tests for the model card / upload helper (no huggingface_hub, no network)."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from quantization import publish_hf as P  # noqa: E402


@pytest.mark.parametrize("kind", sorted(P.KINDS))
def test_card_carries_required_notices(kind):
    card = P.render_model_card(kind, "HyeonSang/Midm-2.0-KorMedMCQA")
    front = card.split("---")[1]
    assert "license: other" in front and "license_link: LICENSE" in front
    assert f"base_model: {P.BASE_MODEL}" in front and f"- {P.DATASET}" in front
    assert "의료 조언이 아닙니다" in card                                  # not medical advice
    assert "비상업 용도로만" in card and "CC BY-NC 2.0" in card             # NC condition
    assert "Copyright (c) 2025 KT Corporation" in card                      # MIT notice
    assert P.DATASET_PAPER in card                                          # attribution
    assert 'date_string="21 Nov 2026"' in card and "131301" in card         # reproducible usage


def test_card_results_table_from_eval_rows():
    rows = [{"method": "A_bf16", "accuracy": 61.2, "accuracy_ci95": [59.4, 62.9], "n_eval": 3009,
             "n_unparsed": 0, "gpu_label": "A100 80GB (Spot)", "engine": "vllm"}]
    card = P.render_model_card("A_bf16", "x/y", eval_rows=rows,
                               train_log={"backend": "unsloth", "global_step": 426, "seed": 42})
    assert "| A_bf16 | 61.2 | 59.4–62.9 | 3009 | 0 | A100 80GB (Spot) | vllm |" in card
    assert "`backend=unsloth`" in card and "`global_step=426`" in card


def test_card_without_results_says_pending():
    assert "평가가 끝나면" in P.render_model_card("C_bf16adapt", "x/y")


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        P.render_model_card("B_int4_ptq", "x/y")


def test_license_file_has_full_mit_text_and_nc_note():
    lic = P.render_license()
    assert lic.startswith("MIT License") and "Copyright (c) 2025 KT Corporation" in lic
    assert "NON-COMMERCIAL" in lic and P.DATASET in lic


def test_upload_patterns_exclude_pickles_and_trainer_state():
    assert "*.bin" in P.IGNORE_PATTERNS and "optimizer*" in P.IGNORE_PATTERNS
    assert "*.safetensors" in P.ALLOW_PATTERNS and "LICENSE" in P.ALLOW_PATTERNS


def test_dry_run_writes_card_and_license(tmp_path):
    d = tmp_path / "A_bf16_seed42"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"x")
    tj = tmp_path / "A_train_seed42.json"
    tj.write_text(json.dumps({"backend": "hf", "global_step": 5}))
    assert P.main(["--kind", "A_bf16", "--dir", str(d), "--repo", "x/y", "--train-json", str(tj),
                   "--dry-run"]) == 0
    assert "license: other" in (d / "README.md").read_text(encoding="utf-8")
    assert "KT Corporation" in (d / "LICENSE").read_text(encoding="utf-8")


def test_refuses_pickled_int4_dir(tmp_path):
    d = tmp_path / "B_int4_ptq_seed42"
    d.mkdir()
    (d / "pytorch_model.bin").write_bytes(b"x")
    with pytest.raises(SystemExit):
        P.main(["--kind", "A_bf16", "--dir", str(d), "--repo", "x/y", "--dry-run"])
