"""The evidence checker's optional ``scale`` turns a JSON unit (s, MiB) into the unit shown in
the docs (ms, minutes, GiB) before comparing, and still demands the literal in the doc."""
import importlib.util
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
_spec = importlib.util.spec_from_file_location(
    "build_evidence_index", os.path.join(ROOT, "scripts", "build_evidence_index.py"))
bei = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bei)


def _claim(**kw):
    base = {"id": "x", "capability": "c", "status": "recorded_hardware_run",
            "source": "r.json", "pointer": "/t", "readme": "doc.md"}
    base.update(kw)
    return base


def test_scale_converts_seconds_to_ms(tmp_path):
    (tmp_path / "r.json").write_text(json.dumps({"t": 0.1088}))
    (tmp_path / "doc.md").write_text("첫 토큰 109 ms")
    assert bei.verify_claim(_claim(scale=1000, expected="109"), str(tmp_path)) == []


def test_scale_mismatch_and_missing_literal_fail(tmp_path):
    (tmp_path / "r.json").write_text(json.dumps({"t": 84267}))
    (tmp_path / "doc.md").write_text("82.3 GiB")
    assert bei.verify_claim(_claim(scale=1 / 1024, expected="82.3"), str(tmp_path)) == []
    errs = bei.verify_claim(_claim(scale=1 / 1024, expected="80.0"), str(tmp_path))
    assert any("!= expected" in e for e in errs) and any("not found in doc.md" in e for e in errs)


def test_without_scale_behaviour_is_unchanged(tmp_path):
    (tmp_path / "r.json").write_text(json.dumps({"t": 72.316}))
    (tmp_path / "doc.md").write_text("72.3%")
    assert bei.verify_claim(_claim(expected="72.3"), str(tmp_path)) == []
