"""CPU tests for the KorMedMCQA task helpers (no datasets / torch / network)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from quantization import kormedmcqa as K  # noqa: E402


def _ex(i, role="doctor", year=2022, gold=3):
    return K.MCQExample(id=f"{role}-{year}-1-{i}", role=role, year=year,
                        question=f"질문 {i}", choices=["가", "나", "다", "라", "마"], gold=gold)


# ---------------------------------------------------------------- answer extraction
@pytest.mark.parametrize("text,expected", [
    ("3", 3), ("3번", 3), ("정답: 3", 3), ("정답은 ③입니다", 3), ("(4)", 4), ("2.", 2),
    ("2. 아스피린", 2), ("**5**", 5), ("1)", 1), ("C", 3), ("답: E", 5), ("A", 1),
    ("<think>1 아니면 2</think>4", 4), ("보기 2번과 3번 중 2번", 2), (" 4\n", 4),
    ("정답은 3번입니다.", 3),
])
def test_extract_choice_accepts_common_formats(text, expected):
    assert K.extract_choice(text) == expected


@pytest.mark.parametrize("text", [
    "", None, "   ", "12", "3.5", "0.3", "6", "정답은 5 mg입니다", "모르겠습니다", "2023년",
])
def test_extract_choice_rejects_non_choices(text):
    assert K.extract_choice(text) is None


def test_circled_digit_wins_over_plain_digit():
    assert K.extract_choice("2023년 기준 정답은 ④") == 4


# ---------------------------------------------------------------- rendering / rows
def test_render_user_content_numbers_choices_and_names_role():
    ex = _ex(1, role="pharm")
    text = K.render_user_content(ex)
    assert "약사 국가시험" in text
    assert "1. 가\n2. 나\n3. 다\n4. 라\n5. 마" in text
    assert text.index("질문 1") < text.index("1. 가")


def test_user_template_with_braces_in_question_is_safe():
    ex = _ex(1)
    ex.question = "수식 {x}와 {}가 포함된 문제"
    assert "{x}" in K.render_user_content(ex)


def test_row_to_example_answer_bases():
    row = {"question": "q", "A": "a", "B": "b", "C": "c", "D": "d", "E": "e",
           "answer": 5, "year": 2022, "period": 1, "q_number": 7}
    ex = K.row_to_example(row, "nurse", answer_base=1)
    assert ex.gold == 5 and ex.answer == "5" and ex.answers == ["5"] and ex.context == ""
    assert ex.id == "nurse-2022-1-7"
    row0 = dict(row, answer=4)
    assert K.row_to_example(row0, "nurse", answer_base=0).gold == 5
    with pytest.raises(ValueError):
        K.row_to_example(dict(row, answer=6), "nurse", answer_base=1)


def test_detect_answer_base():
    assert K._detect_answer_base([{"answer": 1}, {"answer": 5}]) == 1
    assert K._detect_answer_base([{"answer": 0}, {"answer": 4}]) == 0
    with pytest.raises(ValueError):
        K._detect_answer_base([{"answer": 0}, {"answer": 5}])


# ---------------------------------------------------------------- scoring
def test_score_counts_unparsed_as_wrong_and_breaks_down():
    exs = [_ex(1, gold=1), _ex(2, gold=2), _ex(3, role="nurse", year=2023, gold=3),
           _ex(4, role="nurse", year=2023, gold=4)]
    sc = K.score([1, None, 3, 5], exs, raw=["1", "모름", "3", "5"])
    assert sc["n"] == 4 and sc["n_correct"] == 2 and sc["n_unparsed"] == 1
    assert sc["accuracy"] == 50.0
    assert sc["by_role"]["doctor"] == {"accuracy": 50.0, "n": 2}
    assert sc["by_year"]["2023"] == {"accuracy": 50.0, "n": 2}
    assert sc["items"][1] == {"id": exs[1].id, "role": "doctor", "year": 2022, "gold": 2,
                              "pred": None, "correct": 0, "raw": "모름"}
    lo, hi = sc["accuracy_ci95"]
    assert lo < 50.0 < hi


def test_score_length_mismatch_raises():
    with pytest.raises(ValueError):
        K.score([1], [_ex(1), _ex(2)])


def test_wilson_ci_narrows_with_n():
    small = K.wilson_ci(6, 10)
    big = K.wilson_ci(1800, 3000)
    assert (big[1] - big[0]) < (small[1] - small[0])
    assert abs((big[1] - big[0]) / 2 - 1.75) < 0.1      # ≈ ±1.75pt at 60%, n=3,000


# ---------------------------------------------------------------- McNemar / pairing
def test_mcnemar_exact_known_values():
    assert K.mcnemar_exact(0, 0) == 1.0
    assert K.mcnemar_exact(5, 5) == 1.0
    assert abs(K.mcnemar_exact(10, 0) - 2 / 1024) < 1e-12
    assert K.mcnemar_exact(3, 7) == K.mcnemar_exact(7, 3)


def test_paired_compare_counts_discordant_pairs():
    a = [{"id": "x", "correct": 1}, {"id": "y", "correct": 1}, {"id": "z", "correct": 0},
         {"id": "w", "correct": 0}]
    b = [{"id": "x", "correct": 1}, {"id": "y", "correct": 0}, {"id": "z", "correct": 1},
         {"id": "w", "correct": 0}, {"id": "extra", "correct": 1}]
    r = K.paired_compare(a, b)
    assert r["n_paired"] == 4 and r["missing_in_a"] == 1 and r["missing_in_b"] == 0
    assert (r["both_correct"], r["only_a_correct"], r["only_b_correct"], r["both_wrong"]) == (1, 1, 1, 1)
    assert r["acc_a"] == 50.0 and r["acc_b"] == 50.0 and r["mcnemar_p"] == 1.0


# ---------------------------------------------------------------- splits / subsets
def test_stratified_subset_is_deterministic_and_keeps_role_mix():
    exs = [_ex(i, role="doctor") for i in range(60)] + [_ex(i, role="nurse") for i in range(40)]
    s1 = K.stratified_subset(exs, 10, seed=42)
    s2 = K.stratified_subset(list(reversed(exs)), 10, seed=42)
    assert [e.id for e in s1] == [e.id for e in s2]
    assert len(s1) == 10
    roles = [e.role for e in s1]
    assert roles.count("doctor") == 6 and roles.count("nurse") == 4
    assert K.stratified_subset(exs, 1000, seed=1) == exs


def test_assert_disjoint_catches_shared_ids_and_years():
    train = [_ex(1, year=2019), _ex(2, year=2020)]
    test = [_ex(3, year=2022)]
    man = K.assert_disjoint({"train": train, "final_holdout": test})
    assert man["train"]["years"] == [2019, 2020] and man["final_holdout"]["n"] == 1
    with pytest.raises(K.SplitLeakageError):
        K.assert_disjoint({"train": train, "final_holdout": [_ex(9, year=2020)]})
    with pytest.raises(K.SplitLeakageError):
        K.assert_disjoint({"train": train, "final_holdout": [train[0]]})


def test_compare_cli(tmp_path, capsys):
    import json

    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"items": [{"id": "x", "correct": 1}, {"id": "y", "correct": 1}]}))
    b.write_text(json.dumps({"items": [{"id": "x", "correct": 0}, {"id": "y", "correct": 1}]}))
    assert K._main(["compare", str(a), str(b)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["only_a_correct"] == 1 and out["only_b_correct"] == 0
