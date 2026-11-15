"""CPU tests for the option-order views (no datasets / torch import)."""
from collections import Counter

from quantization import kormedmcqa as K


def _ex(choices=("가", "나", "다", "라", "마"), gold=3, role="dentist", i=0):
    return K.MCQExample(id=f"{role}-2020-1-{i}", role=role, year=2020, question="Q?",
                        choices=list(choices), gold=gold)


def test_gold_text_and_option_set_preserved():
    ex = _ex()
    for v in K.option_views(ex, 5, seed=42):
        assert v.choices[v.gold - 1] == ex.choices[ex.gold - 1]
        assert sorted(v.choices) == sorted(ex.choices)


def test_identity_first_and_views_distinct():
    vs = K.option_views(_ex(), 4, seed=42)
    assert vs[0].choices == _ex().choices and vs[0].gold == 3
    assert len({tuple(v.choices) for v in vs}) == 4
    assert len({v.id for v in vs}) == 4


def test_numeric_sorted_items_keep_order():
    ex = _ex(choices=("900명", "1,000명", "1,100명", "1,200명", "1,300명"))
    ex2 = _ex(choices=("0.5 mg", "1 mg", "2 mg", "5 mg", "10 mg"))
    assert K.numeric_sorted(ex.choices) and K.numeric_sorted(ex2.choices)
    assert [v.choices for v in K.option_views(ex, 4, 42)] == [list(ex.choices)]
    assert [v.choices for v in K.option_views(ex2, 4, 42)] == [list(ex2.choices)]


def test_expand_train_counts_and_role_weights():
    items = [_ex(role="doctor", i=i) for i in range(10)] + [_ex(role="dentist", i=i) for i in range(10)]
    out = K.expand_train(items, {"option_permutations": 4}, 4, {"doctor": 0.5, "dentist": 2.0}, 42)
    c = Counter(v.role for v in out)
    assert c["doctor"] == 20 and c["dentist"] == 80


def test_off_is_identity():
    items = [_ex(i=i) for i in range(5)]
    out = K.expand_train(items, {"option_permutations": 1}, 1, None, 42)
    assert sorted(v.choices for v in out) == sorted(e.choices for e in items)


def test_expand_train_epoch_blocks_see_every_item_once_per_epoch():
    items = [_ex(role="nurse", i=i) for i in range(30)]
    out = K.expand_train(items, {"option_permutations": 4}, 4, None, 42)
    assert len(out) == 120
    for e in range(4):
        block = out[30 * e: 30 * (e + 1)]
        assert sorted(v.id.split("#")[0] for v in block) == sorted(x.id for x in items)
    # an item shows a different option order in every epoch
    by_item = {}
    for v in out:
        by_item.setdefault(v.id.split("#")[0], []).append(tuple(v.choices))
    assert all(len(set(orders)) == 4 for orders in by_item.values())


def test_role_upsampling_never_repeats_an_order_when_shuffling():
    out = K.expand_train([_ex(role="dentist", i=0)], {"option_permutations": 4}, 4, {"dentist": 1.8}, 42)
    assert len(out) == 7 and len({tuple(v.choices) for v in out}) == 7


def test_extra_examples_join_every_epoch_block():
    items = [_ex(role="doctor", i=i) for i in range(5)]
    extra = [K.ExplainExample(base=items[0], explanation="해설")]
    out = K.expand_train(items, {}, 2, None, 42, extra_per_epoch=extra)
    assert len(out) == 12
    assert sum(isinstance(v, K.ExplainExample) for v in out[:6]) == 1
    assert sum(isinstance(v, K.ExplainExample) for v in out[6:]) == 1


def test_fractional_epochs_rejected():
    import pytest

    with pytest.raises(ValueError):
        K.expand_train([_ex()], {"option_permutations": 4}, 2.5, None, 42)


def test_extract_final_choice_reads_the_last_answer_line():
    assert K.extract_final_choice("1번은 틀림, 2번도 틀림.\n정답: 4") == 4
    assert K.extract_final_choice("정답은 ②입니다") == 2
    assert K.extract_final_choice("정답: 1\n다시 보면 정답: 3") == 3
    assert K.extract_final_choice("정답: 12") is None
    assert K.extract_final_choice("3") == 3          # answer-only output still reads
    assert K.extract_final_choice("") is None


def test_explain_example_prompt_and_target():
    e = K.ExplainExample(base=_ex(gold=3), explanation="  해설입니다.  ")
    assert e.answer == "해설입니다.\n정답: 3"
    assert e.system_override == K.EXPLAIN_SYSTEM
    assert e.user_content().endswith(K.EXPLAIN_SUFFIX) and e.id.endswith("#explain")
    assert (e.role, e.gold) == ("dentist", 3)
if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
