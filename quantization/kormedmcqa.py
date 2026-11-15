"""KorMedMCQA task for the quantization track (Korean healthcare licensing exams).

Dataset: ``sean0042/KorMedMCQA`` (HF card license: CC BY-NC 2.0; runtime download,
never committed). Five-option multiple choice from the Korean doctor / nurse /
pharmacist / dentist licensing exams. The official HF splits are separated by exam
year, which gives the talk its storyline ("study past exams, sit later exams"):

    train          -> train   (2012–2020)   SFT data for A (LoRA) and C (QAT)
    selection_dev  -> dev     (2021)        prompt / checkpoint / step selection ONLY
    final_holdout  -> test    (2022–2024)   final A/B/C comparison, measured once

Metric: accuracy = share of questions where the chosen option number equals the
official answer (chance 20%; 60% is a reference pass line, not the real rule).
Unparseable outputs count as wrong and are reported separately (``n_unparsed``).

Pure helpers (prompt rendering, answer extraction, scoring, McNemar) import nothing
heavy so the CPU test suite covers them; ``datasets`` is imported lazily.

CLI:
    python -m quantization.kormedmcqa --stats                 # split sizes + year ranges
    python -m quantization.kormedmcqa compare A.json B.json   # paired McNemar on items
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DATASET = "sean0042/KorMedMCQA"
ROLES: Tuple[str, ...] = ("doctor", "nurse", "pharm", "dentist")
ROLE_KO = {"doctor": "의사", "nurse": "간호사", "pharm": "약사", "dentist": "치과의사"}
SPLIT_MAP = {"train": "train", "selection_dev": "dev", "final_holdout": "test"}
CHOICE_KEYS = ("A", "B", "C", "D", "E")
N_CHOICES = len(CHOICE_KEYS)

DEFAULT_USER_TEMPLATE = (
    "다음은 {role_ko} 국가시험 문제입니다. 보기 중 정답의 번호 하나만 숫자로 답하세요.\n\n"
    "{question}\n{choices}"
)


# --------------------------------------------------------------------------- #
# Examples + prompt rendering (pure)
# --------------------------------------------------------------------------- #
@dataclass
class MCQExample:
    id: str
    role: str
    year: int
    question: str
    choices: List[str]
    gold: int                      # 1..5
    period: Optional[int] = None
    q_number: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    # --- compatibility with the KorQuAD QAExample interface used by v2_pipeline ---
    @property
    def answer(self) -> str:       # SFT target text
        return str(self.gold)

    @property
    def answers(self) -> List[str]:
        return [str(self.gold)]

    @property
    def context(self) -> str:      # no passage in MCQA; perplexity is skipped
        return ""

    def user_content(self, template: str = DEFAULT_USER_TEMPLATE) -> str:
        return render_user_content(self, template)


def render_choices(choices: Sequence[str]) -> str:
    return "\n".join(f"{i}. {str(c).strip()}" for i, c in enumerate(choices, start=1))


def render_user_content(ex: MCQExample, template: str = DEFAULT_USER_TEMPLATE) -> str:
    return template.format(role_ko=ROLE_KO.get(ex.role, ex.role), role=ex.role,
                           question=ex.question.strip(), choices=render_choices(ex.choices))


def _gold_from_row(raw: Any, base: int) -> int:
    g = int(raw) - base + 1
    if not 1 <= g <= N_CHOICES:
        raise ValueError(f"answer {raw!r} out of range for base={base}")
    return g


def row_to_example(row: Dict[str, Any], role: str, answer_base: int = 1) -> MCQExample:
    """Convert one HF row. ``answer_base`` is 1 when answers are 1..5 (KorMedMCQA)."""
    choices = [str(row[k]) for k in CHOICE_KEYS]
    year = int(row.get("year") or 0)
    period = row.get("period")
    qn = row.get("q_number")
    ex_id = f"{role}-{year}-{period}-{qn}"
    return MCQExample(id=ex_id, role=role, year=year, question=str(row["question"]),
                      choices=choices, gold=_gold_from_row(row["answer"], answer_base),
                      period=None if period is None else int(period),
                      q_number=None if qn is None else int(qn))


# --------------------------------------------------------------------------- #
# Answer extraction (pure) — validated by planted cases in tests/test_kormedmcqa.py
# --------------------------------------------------------------------------- #
_CIRCLED = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5}
_LETTER = {k: i for i, k in enumerate(CHOICE_KEYS, start=1)}
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
# a standalone 1..5 not glued to other digits or a decimal ("12", "3.5", "0.3" rejected;
# "3.", "3번", "3)" accepted) and not followed by a unit ("5 mg", "3%")
_DIGIT_RE = re.compile(r"(?<![\d.])([1-5])(?!\d|\.\d|\s*(?:%|㎎|㎍|㎖|mg|ml|mL|kg|g\b))")
_LETTER_RE = re.compile(r"(?<![A-Za-z])([A-E])(?![A-Za-z])")


def extract_choice(text: Optional[str]) -> Optional[int]:
    """Return the chosen option 1..5, or None when no unambiguous choice is present.

    Order: circled digit (①..⑤) > standalone digit 1..5 > standalone capital A..E.
    The first match in the (think-stripped) answer wins; everything else is ignored.
    """
    if not text:
        return None
    t = _THINK_RE.sub("", str(text))
    if "</think>" in t:
        t = t.split("</think>")[-1]
    t = t.strip()
    if not t:
        return None
    hits: List[Tuple[int, int]] = []           # (position, choice)
    for ch, v in _CIRCLED.items():
        i = t.find(ch)
        if i >= 0:
            hits.append((i, v))
    if hits:
        return min(hits)[1]
    m = _DIGIT_RE.search(t)
    if m:
        return int(m.group(1))
    m = _LETTER_RE.search(t)
    if m:
        return _LETTER[m.group(1)]
    return None


# --------------------------------------------------------------------------- #
# Scoring + paired comparison (pure)
# --------------------------------------------------------------------------- #
def score(preds: Sequence[Optional[int]], examples: Sequence[MCQExample],
          raw: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Accuracy (percent) overall, by role and by year, plus per-item records."""
    if len(preds) != len(examples):
        raise ValueError("preds and examples differ in length")
    n = len(examples)
    correct = [int(p is not None and p == ex.gold) for p, ex in zip(preds, examples)]
    by_role: Dict[str, List[int]] = defaultdict(list)
    by_year: Dict[int, List[int]] = defaultdict(list)
    for c, ex in zip(correct, examples):
        by_role[ex.role].append(c)
        by_year[ex.year].append(c)

    def pct(v: List[int]) -> float:
        return round(100.0 * sum(v) / len(v), 3) if v else float("nan")

    items = []
    for i, (p, ex, c) in enumerate(zip(preds, examples, correct)):
        rec = {"id": ex.id, "role": ex.role, "year": ex.year, "gold": ex.gold,
               "pred": p, "correct": c}
        if raw is not None:
            rec["raw"] = str(raw[i])[:40]
        items.append(rec)
    return {
        "accuracy": pct(correct), "n": n, "n_correct": int(sum(correct)),
        "n_unparsed": int(sum(1 for p in preds if p is None)),
        "accuracy_ci95": wilson_ci(int(sum(correct)), n),
        "by_role": {r: {"accuracy": pct(v), "n": len(v)} for r, v in sorted(by_role.items())},
        "by_year": {str(y): {"accuracy": pct(v), "n": len(v)} for y, v in sorted(by_year.items())},
        "items": items,
    }


def wilson_ci(k: int, n: int, z: float = 1.96) -> Optional[List[float]]:
    """95% Wilson score interval for k/n, in percent."""
    if n <= 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(100 * (centre - half), 3), round(100 * (centre + half), 3)]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b (A right, B wrong)
    and c (A wrong, B right)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def paired_compare(items_a: Iterable[Dict[str, Any]],
                   items_b: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Pair two runs' per-item records by id (same questions) and test the difference."""
    a = {r["id"]: int(r["correct"]) for r in items_a}
    b = {r["id"]: int(r["correct"]) for r in items_b}
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError("no common item ids")
    both = sum(1 for i in common if a[i] and b[i])
    only_a = sum(1 for i in common if a[i] and not b[i])
    only_b = sum(1 for i in common if b[i] and not a[i])
    neither = len(common) - both - only_a - only_b
    n = len(common)
    return {"n_paired": n, "missing_in_a": len(set(b) - set(a)), "missing_in_b": len(set(a) - set(b)),
            "acc_a": round(100.0 * (both + only_a) / n, 3), "acc_b": round(100.0 * (both + only_b) / n, 3),
            "both_correct": both, "only_a_correct": only_a, "only_b_correct": only_b,
            "both_wrong": neither, "mcnemar_p": round(mcnemar_exact(only_a, only_b), 6)}


# --------------------------------------------------------------------------- #
# Splits / subsets (pure) + leakage guard
# --------------------------------------------------------------------------- #
class SplitLeakageError(Exception):
    """Raised when train/selection/holdout share questions or exam years."""


def stratified_subset(examples: Sequence[MCQExample], n: int, seed: int) -> List[MCQExample]:
    """Deterministic subset of size ~n keeping the role mix (for smoke / HF-generate path)."""
    if n >= len(examples):
        return list(examples)
    by_role: Dict[str, List[MCQExample]] = defaultdict(list)
    for ex in sorted(examples, key=lambda e: e.id):
        by_role[ex.role].append(ex)
    rng = random.Random(seed)
    total = len(examples)
    quotas = {r: max(1, round(n * len(v) / total)) for r, v in by_role.items()}
    while sum(quotas.values()) > n:           # rounding can overshoot by a few
        r = max(quotas, key=lambda k: quotas[k])
        quotas[r] -= 1
    out: List[MCQExample] = []
    for r in sorted(by_role):
        pool = list(by_role[r])
        rng.shuffle(pool)
        out.extend(pool[:quotas[r]])
    return sorted(out, key=lambda e: e.id)


def assert_disjoint(splits: Dict[str, Sequence[MCQExample]]) -> Dict[str, Any]:
    """Fail when two named splits share an item id or an exam year; return a manifest."""
    names = list(splits)
    ids = {k: {e.id for e in v} for k, v in splits.items()}
    years = {k: sorted({e.year for e in v}) for k, v in splits.items()}
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            if ids[x] & ids[y]:
                raise SplitLeakageError(f"{x} and {y} share {len(ids[x] & ids[y])} item ids")
            if set(years[x]) & set(years[y]):
                raise SplitLeakageError(f"{x} and {y} share exam years {sorted(set(years[x]) & set(years[y]))}")
    return {k: {"n": len(v), "years": years[k],
                "roles": dict(sorted(Counter(e.role for e in v).items()))} for k, v in splits.items()}


# --------------------------------------------------------------------------- #
# Training-side augmentation (pure): option-order views + role weights
# Uses only the gold-labelled exam items themselves; no other model's outputs.
# --------------------------------------------------------------------------- #
_NUM_LEAD = re.compile(r"^\s*[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def numeric_sorted(choices: Sequence[str]) -> bool:
    """True when every option starts with a number and the numbers are sorted (the exam's
    ascending/descending convention). 65/3,401 train items; their order is kept as is."""
    vals = []
    for c in choices:
        m = _NUM_LEAD.match(str(c))
        if not m:
            return False
        vals.append(float(m.group(0).replace(",", "")))
    return vals == sorted(vals) or vals == sorted(vals, reverse=True)


def _keyed_rng(seed: int, key: str) -> random.Random:
    import hashlib

    return random.Random(int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:16], 16))


def permute_example(ex: MCQExample, perm: Sequence[int], tag: str) -> MCQExample:
    """``perm[i]`` = 0-based index of the ORIGINAL option shown at new position i."""
    import dataclasses

    choices = [ex.choices[j] for j in perm]
    gold = list(perm).index(ex.gold - 1) + 1
    return dataclasses.replace(ex, id=f"{ex.id}#{tag}", choices=choices, gold=gold,
                               meta={**ex.meta, "perm": list(perm)})


def option_views(ex: MCQExample, k: int, seed: int, keep_identity: bool = True,
                 skip_numeric_sorted: bool = True, at_least: int = 0) -> List[MCQExample]:
    """Up to ``max(k, at_least)`` distinct option orders of one item (view 0 = original when
    keep_identity). Orders are cyclic shifts of one random base order, so across 5 views the gold
    option sits at every position once; more views add random distinct orders. Items with sorted
    numeric options keep only the original order. k <= 1 means no shuffling (original only)."""
    n = len(ex.choices)
    if k <= 1 or (skip_numeric_sorted and numeric_sorted(ex.choices)):
        return [permute_example(ex, tuple(range(n)), "p0")]
    rng = _keyed_rng(seed, ex.id)
    base = list(range(n))
    rng.shuffle(base)
    shifts = list(range(n))
    rng.shuffle(shifts)
    cands = [tuple(base[(i + s) % n] for i in range(n)) for s in shifts]
    ident = tuple(range(n))
    if keep_identity:
        cands = [ident] + [p for p in cands if p != ident]
    perms: List[Tuple[int, ...]] = []
    for p in cands:
        if p not in perms:
            perms.append(p)
    want = min(max(k, at_least), math.factorial(n))
    while len(perms) < want:
        p = tuple(rng.sample(range(n), n))
        if p not in perms:
            perms.append(p)
    return [permute_example(ex, p, f"p{i}") for i, p in enumerate(perms[:want])]


def expand_train(examples: Sequence[Any], aug: Dict[str, Any], epochs: float,
                 role_weights: Optional[Dict[str, float]], seed: int,
                 extra_per_epoch: Optional[Sequence[Any]] = None) -> List[Any]:
    """Materialize every training view of the run in EPOCH BLOCKS (the Trainer then reads them in
    order, one pass). Block e holds, for each item, the views assigned to epoch e, plus
    ``extra_per_epoch`` (e.g. explanation examples), shuffled within the block only. So a
    checkpoint saved after block e has seen every item e+1 times, as with plain epochs.

    Item i gets round(epochs * role_weight[role]) views spread evenly over the epochs. With option
    shuffling (k > 1) every view of an item is a different option order (no order repeats)."""
    k = int((aug or {}).get("option_permutations", 1) or 1)
    keep_id = bool((aug or {}).get("keep_identity", True))
    skip_num = bool((aug or {}).get("skip_numeric_sorted", True))
    aseed = int((aug or {}).get("seed", seed))
    n_ep = int(round(float(epochs)))
    if n_ep < 1 or abs(float(epochs) - n_ep) > 1e-9:
        raise ValueError(f"expand_train needs a whole number of epochs, got {epochs}")
    blocks: List[List[Any]] = [[] for _ in range(n_ep)]
    for ex in examples:
        w = float((role_weights or {}).get(ex.role, 1.0))
        n_views = max(1, int(round(n_ep * w)))
        views = option_views(ex, k, aseed, keep_id, skip_num, at_least=n_views if k > 1 else 0)
        # role weights: an item with fewer (or more) views than epochs gets a keyed block offset, so
        # no block loses (or doubles up on) a whole role; with n_views == n_ep this is a no-op
        off = _keyed_rng(aseed, f"blocks:{ex.id}").randrange(n_ep) if n_views != n_ep else 0
        for v in range(n_views):
            blocks[(min(n_ep - 1, v * n_ep // n_views) + off) % n_ep].append(views[v % len(views)])
    for e, blk in enumerate(blocks):
        blk.extend(extra_per_epoch or [])
        _keyed_rng(aseed, f"order-epoch{e}").shuffle(blk)
    return [x for blk in blocks for x in blk]


# --------------------------------------------------------------------------- #
# Explanation targets (method 3: Mi:dm writes its own explanations; no other model's output)
# --------------------------------------------------------------------------- #
EXPLAIN_SYSTEM = ("당신은 한국 보건의료 국가시험 문제를 푸는 어시스턴트입니다. 먼저 풀이를 짧게 쓰고, "
                  "마지막 줄에 '정답: 번호' 형식으로 정답 번호 하나를 쓰세요.")
EXPLAIN_SUFFIX = "\n\n먼저 풀이를 3~6문장으로 쓰고, 마지막 줄에 '정답: 번호' 형식으로 답하세요."
_FINAL_RE = re.compile(r"\*{0,2}정답\*{0,2}\s*(?:은|는|:|：)?\s*(?:번호\s*)?(?:은|는|:|：)?\s*\*{0,2}"
                       r"([①②③④⑤]|[1-5])(?!\d|\.\d)")
_ANSWER_ONLY_SENTENCE = re.compile(r"\s*보기 중 정답의 번호 하나만 숫자로 답하세요\.")


def final_answer_hit(text: Optional[str]) -> Optional[int]:
    """The option in the LAST '정답: N' line ('정답은 N', '**정답**: N.', circled digits), or None."""
    if not text:
        return None
    hits = list(_FINAL_RE.finditer(_THINK_RE.sub("", str(text))))
    if not hits:
        return None
    g = hits[-1].group(1)
    return _CIRCLED.get(g, None) or int(g)


def extract_final_choice(text: Optional[str]) -> Optional[int]:
    """The LAST '정답: N' in an explain-then-answer output; if there is none, fall back to
    extract_choice on the last non-empty line (an answer-only model replying '3' still reads)."""
    hit = final_answer_hit(text)
    if hit is not None:
        return hit
    if not text:
        return None
    lines = [ln for ln in _THINK_RE.sub("", str(text)).strip().splitlines() if ln.strip()]
    return extract_choice(lines[-1]) if lines else None


@dataclass
class ExplainExample:
    """One explanation-training example built from an MCQExample. The prompt uses the explain
    system prompt and asks for a short explanation; the target is '<explanation>\n정답: N'."""
    base: MCQExample
    explanation: str
    id: str = ""

    def __post_init__(self):
        self.id = self.id or f"{self.base.id}#explain"

    role = property(lambda self: self.base.role)
    year = property(lambda self: self.base.year)
    gold = property(lambda self: self.base.gold)
    system_override = property(lambda self: EXPLAIN_SYSTEM)

    @property
    def answer(self) -> str:
        return f"{self.explanation.strip()}\n정답: {self.base.gold}"

    @property
    def context(self) -> str:
        return ""

    def user_content(self, template: str = DEFAULT_USER_TEMPLATE) -> str:
        # the answer-only instruction ("번호 하나만 숫자로") would contradict the explain request
        return render_user_content(self.base, _ANSWER_ONLY_SENTENCE.sub("", template)) + EXPLAIN_SUFFIX


def explain_prompt_example(ex: MCQExample) -> "ExplainExample":
    """For inference in explain mode: same prompt as training, empty explanation."""
    return ExplainExample(base=ex, explanation="")


def load_explanations(path: str) -> Dict[str, str]:
    """{item id: explanation} from the JSONL written by quantization.self_explain (kept out of git)."""
    out: Dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if r.get("keep"):
                    out[r["id"]] = r["explanation"]
    return out


# --------------------------------------------------------------------------- #
# Loading (lazy datasets import)
# --------------------------------------------------------------------------- #
def _detect_answer_base(rows: Iterable[Dict[str, Any]]) -> int:
    vals = {int(r["answer"]) for r in rows}
    if vals and min(vals) >= 1 and max(vals) <= N_CHOICES:
        return 1
    if vals and min(vals) >= 0 and max(vals) <= N_CHOICES - 1:
        return 0
    raise ValueError(f"unexpected answer values {sorted(vals)[:10]}")


def load_split(split: str, roles: Sequence[str] = ROLES, dataset: str = DATASET,
               revision: Optional[str] = None) -> List[MCQExample]:
    """Load one named split (train / selection_dev / final_holdout) across roles."""
    from datasets import load_dataset

    hf_split = SPLIT_MAP.get(split, split)
    out: List[MCQExample] = []
    for role in roles:
        ds = load_dataset(dataset, role, split=hf_split, revision=revision)
        rows = list(ds)
        base = _detect_answer_base(rows)
        out.extend(row_to_example(r, role, base) for r in rows)
    return sorted(out, key=lambda e: e.id)


def load_slices(cfg: Dict[str, Any], train_seed: Optional[int] = None,
                split: str = "final_holdout") -> Dict[str, Any]:
    """Same return shape as v2_pipeline.load_slices (train/eval/fewshot/...)."""
    dcfg = cfg["data"]
    roles = tuple(dcfg.get("roles") or ROLES)
    revision = dcfg.get("revision")
    eval_seed = int(dcfg.get("seed", 42))
    tseed = int(train_seed if train_seed is not None else eval_seed)

    # data.train_splits: [train] (default). [train, selection_dev] is the optional post-selection
    # refit; assert_disjoint then refuses to evaluate that config on selection_dev (shared ids/year).
    train_splits = list(dcfg.get("train_splits") or ["train"])
    if "final_holdout" in train_splits:
        raise SplitLeakageError("final_holdout can never be a training split")
    train = [e for s in train_splits for e in load_split(s, roles, dcfg.get("dataset", DATASET), revision)]
    ev = load_split(split, roles, dcfg.get("dataset", DATASET), revision)
    assert_disjoint({"train": train, split: ev})

    subset = dcfg.get("train_subset")
    rng = random.Random(tseed)
    train = list(train)
    rng.shuffle(train)
    if subset:
        train = train[:int(subset)]
    eval_size = dcfg.get("eval_size")
    if eval_size:
        ev = stratified_subset(ev, int(eval_size), eval_seed)
    return {"train": train, "eval": ev, "fewshot": [], "train_seed": tseed,
            "eval_seed": eval_seed, "split": split}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    ap.add_argument("--stats", action="store_true", help="print split sizes, years and roles")
    cmp = sub.add_parser("compare", help="paired McNemar between two eval JSONs (items)")
    cmp.add_argument("a")
    cmp.add_argument("b")
    args = ap.parse_args(argv)

    if args.cmd == "compare":
        A = json.load(open(args.a, encoding="utf-8"))
        B = json.load(open(args.b, encoding="utf-8"))
        res = paired_compare(A["items"], B["items"])
        res.update({"a": args.a, "b": args.b})
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0
    if args.stats:
        splits = {k: load_split(k) for k in ("train", "selection_dev", "final_holdout")}
        print(json.dumps(assert_disjoint(splits), ensure_ascii=False, indent=2))
        ex = splits["final_holdout"][0]
        print("\n----- example user content -----")
        print(render_user_content(ex))
        print("----- gold:", ex.gold)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
