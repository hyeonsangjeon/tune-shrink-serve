"""v2 pipeline — chat-template, completion-only, multi-seed 3-way quantization run.

This module layers the v2 overhaul on top of the existing quantization/ code. It fixes
the five v1 weaknesses at their root:

  * W1  chat-template prompts (not the raw text template) for BOTH training and eval, so
        the strong instruct base is measured/used correctly (v1's zero-shot artifact).
  * W2  completion-only loss (loss ONLY on the answer tokens) + full-epoch A + long QAT,
        so fine-tuning actually teaches the task (v1 spent <5% of loss on the answer).
  * W5  multi-seed training -> mean +/- std on a large (2k) fixed held-out slice.

The INT4 recipe (TorchAO tile-packed int4 weight-only, group 128) and the KorQuAD official
EM/F1 metric are reused unchanged from ``eval_qa`` so B and C share an identical serving
format and the numbers stay comparable to the metric definition.

Everything is driven by ``config.yaml``. Per-seed artifacts live under
``artifacts/<Method>_seed<seed>/``.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .data_korquad import DEFAULT_CONFIG, QAExample, load_config
from . import eval_qa as E

# --------------------------------------------------------------------------- #
# Prompting (chat template) + answer extraction
# --------------------------------------------------------------------------- #
FALLBACK_SYSTEM = ("당신은 한국어 독해 질의응답 어시스턴트입니다. 주어진 문맥에서 질문의 정답을 찾아 "
                   "정답에 해당하는 어구만 간결하게 출력하세요. 부연 설명이나 문장 없이 정답만 쓰세요.")


def system_prompt(cfg: Dict[str, Any]) -> str:
    return (cfg.get("data", {}).get("chat", {}) or {}).get("system", FALLBACK_SYSTEM).strip()


def enable_thinking_flag(cfg: Dict[str, Any]) -> bool:
    return bool((cfg.get("data", {}).get("chat", {}) or {}).get("enable_thinking", False))


def build_chat_prompt(tokenizer, system: str, context: str, question: str,
                      fewshot: Optional[List] = None, enable_thinking: bool = False,
                      add_generation_prompt: bool = True) -> str:
    """Render a chat prompt string via the tokenizer's chat template.

    ``fewshot`` is a list of (context, question, answer) exemplars (train split; used only
    for the base-select few-shot column — the fine-tuned models are eval'd zero-shot).
    Qwen3 accepts ``enable_thinking``; other templates raise TypeError -> retry without it.
    """
    msgs = [{"role": "system", "content": system}]
    for (c, q, a) in (fewshot or []):
        msgs.append({"role": "user", "content": f"[문맥]\n{c}\n\n[질문]\n{q}"})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": f"[문맥]\n{context}\n\n[질문]\n{question}"})
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking)
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt)


def extract_answer(text: str) -> str:
    """Keep the assistant's answer only: drop any <think> block, strip labels, first line."""
    t = text.strip()
    if "</think>" in t:
        t = t.split("</think>")[-1].strip()
    for lead in ("답:", "정답:", "답변:", "[답]", "답)", "A:"):
        if t.startswith(lead):
            t = t[len(lead):].strip()
    return t.split("\n")[0].strip()


# --------------------------------------------------------------------------- #
# Task-agnostic chat settings (KorQuAD extractive QA or KorMedMCQA multiple choice)
# --------------------------------------------------------------------------- #
def task_name(cfg: Dict[str, Any]) -> str:
    return str((cfg.get("data", {}) or {}).get("task", "korquad")).lower()


def chat_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Everything prompt/tokenizer related, read from ``data.chat`` (all keys optional).

    * ``template_kwargs``  extra variables for the chat template (Mi:dm: a FIXED
      ``date_string`` so the auto-inserted "today" line never changes between runs)
    * ``user_template``    MCQA user-turn template (see kormedmcqa.DEFAULT_USER_TEMPLATE)
    * ``answer_end_token`` token appended to the SFT target (Mi:dm ends assistant turns
      with ``<|eot_id|>``, not its eos ``<|end_of_text|>``); default = tokenizer eos
    * ``stop_tokens``      extra stop tokens for generation (default = answer_end + eos)
    * ``pad_token``        dedicated pad token when the vocab has one
    """
    chat = (cfg.get("data", {}) or {}).get("chat", {}) or {}
    return {"system": system_prompt(cfg), "enable_thinking": enable_thinking_flag(cfg),
            "template_kwargs": dict(chat.get("template_kwargs") or {}),
            "user_template": chat.get("user_template"),
            "answer_end_token": chat.get("answer_end_token"),
            "stop_tokens": list(chat.get("stop_tokens") or []),
            "pad_token": chat.get("pad_token")}


def resolve_token_id(tokenizer, token: Optional[str]) -> Optional[int]:
    """Id of a special token string, or None when the vocab does not have it."""
    if not token:
        return None
    try:
        tid = tokenizer.convert_tokens_to_ids(token)
    except Exception:
        return None
    unk = getattr(tokenizer, "unk_token_id", None)
    if tid is None or (unk is not None and tid == unk and token != getattr(tokenizer, "unk_token", None)):
        return None
    return int(tid)


def prepare_tokenizer(tokenizer, settings: Dict[str, Any]):
    """Pick the pad token: the configured dedicated one if present, else eos (legacy)."""
    pad = settings.get("pad_token")
    if pad and resolve_token_id(tokenizer, pad) is not None:
        tokenizer.pad_token = pad
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def answer_end_id(tokenizer, settings: Dict[str, Any]) -> Optional[int]:
    tid = resolve_token_id(tokenizer, settings.get("answer_end_token"))
    return tid if tid is not None else tokenizer.eos_token_id


def stop_token_ids(tokenizer, settings: Dict[str, Any]) -> List[int]:
    ids: List[int] = []
    for t in [settings.get("answer_end_token")] + list(settings.get("stop_tokens") or []):
        tid = resolve_token_id(tokenizer, t)
        if tid is not None and tid not in ids:
            ids.append(tid)
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in ids:
        ids.append(int(tokenizer.eos_token_id))
    return ids


def user_content_for(ex, settings: Dict[str, Any]) -> str:
    if hasattr(ex, "user_content"):               # MCQExample (KorMedMCQA)
        tmpl = settings.get("user_template")
        return ex.user_content(tmpl) if tmpl else ex.user_content()
    return f"[문맥]\n{ex.context}\n\n[질문]\n{ex.question}"   # KorQuAD (same as build_chat_prompt)


def render_prompt(tokenizer, ex, settings: Dict[str, Any], add_generation_prompt: bool = True) -> str:
    """Chat-template prompt for one example. For KorQuAD with no template kwargs this is
    byte-identical to ``build_chat_prompt`` (the historical path)."""
    system = getattr(ex, "system_override", None) or settings["system"]
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": user_content_for(ex, settings)}]
    kw = dict(settings.get("template_kwargs") or {})
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt,
            enable_thinking=settings.get("enable_thinking", False), **kw)
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt, **kw)


def _default_settings(system: str, enable_thinking: bool) -> Dict[str, Any]:
    return {"system": system, "enable_thinking": enable_thinking, "template_kwargs": {},
            "user_template": None, "answer_end_token": None, "stop_tokens": [], "pad_token": None}


# --------------------------------------------------------------------------- #
# Data slices (multi-seed): eval held-out FIXED (data.seed); train shuffled by train_seed
# --------------------------------------------------------------------------- #
def load_slices(cfg: Dict[str, Any], train_seed: Optional[int] = None,
                n_fewshot: int = 0, split: str = "final_holdout") -> Dict[str, Any]:
    if task_name(cfg) == "kormedmcqa":
        from . import kormedmcqa as K
        return K.load_slices(cfg, train_seed=train_seed, split=split)

    from datasets import load_dataset
    from . import splits as S

    dcfg = cfg["data"]
    S.assert_config_splits_disjoint(cfg)   # P1-1: fail fast if selection/holdout overlap
    eval_seed = int(dcfg.get("seed", 42))
    tseed = int(train_seed if train_seed is not None else eval_seed)
    ds = load_dataset(dcfg["dataset"])

    # P1-1: eval draws from the named DISJOINT slice of the fixed shuffle ordering
    # (final_holdout = [800:1800] by default; selection_dev = [0:800] for tuning).
    val = ds["validation"].shuffle(seed=eval_seed)
    start, end = S.split_bounds(cfg, split)
    size = end - start
    eval_size = dcfg.get("eval_size")          # smoke override shrinks this -> tiny slice
    if eval_size:
        size = min(size, int(eval_size))
    val = val.select(range(start, min(start + size, len(val))))

    train = ds["train"].shuffle(seed=tseed)
    subset = dcfg.get("train_subset")
    if subset:
        train = train.select(range(min(int(subset), len(train))))

    def to_ex(r):
        answers = [a for a in (r["answers"]["text"] if r.get("answers") else []) if a and a.strip()] or [""]
        return QAExample(id=str(r.get("id", "")), prompt="", answer=answers[0],
                         answers=answers, context=r["context"], question=r["question"])

    train_ex = [to_ex(r) for r in train]
    eval_ex = [to_ex(r) for r in val]
    fewshot = [(train[i]["context"], train[i]["question"], train[i]["answers"]["text"][0])
               for i in range(n_fewshot)] if n_fewshot else []
    return {"train": train_ex, "eval": eval_ex, "fewshot": fewshot,
            "train_seed": tseed, "eval_seed": eval_seed, "split": split}


# --------------------------------------------------------------------------- #
# Completion-only tokenization (loss ONLY on the answer tokens) + collator
# --------------------------------------------------------------------------- #
def _user_token_mask(tokenizer, prompt: str, user: str, n_ids: int) -> Optional[List[int]]:
    """1 for prompt tokens that fall inside the user turn (question + options), else 0.
    The fixed system block and date line are never included. None when the user text cannot
    be located in the rendered prompt (the example then gets no prompt loss)."""
    start = prompt.find(user)
    if start < 0:
        return None
    end = start + len(user)
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    if len(offs) != n_ids:
        return None
    return [1 if (s >= start and e <= end and e > s) else 0 for (s, e) in offs]


def tokenize_completion_only(tokenizer, examples: List[QAExample], system: str,
                             max_len: int, enable_thinking: bool = False,
                             settings: Optional[Dict[str, Any]] = None,
                             with_user_mask: bool = False):
    from datasets import Dataset

    st = settings or _default_settings(system, enable_thinking)
    eos = answer_end_id(tokenizer, st)          # Mi:dm: <|eot_id|>; legacy: tokenizer eos
    input_ids_all, labels_all, umask_all = [], [], []
    n_trunc, max_seen, n_nomask = 0, 0, 0
    for ex in examples:
        prompt = render_prompt(tokenizer, ex, st, add_generation_prompt=True)
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tokenizer(ex.answer, add_special_tokens=False)["input_ids"]
        if eos is not None:
            a_ids = a_ids + [eos]
        input_ids = p_ids + a_ids
        labels = [-100] * len(p_ids) + list(a_ids)
        umask = None
        if with_user_mask:
            pm = _user_token_mask(tokenizer, prompt, user_content_for(ex, st), len(p_ids))
            n_nomask += pm is None
            umask = (pm or [0] * len(p_ids)) + [0] * len(a_ids)
        max_seen = max(max_seen, len(input_ids))
        if len(input_ids) > max_len:            # rare: left-truncate (keep the answer intact)
            n_trunc += 1
            input_ids = input_ids[-max_len:]
            labels = labels[-max_len:]
            umask = umask[-max_len:] if umask is not None else None
        input_ids_all.append(input_ids)
        labels_all.append(labels)
        if umask is not None:
            umask_all.append(umask)
    if n_trunc:
        # Truncation drops the start of the system prompt, so train and eval prompts differ.
        print(f"[tokenize] WARNING {n_trunc}/{len(examples)} examples exceed max_seq_len={max_len} "
              f"(longest {max_seen}); raise data.max_seq_len")
    cols = {"input_ids": input_ids_all, "labels": labels_all}
    if with_user_mask:
        if n_nomask:
            print(f"[tokenize] WARNING user turn not located in {n_nomask}/{len(examples)} prompts "
                  f"(no prompt loss for those)")
        cols["user_mask"] = umask_all
    return Dataset.from_dict(cols)


@dataclass
class DataCollatorCompletionOnly:
    pad_token_id: int
    label_pad: int = -100

    def __call__(self, feats):
        import torch

        maxlen = max(len(f["input_ids"]) for f in feats)
        ids, labs, attn, um = [], [], [], []
        has_um = "user_mask" in feats[0]
        for f in feats:
            n = len(f["input_ids"])
            pad = maxlen - n
            ids.append(list(f["input_ids"]) + [self.pad_token_id] * pad)
            labs.append(list(f["labels"]) + [self.label_pad] * pad)
            attn.append([1] * n + [0] * pad)
            if has_um:
                um.append(list(f["user_mask"]) + [0] * pad)
        out = {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(attn),
               "labels": torch.tensor(labs)}
        if has_um:
            out["user_mask"] = torch.tensor(um)
        return out


# --------------------------------------------------------------------------- #
# Recipe knobs for the no-distillation sweep (all default to today's behaviour)
# --------------------------------------------------------------------------- #
def custom_loss_on(tcfg: Dict[str, Any]) -> bool:
    return bool(float(tcfg.get("prompt_loss_weight", 0) or 0) > 0
                or float(tcfg.get("option_label_smoothing", 0) or 0) > 0
                or bool(tcfg.get("loss_per_example", False)))


def extra_train_args(tcfg: Dict[str, Any]) -> Dict[str, Any]:
    kw: Dict[str, Any] = {"save_total_limit": tcfg.get("save_total_limit", 2),
                          "lr_scheduler_type": tcfg.get("lr_scheduler_type", "linear")}
    if tcfg.get("neftune_noise_alpha"):
        kw["neftune_noise_alpha"] = float(tcfg["neftune_noise_alpha"])
    if custom_loss_on(tcfg):
        kw["remove_unused_columns"] = False     # keep user_mask in the batch
    return kw


def train_examples_for_run(cfg: Dict[str, Any], data: Dict[str, Any], seed: int):
    """(examples, trainer_epochs, info). With no data.augment.option_permutations > 1, no
    data.role_weights and no data.self_explain this returns the train split and train.epochs
    unchanged (today's run, trainer shuffles each epoch). Otherwise every view of the run is
    materialized in epoch blocks (kormedmcqa.expand_train) and the Trainer reads them IN ORDER
    in one pass (``info["sequential"]`` is True; see make_trainer)."""
    dcfg, tcfg = cfg["data"], cfg["train"]
    aug = dict(dcfg.get("augment") or {})
    rw = dcfg.get("role_weights") or None
    k = int(aug.get("option_permutations", 1) or 1)
    se = dict(dcfg.get("self_explain") or {})
    if task_name(cfg) != "kormedmcqa" or (k <= 1 and not rw and not se.get("path")):
        return data["train"], float(tcfg["epochs"]), None
    from collections import Counter
    from . import kormedmcqa as K

    extra = []
    if se.get("path"):
        expl = K.load_explanations(se["path"])
        extra = [K.ExplainExample(base=e, explanation=expl[e.id]) for e in data["train"] if e.id in expl]
        if not extra:
            raise ValueError(f"no usable explanations in {se['path']}")
    views = K.expand_train(data["train"], aug, float(tcfg["epochs"]), rw, seed, extra_per_epoch=extra)
    answer_views = [v for v in views if not isinstance(v, K.ExplainExample)]
    gold = Counter(v.gold for v in answer_views)
    info = {"option_permutations": k, "keep_identity": bool(aug.get("keep_identity", True)),
            "skip_numeric_sorted": bool(aug.get("skip_numeric_sorted", True)),
            "role_weights": rw, "epochs_equivalent": float(tcfg["epochs"]), "n_views": len(views),
            "n_explain_items": len(extra), "explain_path": se.get("path"), "sequential": True,
            "views_by_role": dict(sorted(Counter(v.role for v in answer_views).items())),
            "gold_position_share": {str(p): round(100.0 * c / len(answer_views), 2)
                                    for p, c in sorted(gold.items())}}
    return views, 1.0, info


def save_final_adapter(trainer, run_dir: str) -> str:
    """Always keep the LAST step as an adapter checkpoint (save_steps need not divide the step
    count), so the fully-trained adapter can be scored on 2021 like the others."""
    path = os.path.join(run_dir, f"checkpoint-{int(trainer.state.global_step)}")
    if not os.path.isfile(os.path.join(path, "adapter_config.json")):
        trainer.save_model(path)
    return path


def recipe_record(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The knobs a sweep arm may change, copied into A_train_seed<S>.json."""
    t, l = cfg["train"], cfg["lora"]
    return {"learning_rate": float(t["learning_rate"]), "epochs": float(t["epochs"]),
            "max_seq_len": int(cfg["data"]["max_seq_len"]),
            "lr_scheduler_type": t.get("lr_scheduler_type", "linear"),
            "lora_r": int(l["r"]), "lora_alpha": int(l["alpha"]),
            "use_rslora": bool(l.get("use_rslora", False)),
            "neftune_noise_alpha": t.get("neftune_noise_alpha"),
            "loss_per_example": bool(t.get("loss_per_example", False)),
            "prompt_loss_weight": float(t.get("prompt_loss_weight", 0) or 0),
            "option_label_smoothing": float(t.get("option_label_smoothing", 0) or 0),
            "save_steps": int(t.get("save_steps", 500)),
            "save_total_limit": t.get("save_total_limit", 2)}


def _sequential(cls):
    """Trainer subclass that reads the training set in order (views are pre-shuffled per epoch)."""
    from torch.utils.data import SequentialSampler

    class SequentialTrainer(cls):
        def _get_train_sampler(self, *args, **kwargs):
            return SequentialSampler(self.train_dataset)

    return SequentialTrainer


def make_trainer(tcfg: Dict[str, Any], tok, sequential: bool = False, **trainer_kwargs):
    """Plain HF Trainer unless prompt_loss_weight / option_label_smoothing is set. ``sequential``:
    no shuffling by the Trainer (expand_train already shuffled within each epoch block)."""
    from transformers import Trainer

    if not custom_loss_on(tcfg):
        return (_sequential(Trainer) if sequential else Trainer)(**trainer_kwargs)

    # same tokenization as the SFT target (tokenize_completion_only: tokenizer(ex.answer))
    digit_ids = []
    for d in range(1, 6):
        ids = tok(str(d), add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, f"answer digit {d} is {len(ids)} tokens; option smoothing assumes 1"
        digit_ids.append(int(ids[0]))

    class WeightedCompletionTrainer(Trainer):
        """loss = CE(answer tokens; the digit's target is (1-eps)*gold + eps*uniform over the
        5 digit tokens) + plw * mean CE(user-turn tokens). Needs real logits (Unsloth: set
        UNSLOTH_RETURN_LOGITS=1 before importing unsloth)."""

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.model_accepts_loss_kwargs = False   # our loss is a per-micro-batch mean
            self.plw = float(tcfg.get("prompt_loss_weight", 0) or 0)
            self.eps = float(tcfg.get("option_label_smoothing", 0) or 0)
            self.per_example = bool(tcfg.get("loss_per_example", False))

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            import torch
            import torch.nn.functional as F

            labels = inputs.pop("labels")
            umask = inputs.pop("user_mask", None)
            out = model(**inputs)
            logits = out.logits[:, :-1, :]
            tgt = labels[:, 1:]
            ans = tgt != -100
            first = ans & ~torch.nn.functional.pad(ans[:, :-1], (1, 0), value=False)
            lp_ans = F.log_softmax(logits[ans].float(), dim=-1)
            nll = -lp_ans.gather(-1, tgt[ans].unsqueeze(-1)).squeeze(-1)
            if self.eps > 0:
                is_first = first[ans]
                d = torch.tensor(digit_ids, device=lp_ans.device)
                uni = -lp_ans[:, d].mean(-1)
                nll = torch.where(is_first, (1 - self.eps) * nll + self.eps * uni, nll)
            if self.per_example:
                # mean over each row's target tokens, then over rows: a 2-token answer example
                # weighs as much as a 200-token explanation example
                rows = ans.nonzero(as_tuple=True)[0]
                b = ans.shape[0]
                tot = torch.zeros(b, device=nll.device, dtype=nll.dtype).index_add_(0, rows, nll)
                cnt = torch.zeros(b, device=nll.device, dtype=nll.dtype).index_add_(
                    0, rows, torch.ones_like(nll))
                has = cnt > 0
                loss = (tot[has] / cnt[has]).mean()
            else:
                loss = nll.mean()
            if self.plw > 0 and umask is not None:
                um = umask[:, 1:].bool() & (inputs["attention_mask"][:, 1:] > 0)
                if um.any():
                    ids_next = inputs["input_ids"][:, 1:]
                    lp_u = F.log_softmax(logits[um].float(), dim=-1)
                    loss = loss + self.plw * (-lp_u.gather(-1, ids_next[um].unsqueeze(-1))).mean()
            return (loss, out) if return_outputs else loss

    return (_sequential(WeightedCompletionTrainer) if sequential else WeightedCompletionTrainer)(**trainer_kwargs)


# --------------------------------------------------------------------------- #
# Method A — BF16 LoRA (chat + completion-only, full epochs, per seed)
# --------------------------------------------------------------------------- #
def train_method_a(cfg: Dict[str, Any], seed: int, out_merged: str,
                   resume: bool = False) -> Dict[str, Any]:
    if str(cfg["train"].get("backend", "hf")).lower() == "unsloth":
        # Must run before transformers is imported in this process (v2_run a = own process).
        return _train_method_a_unsloth(cfg, seed, out_merged, resume=resume)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)

    base = cfg["base_model"]["selected"]
    rev = cfg["base_model"].get("revision")
    tcfg, lcfg, dcfg = cfg["train"], cfg["lora"], cfg["data"]
    sys_p = system_prompt(cfg)
    st = chat_settings(cfg)
    max_len = int(dcfg["max_seq_len"])

    tok = prepare_tokenizer(AutoTokenizer.from_pretrained(base, revision=rev), st)

    try:
        model = AutoModelForCausalLM.from_pretrained(base, revision=rev, dtype=torch.bfloat16,
                                                     device_map="cuda")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(base, revision=rev, torch_dtype=torch.bfloat16,
                                                     device_map="cuda")
    model.config.use_cache = False
    model.enable_input_require_grads()
    peft_cfg = LoraConfig(r=int(lcfg["r"]), lora_alpha=int(lcfg["alpha"]),
                          lora_dropout=float(lcfg["dropout"]),
                          target_modules=list(lcfg["target_modules"]),
                          use_rslora=bool(lcfg.get("use_rslora", False)),
                          bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_cfg)

    data = load_slices(cfg, train_seed=seed)
    train_ex, trainer_epochs, aug_info = train_examples_for_run(cfg, data, seed)
    train_ds = tokenize_completion_only(tok, train_ex, sys_p, max_len,
                                        enable_thinking_flag(cfg), settings=st,
                                        with_user_mask=float(tcfg.get("prompt_loss_weight", 0) or 0) > 0)
    run_dir = out_merged + "_run"
    ta_kwargs = dict(
        output_dir=run_dir, per_device_train_batch_size=int(tcfg["per_device_batch_size"]),
        gradient_accumulation_steps=int(tcfg["grad_accum"]), learning_rate=float(tcfg["learning_rate"]),
        warmup_ratio=float(tcfg["warmup_ratio"]), weight_decay=float(tcfg["weight_decay"]),
        logging_steps=int(tcfg["logging_steps"]), num_train_epochs=trainer_epochs,
        max_grad_norm=float(tcfg.get("max_grad_norm", 1.0)), seed=int(seed),
        bf16=True, gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", False)),
        save_steps=int(tcfg.get("save_steps", 500)),
        report_to="none", logging_first_step=True, **extra_train_args(tcfg))
    if tcfg.get("max_steps"):
        ta_kwargs["max_steps"] = int(tcfg["max_steps"])
    args = TrainingArguments(**ta_kwargs)

    trainer = make_trainer(tcfg, tok, sequential=bool(aug_info and aug_info.get("sequential")),
                           model=model, args=args, train_dataset=train_ds,
                           data_collator=DataCollatorCompletionOnly(tok.pad_token_id))
    t0 = time.time()
    ckpt = _last_checkpoint(run_dir) if resume else None
    out = trainer.train(resume_from_checkpoint=ckpt)
    secs = time.time() - t0
    final_ckpt = save_final_adapter(trainer, run_dir)
    peak_gb = round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None

    merged = None
    if tcfg.get("save_merged", True):           # sweep arms: false (keep adapter checkpoints only)
        os.makedirs(out_merged, exist_ok=True)
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(out_merged)
        tok.save_pretrained(out_merged)

    loss_curve = [(l.get("step"), l.get("loss")) for l in trainer.state.log_history if "loss" in l]
    log = {"method": "A_bf16", "base_model": base, "seed": int(seed), "backend": "hf",
           "peak_vram_gb": peak_gb, "task": task_name(cfg), "n_train": len(data["train"]), "epochs": float(tcfg["epochs"]),
           "augment": aug_info, "recipe": recipe_record(cfg), "final_checkpoint": final_ckpt,
           "global_step": int(trainer.state.global_step), "train_seconds": round(secs, 1),
           "train_loss": float(getattr(out, "training_loss", float("nan"))),
           "loss_curve": loss_curve, "merged_dir": out_merged}
    del model, trainer, merged
    torch.cuda.empty_cache()
    return log


def _train_method_a_unsloth(cfg: Dict[str, Any], seed: int, out_merged: str,
                            resume: bool = False) -> Dict[str, Any]:
    """Method A with Unsloth's FastLanguageModel (BF16 LoRA, no 4-bit base).

    Same data, completion-only labels and collator as the hf backend, so only the
    training engine differs. Unsloth patches transformers globally, which is why B/C
    (TorchAO PTQ/QAT) always run in their own processes and never import unsloth.
    GPU-only: validate with ``v2_run a --max-steps 5`` before a full run.
    """
    base = cfg["base_model"]["selected"]
    rev = cfg["base_model"].get("revision")
    tcfg, lcfg, dcfg = cfg["train"], cfg["lora"], cfg["data"]
    external_tok = bool(tcfg.get("unsloth_external_tokenizer"))
    tok_pkl = _pickle_tokenizer_in_subprocess(base, rev) if external_tok else None
    if custom_loss_on(tcfg):
        # Unsloth's fused CE returns no logits in training; the custom loss needs them.
        os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

    import unsloth  # noqa: F401  (import first: it patches transformers/peft)
    from unsloth import FastLanguageModel
    import torch
    from transformers import Trainer, TrainingArguments

    st = chat_settings(cfg)
    max_len = int(dcfg["max_seq_len"])
    if external_tok:
        # Unsloth's GPU-init patches break Mi:dm's tokenizer load; hand Unsloth a tokenizer that
        # plain transformers (same version) loaded, and skip Unsloth's tokenizer loader only.
        import pickle
        import unsloth.models.llama as _unsloth_llama

        with open(tok_pkl, "rb") as fh:
            pre_tok = pickle.load(fh)
        _unsloth_llama.load_correct_tokenizer = lambda *a, **k: pre_tok

    fl_kwargs = dict(model_name=base, max_seq_length=max_len, dtype=torch.bfloat16,
                     load_in_4bit=False)
    if rev:                                     # only pass when pinned (older unsloth lacks it)
        fl_kwargs["revision"] = rev
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model, tok = FastLanguageModel.from_pretrained(**fl_kwargs)
    tok = prepare_tokenizer(tok, st)
    model = FastLanguageModel.get_peft_model(
        model, r=int(lcfg["r"]), lora_alpha=int(lcfg["alpha"]),
        lora_dropout=float(lcfg["dropout"]), target_modules=list(lcfg["target_modules"]),
        bias="none", random_state=int(seed), use_rslora=bool(lcfg.get("use_rslora", False)),
        use_gradient_checkpointing="unsloth" if tcfg.get("gradient_checkpointing") else False)

    data = load_slices(cfg, train_seed=seed)
    train_ex, trainer_epochs, aug_info = train_examples_for_run(cfg, data, seed)
    train_ds = tokenize_completion_only(tok, train_ex, st["system"], max_len,
                                        st["enable_thinking"], settings=st,
                                        with_user_mask=float(tcfg.get("prompt_loss_weight", 0) or 0) > 0)
    run_dir = out_merged + "_run"
    ta_kwargs = dict(
        output_dir=run_dir, per_device_train_batch_size=int(tcfg["per_device_batch_size"]),
        gradient_accumulation_steps=int(tcfg["grad_accum"]), learning_rate=float(tcfg["learning_rate"]),
        warmup_ratio=float(tcfg["warmup_ratio"]), weight_decay=float(tcfg["weight_decay"]),
        logging_steps=int(tcfg["logging_steps"]), num_train_epochs=trainer_epochs,
        max_grad_norm=float(tcfg.get("max_grad_norm", 1.0)), seed=int(seed), bf16=True,
        gradient_checkpointing=False,           # handled by unsloth above
        save_steps=int(tcfg.get("save_steps", 500)),
        report_to="none", logging_first_step=True, **extra_train_args(tcfg))
    if tcfg.get("max_steps"):
        ta_kwargs["max_steps"] = int(tcfg["max_steps"])
    trainer = make_trainer(tcfg, tok, sequential=bool(aug_info and aug_info.get("sequential")),
                           model=model, args=TrainingArguments(**ta_kwargs),
                           train_dataset=train_ds,
                           data_collator=DataCollatorCompletionOnly(tok.pad_token_id))
    t0 = time.time()
    ckpt = _last_checkpoint(run_dir) if resume else None
    out = trainer.train(resume_from_checkpoint=ckpt)
    secs = time.time() - t0
    final_ckpt = save_final_adapter(trainer, run_dir)

    if tcfg.get("save_merged", True):           # sweep arms: false (keep adapter checkpoints only)
        os.makedirs(out_merged, exist_ok=True)
        model.save_pretrained_merged(out_merged, tok, save_method="merged_16bit")
    loss_curve = [(l.get("step"), l.get("loss")) for l in trainer.state.log_history if "loss" in l]
    log = {"method": "A_bf16", "base_model": base, "seed": int(seed), "backend": "unsloth",
           "unsloth_version": getattr(unsloth, "__version__", None),
           "tokenizer_source": "external_pickle" if external_tok else "unsloth",
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
           "task": task_name(cfg),
           "n_train": len(data["train"]), "epochs": float(tcfg["epochs"]),
           "augment": aug_info, "recipe": recipe_record(cfg), "final_checkpoint": final_ckpt,
           "global_step": int(trainer.state.global_step), "train_seconds": round(secs, 1),
           "train_loss": float(getattr(out, "training_loss", float("nan"))),
           "loss_curve": loss_curve, "merged_dir": out_merged}
    del model, trainer
    torch.cuda.empty_cache()
    return log


_TOKENIZER_PICKLE_SNIPPET = (
    "import pickle, sys\n"
    "from transformers import AutoTokenizer\n"
    "tok = AutoTokenizer.from_pretrained(sys.argv[1], revision=sys.argv[2] or None)\n"
    "pickle.dump(tok, open(sys.argv[3], 'wb'))\n")


def _pickle_tokenizer_in_subprocess(base: str, revision: Optional[str]) -> str:
    """Load the base tokenizer with plain transformers in a fresh interpreter (no unsloth
    import, so none of its patches) and return the path of a pickle of it."""
    import subprocess
    import sys
    import tempfile

    out = os.path.join(tempfile.mkdtemp(prefix="tok_"), "tokenizer.pkl")
    subprocess.run([sys.executable, "-c", _TOKENIZER_PICKLE_SNIPPET, base, revision or "", out],
                   check=True)
    return out


def merged_vocab_check(cfg: Dict[str, Any], merged_dir: str) -> Dict[str, Any]:
    """Compare the merged A checkpoint's vocab_size with the pinned base (config
    ``base_model.vocab_size``). Unsloth's tokenizer fixes can add a pad token and resize
    the embeddings; the 5-step smoke run must show ``vocab_ok: true`` before the full run."""
    expected = (cfg.get("base_model", {}) or {}).get("vocab_size")
    got = None
    p = os.path.join(merged_dir, "config.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            c = json.load(fh)
        got = c.get("vocab_size", (c.get("text_config") or {}).get("vocab_size"))
    ok = None if expected is None or got is None else int(got) == int(expected)
    return {"vocab_size_expected": expected, "vocab_size_merged": got, "vocab_ok": ok}


def merge_adapter_checkpoint(cfg: Dict[str, Any], ckpt_dir: str, out_merged: str) -> Dict[str, Any]:
    """Merge one saved LoRA checkpoint (the dev-selected epoch) into the pinned base with plain
    transformers + peft (no unsloth import) and save it where v2_run/eval_vllm expect A_bf16."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = cfg["base_model"]["selected"]
    rev = cfg["base_model"].get("revision")
    tok = prepare_tokenizer(AutoTokenizer.from_pretrained(base, revision=rev), chat_settings(cfg))
    model = AutoModelForCausalLM.from_pretrained(base, revision=rev, dtype=torch.bfloat16,
                                                 device_map="cuda")
    merged = PeftModel.from_pretrained(model, ckpt_dir).merge_and_unload()
    os.makedirs(out_merged, exist_ok=True)
    merged.save_pretrained(out_merged)
    tok.save_pretrained(out_merged)
    del model, merged
    torch.cuda.empty_cache()
    return {"merged_from": ckpt_dir, "merged_dir": out_merged}


def _last_checkpoint(run_dir: str) -> Optional[str]:
    try:
        from transformers.trainer_utils import get_last_checkpoint

        return get_last_checkpoint(run_dir) if os.path.isdir(run_dir) else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Method B — INT4 PTQ (TorchAO tile-packed int4) of a merged A seed
# --------------------------------------------------------------------------- #
def build_method_b(cfg: Dict[str, Any], a_dir: str, b_dir: str) -> Dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

    gs = int(cfg["ptq"]["group_size"])
    tok = AutoTokenizer.from_pretrained(a_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    qmodel = AutoModelForCausalLM.from_pretrained(
        a_dir, dtype=torch.bfloat16, device_map="cuda",
        quantization_config=TorchAoConfig(quant_type=E.make_int4_weightonly_config(gs)))
    os.makedirs(b_dir, exist_ok=True)
    qmodel.save_pretrained(b_dir, safe_serialization=False)
    tok.save_pretrained(b_dir)
    size_gb = E.dir_size_gb(b_dir)
    del qmodel
    torch.cuda.empty_cache()
    return {"method": "B_int4_ptq", "src_A": a_dir, "int4_dir": b_dir,
            "group_size": gs, "size_gb": round(size_gb, 4)}


# --------------------------------------------------------------------------- #
# Method C — INT4 QAT (matched fake-quant STE, full-param, 8-bit Adam) of a merged A seed
# --------------------------------------------------------------------------- #
def train_method_c(cfg: Dict[str, Any], a_dir: str, c_dir: str, seed: int,
                   resume: bool = False) -> Dict[str, Any]:
    import torch
    import torch.nn as nn
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, TorchAoConfig)
    from torchao.quantization import quantize_, Int4WeightOnlyConfig
    from torchao.quantization.qat import QATConfig
    from torchao.quantization.qat.fake_quantize_config import _infer_fake_quantize_configs

    qcfg, dcfg = cfg["qat"], cfg["data"]
    gs = int(qcfg["group_size"])
    sys_p = system_prompt(cfg)
    max_len = int(dcfg["max_seq_len"])
    c_bf16 = c_dir + "_bf16adapt"

    def _not_lmhead(m, fqn):
        return isinstance(m, nn.Linear) and "lm_head" not in fqn

    st = chat_settings(cfg)
    tok = prepare_tokenizer(AutoTokenizer.from_pretrained(a_dir), st)
    model = AutoModelForCausalLM.from_pretrained(a_dir, dtype=torch.bfloat16, device_map="cuda")

    _, wcfg = _infer_fake_quantize_configs(Int4WeightOnlyConfig(group_size=gs))
    quantize_(model, QATConfig(weight_config=wcfg, step="prepare"), filter_fn=_not_lmhead)

    for p in model.parameters():
        p.requires_grad_(False)
    n_train = 0
    for m in model.modules():
        if type(m).__name__ == "FakeQuantizedLinear":
            m.weight.requires_grad_(True)
            n_train += m.weight.numel()
    if qcfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False

    data = load_slices(cfg, train_seed=seed)
    train_ds = tokenize_completion_only(tok, data["train"], sys_p, max_len,
                                        enable_thinking_flag(cfg), settings=st)
    args = TrainingArguments(
        output_dir=c_dir + "_run", per_device_train_batch_size=int(qcfg["per_device_batch_size"]),
        gradient_accumulation_steps=int(qcfg["grad_accum"]), learning_rate=float(qcfg["learning_rate"]),
        max_steps=int(qcfg["max_steps"]), warmup_ratio=0.03, logging_steps=20, seed=int(seed),
        bf16=True, optim=qcfg.get("optim", "adamw_8bit"), gradient_checkpointing=False,
        save_strategy=qcfg.get("save_strategy", "steps"),
        save_steps=int(qcfg.get("save_steps", 150)), save_total_limit=2,
        report_to="none", logging_first_step=True)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      data_collator=DataCollatorCompletionOnly(tok.pad_token_id))
    t0 = time.time()
    ckpt = _last_checkpoint(c_dir + "_run") if resume else None
    out = trainer.train(resume_from_checkpoint=ckpt)
    secs = time.time() - t0
    model = trainer.model

    quantize_(model, QATConfig(step="convert"), filter_fn=_not_lmhead)
    os.makedirs(c_bf16, exist_ok=True)
    model.save_pretrained(c_bf16)
    tok.save_pretrained(c_bf16)
    loss_curve = [(l.get("step"), l.get("loss")) for l in trainer.state.log_history if "loss" in l]
    del model, trainer
    torch.cuda.empty_cache()

    exp = export_method_c(cfg, c_bf16, c_dir)
    return {"method": "C_int4_qat", "src_A": a_dir, "int4_dir": c_dir, "bf16_dir": c_bf16,
            "seed": int(seed), "group_size": gs, "max_steps": int(qcfg["max_steps"]),
            "trainable_millions": round(n_train / 1e6, 1), "train_seconds": round(secs, 1),
            "train_loss": float(getattr(out, "training_loss", float("nan"))),
            "loss_curve": loss_curve, "size_gb": exp["size_gb"]}


def export_method_c(cfg: Dict[str, Any], c_bf16: str, c_dir: str) -> Dict[str, Any]:
    """INT4 (tile-packed) export of C's converted bf16 weights — no training.

    The int4 dirs are saved with ``safe_serialization=False`` (torchao tensor subclasses
    pickled), so they are tied to the exact torch/torchao build that wrote them. When
    training and serving GPUs differ, move only the safetensors bf16 dirs (A merge and
    ``<C>_bf16adapt``) and rebuild B/C int4 on the serving GPU with ``v2_run b`` and
    ``v2_run c --export-only``.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

    assert os.path.isdir(c_bf16), f"missing C bf16 weights: {c_bf16}"
    gs = int(cfg["qat"]["group_size"])
    tok = prepare_tokenizer(AutoTokenizer.from_pretrained(c_bf16), chat_settings(cfg))
    qmodel = AutoModelForCausalLM.from_pretrained(
        c_bf16, dtype=torch.bfloat16, device_map="cuda",
        quantization_config=TorchAoConfig(quant_type=E.make_int4_weightonly_config(gs)))
    os.makedirs(c_dir, exist_ok=True)
    qmodel.save_pretrained(c_dir, safe_serialization=False)
    tok.save_pretrained(c_dir)
    size_gb = E.dir_size_gb(c_dir)
    del qmodel
    torch.cuda.empty_cache()
    return {"method": "C_int4_qat", "src_bf16": c_bf16, "int4_dir": c_dir, "group_size": gs,
            "size_gb": round(size_gb, 4)}


# --------------------------------------------------------------------------- #
# Eval (chat-aware) — reuses the official KorQuAD EM/F1 + perplexity from eval_qa
# --------------------------------------------------------------------------- #
def greedy_generation_config(model, max_new_tokens: int, stop_ids: List[int], pad_id: Optional[int]):
    """Greedy decoding that ignores the checkpoint's sampling defaults (Mi:dm ships
    do_sample=True, T=0.8, top_p=0.7, top_k=20, repetition_penalty=1.05)."""
    import copy

    gc = copy.deepcopy(model.generation_config)
    gc.do_sample = False
    gc.temperature = None
    gc.top_p = None
    gc.top_k = None
    gc.repetition_penalty = 1.0
    gc.max_new_tokens = int(max_new_tokens)
    gc.eos_token_id = list(stop_ids) if stop_ids else gc.eos_token_id
    if pad_id is not None:
        gc.pad_token_id = int(pad_id)
    return gc


def eval_model_chat_mcqa(cfg: Dict[str, Any], model, tok, eval_ex: List[Any], *,
                         method: str, seed: int, model_dir: Optional[str] = None,
                         precision: str = "", split: str = "final_holdout") -> Dict[str, Any]:
    """KorMedMCQA accuracy via HF generate (greedy). Headline numbers use eval_vllm.py
    (the served path); this path is for selection_dev checks and parity."""
    import torch
    from . import kormedmcqa as K

    st = chat_settings(cfg)
    ecfg = cfg["eval"]
    mnt, bs = int(ecfg["max_new_tokens"]), int(ecfg["batch_size"])
    tok = prepare_tokenizer(tok, st)
    tok.padding_side = "left"
    tok.truncation_side = "left"
    stop_ids = stop_token_ids(tok, st)
    gc = greedy_generation_config(model, mnt, stop_ids, tok.pad_token_id)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    prompts = [render_prompt(tok, ex, st, True) for ex in eval_ex]
    raw, ntok, t0 = [], 0, time.time()
    with torch.no_grad():
        for i in range(0, len(prompts), bs):
            enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, truncation=True,
                      max_length=int(cfg["data"].get("max_seq_len", 4096))).to(model.device)
            gen = model.generate(**enc, generation_config=gc)
            new = gen[:, enc["input_ids"].shape[1]:]
            ntok += int((new != tok.pad_token_id).sum().item())
            raw.extend(tok.batch_decode(new, skip_special_tokens=True))
    dt = time.time() - t0
    preds = [K.extract_choice(r) for r in raw]
    sc = K.score(preds, eval_ex, raw=raw)
    return {"method": method, "seed": int(seed), "base_model": cfg["base_model"]["selected"],
            "task": "kormedmcqa", "engine": "hf_generate", "split": split,
            "accuracy": sc["accuracy"], "accuracy_ci95": sc["accuracy_ci95"],
            "n_eval": sc["n"], "n_correct": sc["n_correct"], "n_unparsed": sc["n_unparsed"],
            "by_role": sc["by_role"], "by_year": sc["by_year"],
            "size_gb": round(E.dir_size_gb(model_dir), 4) if model_dir and os.path.isdir(model_dir) else None,
            "peak_vram_gb": round(E.peak_vram_gb(), 4) if E.peak_vram_gb() is not None else None,
            "tok_per_s": round(ntok / dt, 2) if dt > 0 else 0.0, "precision": precision,
            "gen_seconds": round(dt, 1), "template_kwargs": st["template_kwargs"],
            "sample_predictions": raw[:3], "items": sc["items"]}


def eval_model_chat(cfg: Dict[str, Any], model, tok, eval_ex: List[QAExample], *,
                    method: str, seed: int, model_dir: Optional[str] = None,
                    precision: str = "", split: str = "final_holdout") -> Dict[str, Any]:
    import torch

    if task_name(cfg) == "kormedmcqa":
        return eval_model_chat_mcqa(cfg, model, tok, eval_ex, method=method, seed=seed,
                                    model_dir=model_dir, precision=precision, split=split)

    sys_p = system_prompt(cfg)
    et = enable_thinking_flag(cfg)
    ecfg = cfg["eval"]
    mnt, bs = int(ecfg["max_new_tokens"]), int(ecfg["batch_size"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"   # keep the question + assistant marker (prompt tail) on overflow
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    prompts = [build_chat_prompt(tok, sys_p, ex.context, ex.question, None, et, True) for ex in eval_ex]
    refs = [ex.answers for ex in eval_ex]
    preds, ntok, t0 = [], 0, time.time()
    with torch.no_grad():
        for i in range(0, len(prompts), bs):
            batch = prompts[i:i + bs]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=3072).to(model.device)
            gen = model.generate(**enc, max_new_tokens=mnt, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            ntok += int((new != tok.pad_token_id).sum().item())
            preds.extend(extract_answer(r) for r in tok.batch_decode(new, skip_special_tokens=True))
    dt = time.time() - t0
    scores = E.korquad_em_f1(preds, refs)
    ppl = E.perplexity(model, tok, [ex.context for ex in eval_ex[:int(ecfg["ppl_samples"])]])
    return {"method": method, "seed": int(seed), "base_model": cfg["base_model"]["selected"],
            "exact_match": round(scores["exact_match"], 3), "f1": round(scores["f1"], 3),
            "n_eval": scores["n"], "perplexity": round(ppl, 4) if ppl == ppl else float("nan"),
            "size_gb": round(E.dir_size_gb(model_dir), 4) if model_dir and os.path.isdir(model_dir) else None,
            "peak_vram_gb": round(E.peak_vram_gb(), 4) if E.peak_vram_gb() is not None else None,
            "tok_per_s": round(ntok / dt, 2) if dt > 0 else 0.0, "precision": precision,
            "gen_seconds": round(dt, 1), "split": split, "sample_predictions": preds[:3]}


def qat_scheme_selftest(device: str = "cuda", d: int = 512, gs: int = 128) -> Dict[str, Any]:
    """Guard the exact QAT prepare/convert path that ``train_method_c`` runs.

    Rubber-duck concern #1 asked whether C trains against what it serves. torchao 0.17 only
    exposes an int4 *fake-quant* for PLAIN/PRESHUFFLED packing (``_infer_fake_quantize_configs``
    rejects TILE_PACKED_TO_4D), while the only kernel-free int4 *serving* path on this box is
    TILE_PACKED_TO_4D. The two use the same asymmetric group-``gs`` int4 family but different
    kernels, so their per-layer outputs are *not* bit-identical — an exact-match assertion would
    be wrong. What actually matters, and what this checks, is:

    1. ``prepare`` truly replaces child Linears with ``FakeQuantizedLinear`` (the transform is a
       no-op on a bare root module, so this must be checked on a real *container*), i.e. QAT
       actually simulates int4 during training rather than silently training full-precision.
    2. The fake-quant rounding error is the same order of magnitude as the tile-packed serving
       error (same int4 family) — so the QAT signal transfers to the served format.
    3. ``convert`` + tile-packed re-quant round-trips without error (the export path in C).

    Returns a diagnostics dict with an ``ok`` flag (True when all three hold).
    """
    import torch
    import torch.nn as nn
    from torchao.quantization import quantize_, Int4WeightOnlyConfig
    from torchao.quantization.qat import QATConfig
    from torchao.quantization.qat.fake_quantize_config import _infer_fake_quantize_configs

    def _lin(m, fqn):
        return isinstance(m, nn.Linear)

    torch.manual_seed(0)
    x = torch.randn(8, d, dtype=torch.bfloat16, device=device)

    def _mk():
        torch.manual_seed(0)
        return nn.Sequential(nn.Linear(d, d, bias=False), nn.ReLU(),
                             nn.Linear(d, d, bias=False)).to(device, torch.bfloat16)

    ref = _mk()
    with torch.no_grad():
        y_orig = ref(x)

    # Real tile-packed int4 serving (what B and C actually deploy).
    serve = _mk()
    quantize_(serve, E.make_int4_weightonly_config(gs))
    with torch.no_grad():
        y_serve = serve(x)
    serve_err = float((y_orig - y_serve).norm() / (y_orig.norm() + 1e-9))

    # Exact QAT prepare path from train_method_c (fake-quant on child Linears).
    fake = _mk()
    _, wcfg = _infer_fake_quantize_configs(Int4WeightOnlyConfig(group_size=gs))
    quantize_(fake, QATConfig(weight_config=wcfg, step="prepare"), filter_fn=_lin)
    n_fq = sum(1 for m in fake.modules() if type(m).__name__ == "FakeQuantizedLinear")
    with torch.no_grad():
        y_fake = fake(x)
    fake_err = float((y_orig - y_fake).norm() / (y_orig.norm() + 1e-9))

    # Convert (strip fake-quant -> int4-robust bf16) then tile-packed re-quant: C's export.
    convert_ok = True
    try:
        quantize_(fake, QATConfig(step="convert"), filter_fn=_lin)
        quantize_(fake, E.make_int4_weightonly_config(gs))
        with torch.no_grad():
            _ = fake(x)
    except Exception:
        convert_ok = False

    ratio = fake_err / serve_err if serve_err > 0 else float("inf")
    fires = n_fq > 0
    same_family = 0.4 <= ratio <= 2.5
    ok = bool(fires and same_family and convert_ok and fake_err > 0.01)
    return {"ok": ok, "prepare_fires": fires, "fake_quant_layers": n_fq,
            "fake_err_vs_orig": round(fake_err, 5), "serve_err_vs_orig": round(serve_err, 5),
            "fake_to_serve_ratio": round(ratio, 3), "same_int4_family": same_family,
            "convert_roundtrip_ok": convert_ok, "group_size": gs}


def aggregate_seeds(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """rows: per-(method,seed) eval dicts -> {method: {mean/std for em/f1/ppl, seeds:[...]}}"""
    def ms(vals):
        vals = [v for v in vals if v is not None and v == v]
        if not vals:
            return {"mean": float("nan"), "std": 0.0, "n": 0}
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)) if len(vals) > 1 else 0.0
        return {"mean": round(mean, 3), "std": round(std, 3), "n": len(vals)}

    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)
    agg = {}
    for method, rs in by_method.items():
        if "accuracy" in rs[0] and "exact_match" not in rs[0]:      # KorMedMCQA rows
            agg[method] = {
                "base_model": rs[0]["base_model"], "precision": rs[0].get("precision", ""),
                "n_eval": rs[0]["n_eval"], "seeds": sorted(r["seed"] for r in rs),
                "accuracy": ms([r["accuracy"] for r in rs]),
                "n_unparsed": ms([r.get("n_unparsed") for r in rs]),
                "size_gb": rs[0].get("size_gb"),
                "tok_per_s": ms([r.get("tok_per_s") for r in rs]),
            }
            continue
        agg[method] = {
            "base_model": rs[0]["base_model"], "precision": rs[0].get("precision", ""),
            "n_eval": rs[0]["n_eval"], "seeds": sorted(r["seed"] for r in rs),
            "exact_match": ms([r["exact_match"] for r in rs]),
            "f1": ms([r["f1"] for r in rs]),
            "perplexity": ms([r["perplexity"] for r in rs]),
            "size_gb": rs[0].get("size_gb"),
            "tok_per_s": ms([r["tok_per_s"] for r in rs]),
        }
    return agg
