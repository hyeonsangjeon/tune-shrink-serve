"""1단계: Mi:dm 2.0 Base에 LoRA를 학습하고, BF16 모델 하나로 합쳐 저장합니다.

실행 (GPU VM에서, 레포 최상위 폴더에서)
    python steps/1_lora_train.py                    # transformers + PEFT. H100 NVL에서 24분
    python steps/1_lora_train.py --max-steps 5      # 먼저 5 step만 돌려 보기
    python steps/1_lora_train.py --engine unsloth   # 같은 학습을 Unsloth로. 16분. ~/venv-unsloth에서
    python steps/1_lora_train.py --resume           # 끊긴 학습을 마지막 체크포인트부터 이어 가기

만드는 것
    outputs/1_lora/            LoRA를 합친 BF16 모델(약 22GB). 2~5단계가 이 폴더를 읽습니다
    outputs/1_lora_adapter/    LoRA 어댑터만(작음). 원래 모델에 얹어 쓸 수 있습니다
    outputs/1_lora_train.json  걸린 시간, step 수, 마지막 loss, GPU 메모리
    (--engine unsloth면 폴더 이름 끝에 _unsloth가 붙습니다)
    학습하는 동안 outputs/1_lora_run/에 체크포인트가 생기고, 끝까지 저장하면 지웁니다.

두 엔진은 데이터, 프롬프트, LoRA 설정, 학습률, 배치가 모두 같고 학습 엔진만 다릅니다.
발표 ①의 비교(24.3분 대 15.7분, 정답률 차이 없음)가 이 두 경로입니다.
"""
from __future__ import annotations

import argparse
import os
import shutil
import time

import common as C

# --------------------------------------------------------------------------- #
# 학습 설정. 발표 때 쓴 값이고 quantization/configs/kormedmcqa_midm.yaml 의
# lora:, train: 과 같아야 합니다(다르면 CI가 실패합니다).
# --------------------------------------------------------------------------- #
LORA_R = 16               # LoRA 행렬의 rank. 클수록 학습하는 값이 많아집니다
LORA_ALPHA = 32           # LoRA 출력에 곱하는 크기(alpha / r = 2배)
LORA_DROPOUT = 0.0
LORA_TARGETS = [          # LoRA를 붙일 층: 모든 블록의 어텐션 4개 + MLP 3개
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

EPOCHS = 2                # 학습 문항 3,401개를 두 번 봅니다 → 3,401 × 2 ÷ 16 = 426 step
BATCH_SIZE = 8            # GPU에 한 번에 올리는 문항 수
GRAD_ACCUM = 2            # 8문항씩 두 번 모아(16문항) 가중치를 한 번 고칩니다
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03       # 처음 3%의 step 동안 학습률을 0에서 천천히 올립니다
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0       # 기울기가 너무 크면 이 크기로 잘라 학습이 튀지 않게 합니다
SAVE_STEPS = 100          # 100 step마다 체크포인트를 남깁니다(끊기면 --resume으로 이어 갑니다)


def load_with_peft(st: dict):
    """transformers로 모델을 BF16으로 올리고, PEFT로 LoRA 어댑터를 붙입니다.

    원래 가중치(11.5B개)는 얼려 두고, 각 층 옆에 붙인 작은 행렬 두 개(A, B)만 학습합니다.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(C.BASE_MODEL, revision=C.BASE_REVISION)
    tok = C.prepare_tokenizer(tok, st)
    model = AutoModelForCausalLM.from_pretrained(
        C.BASE_MODEL, revision=C.BASE_REVISION, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False          # 학습 중에는 생성용 KV 캐시가 필요 없습니다
    model.enable_input_require_grads()      # gradient checkpointing과 LoRA를 같이 쓸 때 필요합니다

    lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                      target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()      # 학습하는 값의 수와 전체 대비 비율을 찍습니다
    return model, tok


def load_with_unsloth(st: dict):
    """Unsloth로 모델을 올리고 LoRA를 붙입니다. 4비트로 줄이지 않고 BF16 그대로 씁니다.

    Mi:dm 토크나이저 우회: Unsloth(2026.11.3)는 처음 불러올 때 transformers 일부를 고치는데,
    그 상태에서 Mi:dm 토크나이저를 읽으면 에러가 납니다
    ('tokenizers.AddedToken' object has no attribute 'keys').
    그래서 토크나이저만 Unsloth를 불러오지 않은 별도 프로세스에서 읽어 건네줍니다.
    모델을 올리는 것과 학습은 Unsloth가 합니다.
    """
    # (1) Unsloth를 불러오기 전에, 깨끗한 transformers로 토크나이저를 읽어 파일로 저장합니다.
    tok_file = C.V._pickle_tokenizer_in_subprocess(C.BASE_MODEL, C.BASE_REVISION)

    # (2) Unsloth는 transformers보다 먼저 불러와야 속도 최적화가 들어갑니다.
    import unsloth  # noqa: F401
    import unsloth.models.llama as unsloth_llama
    from unsloth import FastLanguageModel
    import pickle
    import torch

    # (3) Unsloth의 토크나이저 로더 대신 (1)에서 읽은 토크나이저를 쓰게 합니다.
    with open(tok_file, "rb") as fh:
        clean_tok = pickle.load(fh)
    unsloth_llama.load_correct_tokenizer = lambda *a, **k: clean_tok

    # (4) 모델을 BF16으로 올리고 LoRA를 붙입니다. 설정값은 PEFT 경로와 같습니다.
    model, tok = FastLanguageModel.from_pretrained(
        model_name=C.BASE_MODEL, revision=C.BASE_REVISION, max_seq_length=C.MAX_SEQ_LEN,
        dtype=torch.bfloat16, load_in_4bit=False)
    tok = C.prepare_tokenizer(tok, st)
    model = FastLanguageModel.get_peft_model(
        model, r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS, bias="none", random_state=C.SEED,
        use_gradient_checkpointing="unsloth")   # Unsloth 방식의 gradient checkpointing
    return model, tok


def train(model, tok, train_dataset, run_dir: str, engine: str, max_steps: int | None,
          resume: bool = False):
    """Hugging Face Trainer로 학습합니다. 두 엔진 모두 같은 Trainer와 같은 설정을 씁니다.

    resume=True면 run_dir의 마지막 체크포인트에서 가중치, optimizer, 학습률, step 수를 읽어
    그 자리부터 이어 갑니다. 체크포인트가 없으면 Trainer가 에러를 냅니다.
    """
    from transformers import Trainer, TrainingArguments

    args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=EPOCHS,
        max_steps=max_steps or -1,                  # -1이면 EPOCHS만큼 끝까지
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="linear",                 # warmup 뒤 학습률이 0까지 곧게 줄어듭니다
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        bf16=True,
        # 층마다 중간 결과를 버렸다가 backward에서 다시 계산해 메모리를 아낍니다.
        # Unsloth는 load_with_unsloth()에서 자기 방식으로 이미 켰습니다.
        gradient_checkpointing=(engine == "peft"),
        seed=C.SEED,
        logging_steps=10,
        logging_first_step=True,
        save_steps=SAVE_STEPS,
        save_total_limit=2,                         # 체크포인트는 최근 두 개만 남깁니다
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_dataset,
                      data_collator=C.make_collator(tok))
    trainer.train(resume_from_checkpoint=True if resume else None)
    return trainer


def save(trainer, model, tok, out_dir: str, adapter_dir: str, engine: str) -> None:
    """LoRA 어댑터를 따로 저장하고, 원래 가중치에 더해 BF16 모델 하나로 저장합니다.

    합친 모델은 LoRA 없이 보통 모델처럼 읽히므로, 2단계(INT4)와 vLLM 서빙이 그대로 씁니다.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)              # 어댑터만(수백 MB)
    tok.save_pretrained(adapter_dir)
    if engine == "peft":
        merged = trainer.model.merge_and_unload()   # W + (alpha/r)·B·A 를 계산해 W 자리에 넣습니다
        merged.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)
    else:
        model.save_pretrained_merged(out_dir, tok, save_method="merged_16bit")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="1단계: LoRA 학습 후 BF16 모델 하나로 합치기")
    ap.add_argument("--engine", choices=["peft", "unsloth"], default="peft",
                    help="peft = transformers+PEFT(~/venv-q), unsloth = Unsloth(~/venv-unsloth)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="이만큼만 학습합니다(시험용). 생략하면 2 epoch, 426 step")
    ap.add_argument("--out", default=None, help="저장할 폴더 (기본: outputs/1_lora)")
    ap.add_argument("--resume", action="store_true",
                    help="끊긴 학습을 outputs/1_lora_run/의 마지막 체크포인트부터 이어 갑니다")
    args = ap.parse_args(argv)

    suffix = "" if args.engine == "peft" else "_unsloth"
    out_dir = args.out or C.LORA_DIR + suffix
    adapter_dir = out_dir + "_adapter"
    run_dir = out_dir + "_run"

    # 1. 프롬프트 설정을 읽고 모델을 올립니다.
    #    Unsloth는 transformers보다 먼저 불러와야 해서, 모델을 데이터보다 먼저 올립니다.
    _, st = C.load_settings()
    model, tok = load_with_unsloth(st) if args.engine == "unsloth" else load_with_peft(st)

    # 2. 학습 문항(2012~2020년 3,401개)을 받아 토큰으로 바꿉니다. 손실은 정답 토큰에만 겁니다.
    examples = C.load_questions("train")
    train_dataset = C.make_train_dataset(tok, examples, st)
    print(f"[1단계] 학습 문항 {len(examples)}개, 엔진 {args.engine}")

    # 3. 학습합니다. 시간과 GPU 메모리 최댓값을 같이 잽니다.
    import torch

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer = train(model, tok, train_dataset, run_dir, args.engine, args.max_steps, args.resume)
    seconds = time.time() - t0

    # 4. 어댑터와 합친 BF16 모델을 저장합니다. 다 저장했으면 이어 가기용 체크포인트는 지웁니다.
    save(trainer, model, tok, out_dir, adapter_dir, args.engine)
    shutil.rmtree(run_dir, ignore_errors=True)

    losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    C.write_json(os.path.join(C.OUT, f"1_lora_train{suffix}.json"), {
        "engine": args.engine,
        "steps": trainer.state.global_step,
        "train_minutes": round(seconds / 60, 1),
        "last_loss": losses[-1] if losses else None,
        # PyTorch가 잡은 메모리의 최댓값. 발표의 메모리 숫자(nvidia-smi로 읽은 값)와는 재는 방법이 다릅니다
        "peak_torch_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        "model_dir": os.path.relpath(out_dir, C.ROOT),
    })
    print(f"[1단계] 끝: {trainer.state.global_step} step, {seconds / 60:.1f}분 → {out_dir}")
    print("[다음] python steps/2_int4_ptq.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
