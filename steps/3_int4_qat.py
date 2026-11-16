"""3단계: INT4를 흉내 내며 더 학습한 뒤 INT4로 바꿉니다(QAT, quantization-aware training).

실행
    python steps/3_int4_qat.py                  # outputs/1_lora → outputs/3_int4_qat. H100 NVL에서 1시간 42분
    python steps/3_int4_qat.py --max-steps 5    # 먼저 5 step만 돌려 보기
    python steps/3_int4_qat.py --batch-size 2 --grad-accum 8   # 80GB GPU (한 step 16문항은 같음)
    python steps/3_int4_qat.py --resume         # 끊긴 학습을 마지막 체크포인트부터 이어 가기

만드는 것
    outputs/3_int4_qat_bf16/   INT4를 흉내 내며 학습한 BF16 모델
    outputs/3_int4_qat/        위 모델을 2단계와 같은 방법으로 바꾼 INT4 모델
    outputs/3_int4_qat.json    걸린 시간, step 수, 마지막 loss

디스크
    학습하는 동안 outputs/3_int4_qat_run/에 체크포인트가 생깁니다. 하나가 약 42GB(가중치 22GB +
    8비트 Adam 상태값)이고, 최근 두 개를 남기며 새것을 쓰는 동안 잠깐 세 개가 됩니다.
    INT4 모델까지 저장하면 지웁니다. 이 단계를 시작할 때 150GB 이상 비어 있어야 합니다.

PTQ(2단계)와 무엇이 다른가
    QAT는 학습하는 동안 forward에서 가중치를 INT4로 반올림했다가 BF16으로 되돌려 씁니다
    (가짜 양자화, fake quantization). 그래서 모델이 반올림 오차를 겪으면서 가중치를 고칩니다.
    backward에서는 반올림이 없었던 것처럼 기울기를 그대로 흘립니다(STE, straight-through estimator).
    학습이 끝나면 가짜 양자화를 걷어내고, 2단계와 똑같이 INT4로 저장합니다.

메모리
    LoRA가 아니라 출력층(lm_head)을 뺀 모든 선형층의 가중치(약 105억 개)를 직접 학습합니다.
    그래서 8비트 Adam과 gradient checkpointing을 써도 H100 NVL(94GB)에서 배치 4가 한계였습니다.
"""
from __future__ import annotations

import argparse
import os
import shutil
import time

import common as C

# --------------------------------------------------------------------------- #
# QAT 설정. 발표 때 쓴 값이고 kormedmcqa_midm.yaml 의 qat: 와 같아야 합니다(CI가 확인).
# --------------------------------------------------------------------------- #
GROUP_SIZE = 128          # 2단계 PTQ와 같은 INT4 형식
MAX_STEPS = 600           # 한 step 16문항 → 9,600문항 ≈ 학습 문항 3,401개를 2.8번
LEARNING_RATE = 1e-5      # 이미 학습한 모델을 조금만 고치므로 1단계(2e-4)보다 훨씬 작게
BATCH_SIZE = 4
GRAD_ACCUM = 4            # 4문항씩 네 번 모아(16문항) 한 번 고칩니다
WARMUP_RATIO = 0.03
OPTIM = "adamw_8bit"      # Adam의 상태값을 8비트로 저장해 메모리를 아낍니다(bitsandbytes)
SAVE_STEPS = 150          # 150 step마다 체크포인트(끊기면 --resume으로 이어 갑니다)


def is_target_linear(module, name: str) -> bool:
    """가짜 양자화를 넣을 층인지: 모든 nn.Linear, 단 마지막 출력층(lm_head)은 뺍니다."""
    import torch.nn as nn

    return isinstance(module, nn.Linear) and "lm_head" not in name


def add_fake_quant(model, group_size: int) -> int:
    """선형층을 TorchAO의 FakeQuantizedLinear로 바꾸고, 바꾼 층의 수를 돌려줍니다.

    가짜 양자화 설정은 서빙에 쓰는 INT4 설정(Int4WeightOnlyConfig, group 128)에서 뽑아,
    학습 때 겪는 반올림이 서빙 때 반올림과 같은 방식(비대칭, 128개씩 묶음)이 되게 합니다.

    `_infer_fake_quantize_configs`는 TorchAO(0.17)의 내부 함수입니다. 서빙 INT4와 같은
    반올림 설정을 얻는 가장 확실한 방법이라 썼습니다. 둘이 같은 방식인지는
    `bash scripts/gpu/stage.sh selftest`(결과: qat_selftest.json)로 확인했습니다.
    """
    from torchao.quantization import Int4WeightOnlyConfig, quantize_
    from torchao.quantization.qat import QATConfig
    from torchao.quantization.qat.fake_quantize_config import _infer_fake_quantize_configs

    _, weight_cfg = _infer_fake_quantize_configs(Int4WeightOnlyConfig(group_size=group_size))
    quantize_(model, QATConfig(weight_config=weight_cfg, step="prepare"), filter_fn=is_target_linear)
    return sum(1 for m in model.modules() if type(m).__name__ == "FakeQuantizedLinear")


def train_only_fake_quant_layers(model) -> int:
    """가짜 양자화가 들어간 선형층의 가중치만 학습하고, 나머지는 얼립니다.

    얼리는 것: 임베딩, 정규화층(RMSNorm), lm_head. 돌려주는 값은 학습하는 가중치의 수입니다.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for m in model.modules():
        if type(m).__name__ == "FakeQuantizedLinear":
            m.weight.requires_grad_(True)
            n += m.weight.numel()
    return n


def remove_fake_quant(model) -> None:
    """학습이 끝나면 FakeQuantizedLinear를 보통 nn.Linear로 되돌립니다.

    가중치는 학습한 BF16 값 그대로입니다. 이 모델은 INT4로 반올림해도 덜 망가지게 학습된 상태라,
    다음에 2단계와 같은 방법으로 INT4로 바꿉니다.
    """
    from torchao.quantization import quantize_
    from torchao.quantization.qat import QATConfig

    quantize_(model, QATConfig(step="convert"), filter_fn=is_target_linear)


def train(model, tok, train_dataset, run_dir: str, batch_size: int, grad_accum: int,
          max_steps: int, resume: bool = False):
    """1단계와 같은 Trainer로 학습합니다. 다른 것은 학습률, step 수, 8비트 Adam입니다.

    resume=True면 run_dir의 마지막 체크포인트부터 이어 갑니다. 가짜 양자화를 넣은 뒤에
    읽으므로, 체크포인트의 층 이름과 모델의 층 이름이 맞습니다.
    """
    from transformers import Trainer, TrainingArguments

    model.gradient_checkpointing_enable()   # 메모리를 아끼려고 중간 결과를 다시 계산합니다
    model.enable_input_require_grads()
    model.config.use_cache = False

    args = TrainingArguments(
        output_dir=run_dir,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        optim=OPTIM,
        bf16=True,
        gradient_checkpointing=False,       # 위에서 모델에 직접 켰습니다
        seed=C.SEED,
        logging_steps=20,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_dataset,
                      data_collator=C.make_collator(tok))
    trainer.train(resume_from_checkpoint=True if resume else None)
    return trainer


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="3단계: INT4를 흉내 내며 더 학습한 뒤 INT4로 바꾸기(QAT)")
    ap.add_argument("--src", default=C.LORA_DIR, help="시작할 BF16 모델 폴더 (기본: outputs/1_lora)")
    ap.add_argument("--out", default=C.QAT_DIR, help="INT4 모델을 저장할 폴더 (기본: outputs/3_int4_qat)")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS, help="학습 step 수 (기본: 600)")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="80GB GPU면 2")
    ap.add_argument("--grad-accum", type=int, default=GRAD_ACCUM, help="80GB GPU면 8")
    ap.add_argument("--resume", action="store_true",
                    help="끊긴 학습을 outputs/3_int4_qat_run/의 마지막 체크포인트부터 이어 갑니다")
    args = ap.parse_args(argv)

    if not os.path.isfile(os.path.join(args.src, "config.json")):
        raise SystemExit(f"[3단계] 모델 폴더가 없습니다: {args.src}  먼저 1단계를 돌리세요.")
    bf16_dir = C.QAT_BF16_DIR if args.out == C.QAT_DIR else args.out + "_bf16"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 1. 1단계 모델을 BF16으로 올리고, 선형층에 가짜 양자화를 넣습니다.
    _, st = C.load_settings()
    tok = C.prepare_tokenizer(AutoTokenizer.from_pretrained(args.src), st)
    model = AutoModelForCausalLM.from_pretrained(args.src, dtype=torch.bfloat16, device_map="cuda")
    n_layers = add_fake_quant(model, GROUP_SIZE)
    n_weights = train_only_fake_quant_layers(model)
    print(f"[3단계] 가짜 양자화를 넣은 선형층 {n_layers}개, 학습하는 가중치 {n_weights / 1e9:.2f}B개")

    # 2. 1단계와 같은 학습 문항으로 학습합니다.
    examples = C.load_questions("train")
    train_dataset = C.make_train_dataset(tok, examples, st)
    run_dir = args.out + "_run"
    t0 = time.time()
    trainer = train(model, tok, train_dataset, run_dir,
                    args.batch_size, args.grad_accum, args.max_steps, args.resume)
    seconds = time.time() - t0

    # 3. 가짜 양자화를 걷어내고 BF16으로 저장합니다. 학습 결과는 이 BF16 모델에 다 들어 있으므로
    #    이어 가기용 체크포인트(하나에 약 42GB)는 여기서 지웁니다.
    model = trainer.model
    remove_fake_quant(model)
    os.makedirs(bf16_dir, exist_ok=True)
    model.save_pretrained(bf16_dir)
    tok.save_pretrained(bf16_dir)
    losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    steps = trainer.state.global_step
    del model, trainer
    torch.cuda.empty_cache()
    shutil.rmtree(run_dir, ignore_errors=True)

    # 4. 2단계와 같은 방법으로 INT4로 바꿉니다. 여기서 실패하면 학습을 다시 할 필요 없이
    #    python steps/2_int4_ptq.py --src outputs/3_int4_qat_bf16 --out outputs/3_int4_qat 로 바꿉니다.
    C.save_as_int4(bf16_dir, args.out, GROUP_SIZE)

    C.write_json(args.out + ".json", {
        "group_size": GROUP_SIZE,
        "steps": steps,
        "train_minutes": round(seconds / 60, 1),
        "last_loss": losses[-1] if losses else None,
        "trained_weights": n_weights,
        "size_gib": round(C.folder_size_gib(args.out), 2),
        "model_dir": os.path.relpath(args.out, C.ROOT),
    })
    print(f"[3단계] 끝: {steps} step, {seconds / 60:.1f}분 → {args.out}")
    print("[다음] python steps/4_evaluate.py outputs/3_int4_qat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
