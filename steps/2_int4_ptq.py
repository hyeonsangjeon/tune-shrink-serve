"""2단계: 1단계 BF16 모델을 학습 없이 INT4로 바꿉니다(PTQ, post-training quantization).

실행
    python steps/2_int4_ptq.py          # outputs/1_lora → outputs/2_int4_ptq. H100 NVL에서 17초

만드는 것
    outputs/2_int4_ptq/        INT4 모델(약 7.2GiB, BF16의 3분의 1)
    outputs/2_int4_ptq.json    걸린 시간, 바꾸기 전과 후의 폴더 크기

PTQ는 다 학습한 가중치를 한 번에 4비트로 반올림합니다. 빠르지만, 반올림에서 생기는 오차를
모델이 미리 겪어 본 적이 없습니다. 3단계 QAT가 이 오차를 학습 중에 겪게 하는 방법입니다.
"""
from __future__ import annotations

import argparse
import os
import time

import common as C

GROUP_SIZE = 128   # 가중치 128개마다 scale 하나. kormedmcqa_midm.yaml 의 ptq.group_size 와 같아야 합니다


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="2단계: BF16 모델을 학습 없이 INT4로 바꾸기(PTQ)")
    ap.add_argument("--src", default=C.LORA_DIR, help="바꿀 BF16 모델 폴더 (기본: outputs/1_lora)")
    ap.add_argument("--out", default=C.PTQ_DIR, help="저장할 폴더 (기본: outputs/2_int4_ptq)")
    args = ap.parse_args(argv)

    if not os.path.isfile(os.path.join(args.src, "config.json")):
        raise SystemExit(f"[2단계] 모델 폴더가 없습니다: {args.src}  먼저 1단계를 돌리세요.")

    # 모델을 GPU에 올리면서 선형층 가중치를 INT4로 바꾸고 저장합니다(common.save_as_int4).
    t0 = time.time()
    C.save_as_int4(args.src, args.out, GROUP_SIZE)
    seconds = time.time() - t0

    before, after = C.folder_size_gib(args.src), C.folder_size_gib(args.out)
    C.write_json(args.out + ".json", {
        "group_size": GROUP_SIZE,
        "seconds": round(seconds, 1),
        "size_gib_before": round(before, 2),
        "size_gib_after": round(after, 2),
        "model_dir": os.path.relpath(args.out, C.ROOT),
    })
    print(f"[2단계] 끝: {before:.2f} GiB → {after:.2f} GiB, {seconds:.0f}초 → {args.out}")
    print("[다음] python steps/3_int4_qat.py   또는 바로 채점: python steps/4_evaluate.py outputs/2_int4_ptq")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
