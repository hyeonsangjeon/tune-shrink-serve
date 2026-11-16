"""4단계: 모델에 국가시험 문제를 풀게 하고 정답률을 잽니다(vLLM으로 한꺼번에).

실행
    python steps/4_evaluate.py                          # outputs/1_lora, 2022~2024년 3,009문항
    python steps/4_evaluate.py outputs/2_int4_ptq       # INT4 모델(폴더를 보고 알아서 INT4로 읽습니다)
    python steps/4_evaluate.py outputs/3_int4_qat
    python steps/4_evaluate.py base                     # 튜닝 전 모델
    python steps/4_evaluate.py --split dev --limit 100  # 2021년 문제 100개만 빠르게

만드는 것
    outputs/4_eval_<모델 폴더 이름>_<test|dev>.json   정답률, 95% 신뢰구간, 직군별·연도별, 문항별 답

채점 방법
    프롬프트는 "보기 중 정답의 번호 하나만 숫자로 답하세요"이고, 답에서 보기 번호(③, 3, C 같은 것)를
    읽어 정답과 맞춰 봅니다. 규칙은 quantization/kormedmcqa.py 의 extract_choice 에 있습니다.
    temperature 0(매번 같은 답), 최대 16토큰까지 생성합니다.
    발표 숫자(튜닝 전 63.3%, LoRA 72.3% 등)는 같은 프롬프트와 같은 채점으로
    quantization/eval_vllm.py가 잰 값입니다.

2022~2024년 문제(test)는 발표에서 모델마다 한 번만 쟀습니다. 설정을 고르거나 고칠 때는
2021년 문제(--split dev)를 쓰세요. test 점수를 보고 고르면 test 점수가 실제보다 좋게 나옵니다.
"""
from __future__ import annotations

import argparse
import os
import time

import common as C

MAX_NEW_TOKENS = 16   # 튜닝한 모델은 "3"처럼 한 글자로 답합니다. 튜닝 전 모델의 "정답은 ③" 같은 답을 위한 여유


def resolve_model(name: str) -> tuple[str, str | None]:
    """"base"면 Hugging Face의 튜닝 전 모델(고정 revision), 아니면 폴더 경로를 돌려줍니다."""
    if name == "base":
        return C.BASE_MODEL, C.BASE_REVISION
    if not os.path.isfile(os.path.join(name, "config.json")):
        raise SystemExit(f"[4단계] 모델 폴더가 없습니다: {name}  1~3단계를 먼저 돌리세요.")
    return name, None


def build_prompts(tok, examples: list, st: dict) -> list[str]:
    """문항마다 Mi:dm 채팅 템플릿을 적용한 프롬프트 문자열을 만듭니다.

    한 프롬프트 = [Mi:dm 기본 시스템 블록 + 날짜] + [시스템 프롬프트] + [문제와 보기 다섯 개]
                 + [assistant 차례를 여는 토큰]
    학습(1단계)과 똑같은 함수로 만들어서, 학습 때와 채점 때 프롬프트가 같습니다.
    """
    return [C.V.render_prompt(tok, ex, st, add_generation_prompt=True) for ex in examples]


def load_vllm(model: str, revision: str | None):
    """vLLM 엔진을 올립니다. INT4 폴더면 TorchAO 형식으로 읽습니다."""
    from vllm import LLM

    kwargs = dict(
        model=model,
        dtype="bfloat16",
        max_model_len=4096,              # 이 과제의 프롬프트+답은 1,000토큰 안쪽입니다
        gpu_memory_utilization=0.85,     # 나머지 GPU 메모리를 KV 캐시로 씁니다
        generation_config="vllm",        # 모델 폴더의 샘플링 기본값(temperature 0.8 등)을 쓰지 않습니다
    )
    if revision:
        kwargs["revision"] = revision
    if C.is_int4_dir(model):
        kwargs["quantization"] = "torchao"
    return LLM(**kwargs)


def answer_all(llm, prompts: list[str], stop_ids: list[int]) -> tuple[list[str], float]:
    """모든 문제를 한꺼번에 넣고 답을 받습니다. 걸린 시간(초)도 돌려줍니다.

    vLLM이 문제들을 알아서 묶어 GPU를 꽉 채워 처리합니다(continuous batching).
    """
    from vllm import SamplingParams

    params = SamplingParams(
        temperature=0.0,                 # 가장 확률이 높은 토큰만 고릅니다(매번 같은 답)
        max_tokens=MAX_NEW_TOKENS,
        stop_token_ids=stop_ids,         # Mi:dm이 답을 끝낼 때 내는 토큰(<|eot_id|> 등)에서 멈춥니다
        skip_special_tokens=True,
    )
    llm.generate(prompts[:8], params, use_tqdm=False)   # 예열: 첫 호출의 준비 시간을 측정에서 뺍니다
    t0 = time.time()
    outputs = llm.generate(prompts, params, use_tqdm=True)
    return [o.outputs[0].text for o in outputs], time.time() - t0


def grade(texts: list[str], examples: list) -> dict:
    """답에서 번호를 읽어 채점합니다. 번호를 못 읽은 답은 틀린 것으로 셉니다."""
    preds = [C.K.extract_choice(t) for t in texts]
    return C.K.score(preds, examples, raw=texts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="4단계: 국가시험 문제로 정답률 재기")
    ap.add_argument("model", nargs="?", default=C.LORA_DIR,
                    help="모델 폴더 또는 base(튜닝 전 모델). 기본: outputs/1_lora")
    ap.add_argument("--split", choices=["test", "dev"], default="test",
                    help="test = 2022~2024년 3,009문항(발표 숫자), dev = 2021년 1,059문항")
    ap.add_argument("--limit", type=int, default=None, help="문항을 이만큼만 씁니다(직군 비율 유지)")
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    # 1. 문항과 프롬프트를 준비합니다.
    model, revision = resolve_model(args.model)
    _, st = C.load_settings()
    examples = C.load_questions(args.split, args.limit)
    tok = C.prepare_tokenizer(AutoTokenizer.from_pretrained(model, revision=revision), st)
    prompts = build_prompts(tok, examples, st)
    stop_ids = C.V.stop_token_ids(tok, st)

    # 2. vLLM으로 답을 받고 채점합니다.
    llm = load_vllm(model, revision)
    texts, seconds = answer_all(llm, prompts, stop_ids)
    result = grade(texts, examples)

    # 3. 결과를 찍고 저장합니다.
    lo, hi = result["accuracy_ci95"]
    print(f"\n[4단계] {args.model} · {args.split} {result['n']}문항")
    print(f"  정답률 {result['accuracy']:.1f}%  (95% 신뢰구간 {lo:.1f}~{hi:.1f}%)")
    print(f"  번호를 못 읽은 답 {result['n_unparsed']}개 · {seconds:.0f}초 · 초당 {result['n'] / seconds:.1f}문항")
    for role, r in result["by_role"].items():
        print(f"  {C.K.ROLE_KO.get(role, role):<5} {r['accuracy']:.1f}%  ({r['n']}문항)")

    name = "base" if args.model == "base" else os.path.basename(os.path.normpath(args.model))
    C.write_json(os.path.join(C.OUT, f"4_eval_{name}_{args.split}.json"), {
        "model": args.model, "split": args.split, "n": result["n"],
        "accuracy": result["accuracy"], "accuracy_ci95": result["accuracy_ci95"],
        "n_unparsed": result["n_unparsed"], "seconds": round(seconds, 1),
        "by_role": result["by_role"], "by_year": result["by_year"], "items": result["items"],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
