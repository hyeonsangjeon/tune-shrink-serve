"""Publish a KorMedMCQA × Mi:dm artifact to the Hugging Face Hub with a model card.

The card carries the four notices the licenses require or the domain needs:
  1. dataset attribution  — KorMedMCQA (CC BY-NC 2.0, HF card) — the "BY" condition
  2. non-commercial use    — inherited from the training data — the "NC" condition
  3. base model notice     — Mi:dm 2.0 Base, MIT, "Copyright (c) 2025 KT Corporation"
  4. not medical advice    — benchmark artifact only

Only safetensors BF16 dirs are published (A merge, C ``_bf16adapt``). INT4 dirs are pickled
torchao tensors tied to the torch/torchao build that wrote them, so they are rebuilt on the
serving GPU (``v2_run b`` / ``v2_run c --export-only``) instead of being shared.

Usage (needs a Hub write token in HF_TOKEN or `hf auth login`):
  python -m quantization.publish_hf --kind A_bf16 \
      --dir quantization/artifacts/kormedmcqa_midm/A_bf16_seed42 \
      --repo HyeonSang/Midm-2.0-KorMedMCQA \
      --train-json quantization/results/kormedmcqa_midm/A_train_seed42.json [--dry-run]
  python -m quantization.publish_hf --kind C_bf16adapt \
      --dir quantization/artifacts/kormedmcqa_midm/C_int4_qat_seed42_bf16adapt \
      --repo HyeonSang/Midm-2.0-KorMedMCQA-QAT-bf16 --train-json .../C_train_seed42.json
Re-run with --card-only and --eval-json ... to refresh the card once results exist.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

BASE_MODEL = "K-intelligence/Midm-2.0-Base-Instruct"
DATASET = "sean0042/KorMedMCQA"
DATASET_PAPER = "https://arxiv.org/abs/2403.01469"
CODE_URL = "https://github.com/hyeonsangjeon/tune-shrink-serve"  # code, result JSONs, step guide

MIDM_MIT = """MIT License

Copyright (c) 2025 KT Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

KINDS = {
    "A_bf16": {
        "title": "Mi:dm 2.0 Base × KorMedMCQA — BF16 LoRA (merged)",
        "what": ("Mi:dm 2.0 Base에 KorMedMCQA 학습 split으로 BF16 LoRA를 학습하고 병합한 모델입니다. "
                 "발표의 방법 A(품질 기준선)이며, 방법 B(INT4 PTQ)는 이 가중치를 TorchAO로 양자화해 만듭니다."),
        "tags": ["lora"],
    },
    "C_bf16adapt": {
        "title": "Mi:dm 2.0 Base × KorMedMCQA — INT4 QAT (BF16 weights before INT4 export)",
        "what": ("방법 A 병합본에서 TorchAO INT4 QAT(fake-quant, 전체 파라미터 STE)를 학습하고 convert한 BF16 "
                 "가중치입니다. INT4로 서빙하도록 학습됐으므로, 그대로 BF16으로 쓰기보다 TorchAO "
                 "Int4WeightOnly(group 128, tile-packed)로 양자화해 쓰는 것이 의도된 사용법입니다(방법 C)."),
        "tags": ["qat", "torchao", "int4"],
    },
}

ALLOW_PATTERNS = ["*.safetensors", "*.safetensors.index.json", "config.json", "generation_config.json",
                  "tokenizer*", "special_tokens_map.json", "chat_template*", "README.md", "LICENSE"]
IGNORE_PATTERNS = ["*.bin", "*.pt", "*.pth", "optimizer*", "scheduler*", "rng_state*",
                   "checkpoint-*", "trainer_state.json", "training_args*"]


def _results_table(eval_rows: List[Dict[str, Any]]) -> str:
    lines = ["| 모델 | 정답률 (%) | 95% CI | 문항 수 | 답 추출 실패 | 측정 GPU | 엔진 |",
             "|---|---|---|---|---|---|---|"]
    for r in eval_rows:
        ci = r.get("accuracy_ci95")
        ci_s = f"{ci[0]}–{ci[1]}" if ci else "—"
        lines.append(f"| {r.get('method')} | {r.get('accuracy')} | {ci_s} | {r.get('n_eval')} | "
                     f"{r.get('n_unparsed')} | {r.get('gpu_label') or '—'} | {r.get('engine', '—')} |")
    return "\n".join(lines)


def render_model_card(kind: str, repo_id: str, *, eval_rows: Optional[List[Dict[str, Any]]] = None,
                      train_log: Optional[Dict[str, Any]] = None, code_url: Optional[str] = None,
                      date_string: str = "21 Nov 2026") -> str:
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {sorted(KINDS)}")
    k = KINDS[kind]
    tags = ["medical", "korean", "kormedmcqa", "midm", "pytorch-day-korea-2026"] + k["tags"]
    front = [
        "---", "language:", "- ko", "license: other", "license_name: mit-base-and-cc-by-nc-2.0-data",
        "license_link: LICENSE", f"base_model: {BASE_MODEL}", "base_model_relation: finetune",
        "datasets:", f"- {DATASET}", "pipeline_tag: text-generation", "library_name: transformers",
        "tags:"] + [f"- {t}" for t in tags] + ["---", ""]

    train_bits = ""
    if train_log:
        keep = {x: train_log.get(x) for x in ("backend", "n_train", "epochs", "global_step", "max_steps",
                                              "train_seconds", "train_loss", "seed") if train_log.get(x) is not None}
        train_bits = "\n학습 기록: " + ", ".join(f"`{x}={v}`" for x, v in keep.items()) + "\n"

    results = ("\n## 결과\n\nvLLM으로 서빙한 상태에서 greedy로 잰 정답률입니다(찍으면 20%, 합격 참고선 60%).\n\n"
               + _results_table(eval_rows) + "\n") if eval_rows else \
        "\n## 결과\n\n평가가 끝나면 이 카드에 추가합니다.\n"

    code = ""
    if code_url:
        label = code_url.split("github.com/", 1)[-1].rstrip("/")
        code = f"\n\n코드와 결과 JSON, 따라 하는 방법은 GitHub [{label}]({code_url})에 있습니다."
    body = f"""# {k['title']}

{k['what']}

PyTorch Day Korea 2026 Serve Track 발표 「작은 모델을 더 작고 빠르게: Unsloth LoRA·TorchAO INT4 QAT부터 vLLM 서빙까지」의 실험 산출물입니다.{code}

> **의료 조언이 아닙니다.** 국가시험 문제 풀이 벤치마크용 연구 산출물입니다. 진단·처방·임상 판단에 쓰지 마세요.
> **비상업 용도로만 사용할 수 있습니다.** 학습 데이터(KorMedMCQA)가 CC BY-NC 2.0입니다.

## 학습 데이터와 평가

- 데이터: [`{DATASET}`](https://huggingface.co/datasets/{DATASET}) — 의사·간호사·약사·치과의사 국가시험 5지선다 ([논문]({DATASET_PAPER}))
- 학습: `train` split (2012–2020년 기출, 3,401문항)
- 선택: `dev` split (2021년, 1,059문항) — 프롬프트·step 선택에만 사용
- 평가: `test` split (2022–2024년, 3,009문항) — 최종 비교에 한 번만 사용
- 정답률 = 모델이 고른 번호가 공식 정답과 같은 비율. 답을 추출하지 못한 응답은 오답으로 셉니다.
{train_bits}{results}
## 사용법

Mi:dm 채팅 템플릿은 KT 시스템 문구와 오늘 날짜를 자동으로 넣습니다. 재현하려면 `date_string`을 고정하고,
`<|eot_id|>`(131301)를 정지 토큰으로 지정하고, 체크포인트의 샘플링 기본값(T 0.8 등) 대신 greedy를 쓰세요.

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

repo = "{repo_id}"
tok = AutoTokenizer.from_pretrained(repo)
msgs = [{{"role": "system", "content": "보기 중 정답의 번호 하나만 숫자로 출력하세요."}},
        {{"role": "user", "content": "다음은 의사 국가시험 문제입니다. ...\\n1. ...\\n2. ...\\n3. ...\\n4. ...\\n5. ..."}}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                 date_string="{date_string}")
llm = LLM(repo, dtype="bfloat16", generation_config="vllm")
out = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=16, stop_token_ids=[131301, 2]))
print(out[0].outputs[0].text)   # 예: "3"
```

## 한계

- 국가시험 기출은 공개 자료라 사전학습 데이터에 섞였을 수 있습니다. 연도별 정답률을 함께 보세요.
- 이미지가 필요한 문항은 데이터셋에 없습니다. 실제 시험의 과목별 과락 기준도 반영하지 않습니다.

## 라이선스

이 가중치에는 두 가지 조건이 함께 적용됩니다.

1. **베이스 모델:** [Mi:dm 2.0 Base]({'https://huggingface.co/' + BASE_MODEL}) — MIT License, Copyright (c) 2025 KT Corporation. 전문은 [`LICENSE`](LICENSE)에 있습니다.
2. **학습 데이터:** [KorMedMCQA](https://huggingface.co/datasets/{DATASET}) — CC BY-NC 2.0 (HF 데이터셋 카드 기준). 출처를 표시해야 하고, 학습 데이터 조건에 따라 이 가중치는 **비상업 용도로만** 사용하세요.

데이터셋 인용: Kweon et al., *KorMedMCQA: Multi-Choice Question Answering Benchmark for Korean Healthcare Professional Licensing Examinations* ({DATASET_PAPER}).
"""
    return "\n".join(front) + body


def render_license() -> str:
    return (MIDM_MIT + "\n---\n\nThis model was fine-tuned on KorMedMCQA (" + DATASET + "), "
            "licensed CC BY-NC 2.0 per its Hugging Face dataset card. Because of the training data, "
            "use of these weights is limited to NON-COMMERCIAL purposes. Attribution: " + DATASET_PAPER + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Publish an A_bf16 / C_bf16adapt dir with a model card.")
    ap.add_argument("--kind", required=True, choices=sorted(KINDS))
    ap.add_argument("--dir", required=True)
    ap.add_argument("--repo", required=True, help="e.g. HyeonSang/Midm-2.0-KorMedMCQA")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--train-json", default=None)
    ap.add_argument("--eval-json", action="append", default=[], help="eval_vllm JSON (repeatable)")
    ap.add_argument("--code-url", default=CODE_URL, help="link to the code repo in the card (default: %(default)s)")
    ap.add_argument("--card-only", action="store_true", help="upload README.md/LICENSE only")
    ap.add_argument("--dry-run", action="store_true", help="write the card locally, upload nothing")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.dir):
        raise SystemExit(f"missing dir: {args.dir}")
    if any(f.endswith(".bin") for f in os.listdir(args.dir)) and not any(
            f.endswith(".safetensors") for f in os.listdir(args.dir)):
        raise SystemExit("this looks like a pickled INT4 dir; publish the BF16 safetensors dirs only")
    train_log = json.load(open(args.train_json, encoding="utf-8")) if args.train_json else None
    eval_rows = [json.load(open(p, encoding="utf-8")) for p in args.eval_json]
    card = render_model_card(args.kind, args.repo, eval_rows=eval_rows or None,
                             train_log=train_log, code_url=args.code_url)
    with open(os.path.join(args.dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(card)
    with open(os.path.join(args.dir, "LICENSE"), "w", encoding="utf-8") as fh:
        fh.write(render_license())
    print(f"[publish] wrote README.md + LICENSE in {args.dir}")
    if args.dry_run:
        print("[publish] dry run — nothing uploaded")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, private=args.private, exist_ok=True, repo_type="model")
    allow = ["README.md", "LICENSE"] if args.card_only else ALLOW_PATTERNS
    api.upload_folder(folder_path=args.dir, repo_id=args.repo, repo_type="model",
                      allow_patterns=allow, ignore_patterns=IGNORE_PATTERNS,
                      commit_message=f"{args.kind}: {'card update' if args.card_only else 'upload'}")
    print(f"[publish] https://huggingface.co/{args.repo} ({'private' if args.private else 'public'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
