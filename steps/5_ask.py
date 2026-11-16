"""5단계(질문): 띄워 둔 vLLM 서버에 2022~2024년 문제 몇 개를 보내고 답을 봅니다.

먼저 다른 터미널에서 서버를 띄웁니다: bash steps/5_serve.sh
    python steps/5_ask.py                  # 문제 3개
    python steps/5_ask.py --n 10
    python steps/5_ask.py --show-request   # 서버에 보내는 JSON을 그대로 보여 줍니다

요청은 OpenAI Chat Completions 형식이라 openai 라이브러리나 curl로도 똑같이 보낼 수 있습니다.
이 스크립트는 표준 라이브러리(urllib)만 씁니다.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

import common as C


def build_request(ex, st: dict, stop_ids: list[int], max_tokens: int = 16) -> dict:
    """문항 하나를 Chat Completions 요청 본문(JSON)으로 만듭니다.

    messages              시스템 프롬프트 + (문제와 보기 다섯 개). 4단계 채점과 같은 내용입니다.
                          채팅 템플릿은 서버(vLLM)가 적용합니다.
    chat_template_kwargs  Mi:dm 템플릿에 넣는 날짜를 고정합니다. 날짜가 바뀌면 프롬프트가 달라집니다
    stop_token_ids        Mi:dm이 답을 끝낼 때 내는 토큰에서 멈춥니다
    (chat_template_kwargs, stop_token_ids는 OpenAI API에는 없고 vLLM이 더 받는 필드입니다)
    """
    return {
        "model": "midm",                   # 5_serve.sh의 --served-model-name
        "messages": [
            {"role": "system", "content": st["system"]},
            {"role": "user", "content": C.V.user_content_for(ex, st)},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": dict(st["template_kwargs"]),
        "stop_token_ids": stop_ids,
    }


def post_json(url: str, body: dict, timeout: float = 120.0) -> tuple[dict, float]:
    """JSON을 POST로 보내고 (응답, 걸린 초)를 돌려줍니다."""
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp), time.time() - t0


def stop_token_ids() -> list[int]:
    """Mi:dm의 멈춤 토큰 번호를 토크나이저에서 읽습니다.

    1~3단계는 어휘를 바꾸지 않으므로 튜닝 전 모델의 토크나이저로 읽어도 같습니다.
    """
    from transformers import AutoTokenizer

    _, st = C.load_settings()
    tok = AutoTokenizer.from_pretrained(C.BASE_MODEL, revision=C.BASE_REVISION)
    return C.V.stop_token_ids(tok, st)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="5단계: vLLM 서버에 문제 보내기")
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="서버 주소 (기본: 5_serve.sh의 주소)")
    ap.add_argument("--n", type=int, default=3, help="보낼 문제 수")
    ap.add_argument("--show-request", action="store_true", help="첫 요청의 JSON 본문을 찍습니다")
    args = ap.parse_args(argv)

    _, st = C.load_settings()
    examples = C.load_questions("test", args.n)
    stops = stop_token_ids()

    n_correct = 0
    for i, ex in enumerate(examples, start=1):
        body = build_request(ex, st, stops)
        if args.show_request and i == 1:
            print(json.dumps(body, ensure_ascii=False, indent=2))
        resp, seconds = post_json(args.url.rstrip("/") + "/v1/chat/completions", body)
        text = resp["choices"][0]["message"]["content"] or ""
        pred = C.K.extract_choice(text)
        ok = pred == ex.gold
        n_correct += ok
        print(f"[{i}/{len(examples)}] {C.K.ROLE_KO.get(ex.role, ex.role)} {ex.year}년 · "
              f"모델 답 {text.strip()!r} · 정답 {ex.gold} · {'맞음' if ok else '틀림'} · {seconds * 1000:.0f} ms")
        print(f"      {ex.question.strip()[:70]}")
    print(f"[5단계] {n_correct}/{len(examples)} 맞음")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
