#!/usr/bin/env bash
# Run one runbook stage on the GPU VM with a log and a state marker.
#
#   bash scripts/gpu/stage.sh <stage> [--detach]
#
# --detach starts the stage under nohup and returns at once (use it for anything longer
# than a minute; `az vm run-command` calls time out). Progress: scripts/gpu/status.sh.
# Logs: ~/work/logs/<stage>.log   State: ~/work/state/<stage>.{running,done,failed}
#
# Stages, in guide order (docs/guide/02~07):
#   selftest  base_dev  a_smoke  a  a_dev  gate  b  c_smoke  c        (training GPU)
#   publish_A  publish_C                                              (needs `hf auth login`)
#   fetch_hf  b  c_export  final_base  final_A  final_B  final_C       (evaluation GPU)
#   compare  bench_A  bench_B  bench_C       (final files are tagged by GPU type: *_h100.json, *_a100.json)
#   online_A  online_B                     (vLLM HTTP server; prefix cache on/off; MCQA + decode-128 traffic;
#                                           FRESH=1: new questions at every concurrency level, MCQA only)
#   sweep_smoke  sweep_a  sweep_dev  sweep_merge  self_explain  sweep_report   (recipe sweep, selection on 2021;
#     CONFIG=quantization/configs/sweep/<arm>.yaml is required). TAG=<arm> keeps one log/state per arm;
#     MODE=probs|explain for sweep_dev; ADAPTER=<run>/checkpoint-N for sweep_merge; LIMIT=N for a self_explain pilot.
#   stop                                   STOP=<stage[_TAG]>: kill that detached stage's whole process group
#
# a_smoke / c_smoke move their outputs to *_smoke afterwards, so the full run never
# resumes from a 5-step smoke checkpoint.
#
# Environment: BACKEND (hf = Method A on transformers+peft in venv-q), SEED (42), PRICES ("--price ondemand=6.98 --price spot=3.537"),
#              GPU_LABEL (default: GPU name + VM size), GATE_PT (2.0), RESUME (unset).
set -uo pipefail

STAGE=${1:?usage: stage.sh <stage> [--detach]}
WORK=${WORK:-$HOME/work}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SELF="$REPO/scripts/gpu/stage.sh"
mkdir -p "$WORK/logs" "$WORK/state"

if [ "${2:-}" = "--detach" ]; then
  # setsid + </dev/null: fully leave the caller's session, or `az vm run-command` waits for the stage
  setsid nohup bash "$SELF" "$STAGE" </dev/null >/dev/null 2>&1 &
  echo "started $STAGE${TAG:+_$TAG} pid=$! log=$WORK/logs/$STAGE${TAG:+_$TAG}.log"
  exit 0
fi

NAME=$STAGE${TAG:+_$TAG}
ST=$WORK/state/$NAME
if [ -f "$ST.running" ] && kill -0 "$(cat "$ST.running")" 2>/dev/null; then
  echo "$NAME is already running (pid $(cat "$ST.running"))"; exit 2
fi
echo $$ >"$ST.running"; rm -f "$ST.done" "$ST.failed"
LOG=$WORK/logs/$NAME.log
exec >>"$LOG" 2>&1
echo "== $NAME start $(date -u +%FT%TZ) config=${CONFIG:-default} seed=${SEED:-42}"

cd "$REPO"
export HF_HOME=${HF_HOME:-$WORK/hf} VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONUNBUFFERED=1
case "$STAGE" in   # sweep stages never fall back to the talk config (that would overwrite its A/results)
  sweep_smoke|sweep_a|sweep_merge|self_explain)
    case "${CONFIG:-}" in quantization/configs/sweep/*.yaml) ;; *)
      echo "$STAGE needs CONFIG=quantization/configs/sweep/<arm>.yaml (got '${CONFIG:-}')"; rm -f "$ST.running"; exit 64;; esac ;;
  sweep_dev)  # also allowed: the Unsloth talk config, to score the reference adapter through vLLM+LoRA
    case "${CONFIG:-}" in quantization/configs/sweep/*.yaml|quantization/configs/kormedmcqa_midm_unsloth.yaml) ;; *)
      echo "sweep_dev needs CONFIG=quantization/configs/sweep/<arm>.yaml (got '${CONFIG:-}')"; rm -f "$ST.running"; exit 64;; esac ;;
esac
case "${CONFIG:-}" in *_seed[0-9]*.yaml)   # e.g. s0_seed43.yaml must run with SEED=43
  want=${CONFIG##*_seed}; want=${want%.yaml}
  [ "${SEED:-42}" = "$want" ] || { echo "$CONFIG is seed $want but SEED=${SEED:-42}"; rm -f "$ST.running"; exit 64; } ;;
esac
CFG=${CONFIG:-quantization/configs/kormedmcqa_midm.yaml}   # CONFIG=...unsloth.yaml for the Unsloth A arm
SEED=${SEED:-42}
Q=$HOME/venv-q/bin/python
U=$HOME/venv-unsloth/bin/python
cfgpath() { "$Q" -c 'import sys; from quantization.data_korquad import load_config; print(load_config(sys.argv[1])["paths"][sys.argv[2]])' "$CFG" "$1"; }
ART=$(cfgpath artifacts_dir)
RES=$(cfgpath results_dir)
ADIR=$(cfgpath method_a_dir)_seed$SEED
read -r -a PRICE_ARGS <<<"${PRICES:-}"
VMSIZE=$(curl -s -m 3 -H Metadata:true \
  "http://169.254.169.254/metadata/instance/compute/vmSize?api-version=2021-02-01&format=text" || true)
GPU_LABEL=${GPU_LABEL:-"$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) (${VMSIZE:-unknown})"}
# Short GPU tag for final/bench file names: one result table per GPU type, never mixed.
GPU_TAG=${GPU_TAG:-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | grep -o -E 'H100|A100|L40S|A10' | head -1 | tr 'A-Z' 'a-z')}
GPU_TAG=${GPU_TAG:-gpu}
RESUME_FLAG=(); [ -n "${RESUME:-}" ] && RESUME_FLAG=(--resume)
# Method A backend: unsloth (config default, unsloth venv) or BACKEND=hf (transformers+peft, venv-q)
if [ "${BACKEND:-}" = "hf" ]; then A_PY=$Q; A_FLAG=(--backend hf); else A_PY=$U; A_FLAG=(); fi

park() {  # move smoke outputs aside so the full run starts clean
  local tag=$1; shift
  for p in "$@"; do
    [ -e "$p" ] || continue
    rm -rf "${p}_smoke"; mv "$p" "${p}_smoke"; echo "parked $p -> ${p}_smoke"
  done
  : "$tag"
}

evalv() {  # evalv <method> <split> [extra args...]
  local m=$1 s=$2; shift 2
  "$Q" -m quantization.eval_vllm --config "$CFG" --method "$m" --seed "$SEED" --split "$s" \
    --gpu-label "$GPU_LABEL" "${PRICE_ARGS[@]}" "$@"
}

online() {  # online <method> <model dir> <quant|''>: HTTP serving, prefix cache on/off
  # FRESH=1: every concurrency level sends 256 questions no earlier level sent (2021 questions,
  # shuffled with seed 0; 4 x 256 = 1,024 of 1,059), the same ones with the cache on and off.
  # The prefix cache then holds only the shared system block. Skips the decode-128 traffic.
  local m=$1 dir=$2 quant=$3 port=8000 cache extra out fresh=()
  [ "${FRESH:-}" = 1 ] && fresh=(--fresh-per-level --shuffle-seed 0)
  for cache in on off; do
    extra=""; [ "$cache" = off ] && extra="--no-enable-prefix-caching"
    PATH="$HOME/venv-q/bin:$PATH" EXTRA_ARGS="$extra" MAX_MODEL_LEN=2048 HEALTH_TRIES=900 LOG_FILE="$WORK/logs/vllm_${m}_${cache}.log" \
      PID_FILE="$WORK/state/vllm.pid" bash quantization/serving/serve_vllm.sh "$dir" $port $quant || return
    # MCQA traffic as served: real prompts (~650 tokens incl. the fixed system block), answer = 1-2 tokens
    out="$RES/online_${m}_mcqa_cache${cache}_seed${SEED}_${GPU_TAG}.json"
    "$Q" -m quantization.serving.v2_bench_serve --model "$dir" --concurrency 1,8,32,64 --num-prompts 256 \
      --prompts-config "$CFG" --prompts-split selection_dev --stop-token-ids 131301,2 --max-tokens 16 \
      --sla-ttft-s 1.0 --sla-e2e-s 2.0 "${fresh[@]}" --output "$out" || { kill "$(cat "$WORK/state/vllm.pid")"; return 1; }
    if [ "$cache" = on ] && [ "${FRESH:-}" != 1 ]; then  # decode-heavy: same prompts, exactly 128 generated tokens
      out="$RES/online_${m}_decode128_cache${cache}_seed${SEED}_${GPU_TAG}.json"
      "$Q" -m quantization.serving.v2_bench_serve --model "$dir" --concurrency 1,8,32,64 --num-prompts 256 \
        --prompts-config "$CFG" --prompts-split selection_dev --max-tokens 128 --ignore-eos \
        --sla-ttft-s 1.0 --sla-e2e-s 10.0 --output "$out" || { kill "$(cat "$WORK/state/vllm.pid")"; return 1; }
    fi
    kill "$(cat "$WORK/state/vllm.pid")"
    for _ in $(seq 90); do  # wait until the server's workers have released the GPU
      [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ] && break; sleep 2
    done
  done
}

run() {
  case "$STAGE" in
    selftest)  "$Q" -m quantization.v2_run selftest --config "$CFG" ;;
    base_dev)  evalv base selection_dev ;;
    a_smoke)   "$A_PY" -m quantization.v2_run a --config "$CFG" --seed "$SEED" --max-steps 5 "${A_FLAG[@]}" || return
               grep -h '"vocab_ok"' "$RES/A_train_seed$SEED.json"
               # A is saved by the unsloth venv (newer transformers) but read by venv-q (4.57.6) for
               # B/C/eval: the merged tokenizer must give the same prompt ids and stop ids as the base.
               "$Q" - "$CFG" "$ADIR" <<'PY' || return
import json, sys
from transformers import AutoConfig, AutoTokenizer
from quantization.data_korquad import load_config
from quantization import v2_pipeline as V
cfg, mdir = load_config(sys.argv[1]), sys.argv[2]
st = V.chat_settings(cfg)
bm = cfg["base_model"]
tb = V.prepare_tokenizer(AutoTokenizer.from_pretrained(bm["selected"], revision=bm.get("revision")), st)
tm = V.prepare_tokenizer(AutoTokenizer.from_pretrained(mdir), st)
ex = V.load_slices(cfg, split="selection_dev")["eval"][:8]
ids = lambda t, e: t(V.render_prompt(t, e, st, True), add_special_tokens=False)["input_ids"]
same = all(ids(tb, e) == ids(tm, e) for e in ex)
rec = {"prompt_ids_same": same, "stop_ids_base": V.stop_token_ids(tb, st), "stop_ids_merged": V.stop_token_ids(tm, st),
       "pad_merged": tm.pad_token_id, "len_tok_base": len(tb), "len_tok_merged": len(tm),
       "vocab_size_merged": AutoConfig.from_pretrained(mdir).vocab_size}
print("[a_smoke] tokenizer check", json.dumps(rec))
sys.exit(0 if same and rec["stop_ids_base"] == rec["stop_ids_merged"] else 4)
PY
               park A "$ADIR" "${ADIR}_run"
               mv -f "$RES/A_train_seed$SEED.json" "$RES/A_train_seed${SEED}_smoke.json" ;;
    a)         "$A_PY" -m quantization.v2_run a --config "$CFG" --seed "$SEED" "${A_FLAG[@]}" "${RESUME_FLAG[@]}" ;;
    a_dev)     evalv A_bf16 selection_dev ;;
    gate)      "$Q" - "$RES" "$SEED" "${GATE_PT:-2.0}" <<'PY'
import json, sys
res, seed, need = sys.argv[1], sys.argv[2], float(sys.argv[3])
b = json.load(open(f"{res}/eval_vllm_base_seed{seed}_selection_dev.json"))
a = json.load(open(f"{res}/eval_vllm_A_bf16_seed{seed}_selection_dev.json"))
assert b.get("gpu_label") == a.get("gpu_label"), f"gate needs one GPU: {b.get('gpu_label')} vs {a.get('gpu_label')}"
d = a["accuracy"] - b["accuracy"]
rec = {"base_dev": b["accuracy"], "A_dev": a["accuracy"], "delta_pt": round(d, 3),
       "threshold_pt": need, "pass": d >= need, "gpu_label": a.get("gpu_label")}
json.dump(rec, open(f"{res}/gate_dev_seed{seed}.json", "w"), indent=2)
print(json.dumps(rec))
sys.exit(0 if rec["pass"] else 3)
PY
               ;;
    b)         "$Q" -m quantization.v2_run b --config "$CFG" --seed "$SEED" ;;
    c_smoke)   "$Q" -m quantization.v2_run c --config "$CFG" --seed "$SEED" --max-steps 5 || return
               park C "$ART/C_int4_qat_seed$SEED" "$ART/C_int4_qat_seed${SEED}_bf16adapt" \
                      "$ART/C_int4_qat_seed${SEED}_run"
               mv -f "$RES/C_train_seed$SEED.json" "$RES/C_train_seed${SEED}_smoke.json" ;;
    c)         "$Q" -m quantization.v2_run c --config "$CFG" --seed "$SEED" "${RESUME_FLAG[@]}" ;;
    c_export)  "$Q" -m quantization.v2_run c --config "$CFG" --seed "$SEED" --export-only ;;
    publish_A) "$Q" -m quantization.publish_hf --kind A_bf16 --dir "$ART/A_bf16_seed$SEED" \
                 --repo HyeonSang/Midm-2.0-KorMedMCQA --train-json "$RES/A_train_seed$SEED.json" ;;
    publish_C) "$Q" -m quantization.publish_hf --kind C_bf16adapt --dir "$ART/C_int4_qat_seed${SEED}_bf16adapt" \
                 --repo HyeonSang/Midm-2.0-KorMedMCQA-QAT-bf16 --train-json "$RES/C_train_seed$SEED.json" ;;
    fetch_hf)  "$HOME/venv-q/bin/hf" download HyeonSang/Midm-2.0-KorMedMCQA --local-dir "$ART/A_bf16_seed$SEED" &&
               "$HOME/venv-q/bin/hf" download HyeonSang/Midm-2.0-KorMedMCQA-QAT-bf16 \
                 --local-dir "$ART/C_int4_qat_seed${SEED}_bf16adapt" ;;
    sweep_smoke)  # 5 steps + merge + a 2021 dry run of 20 items; everything is parked to *_smoke, pass or fail
               for p in "${ADIR}_run" "$ADIR" "$RES/ckpt_dev"; do   # never park a real run's outputs
                 [ -e "$p" ] && { echo "sweep_smoke must run before the arm: $p exists"; return 64; }; done
               "$U" -m quantization.v2_run a --config "$CFG" --seed "$SEED" --max-steps 5; r1=$?
               r2=1; r3=1
               if [ $r1 -eq 0 ]; then
                 "$Q" -m quantization.eval_vllm --config "$CFG" --method base --seed "$SEED" --split selection_dev \
                   --eval-size 20 --lora-dir "${ADIR}_run/checkpoint-5"; r2=$?
                 "$Q" -m quantization.v2_run merge --config "$CFG" --seed "$SEED" --adapter "${ADIR}_run/checkpoint-5"; r3=$?
               fi
               park S "${ADIR}_run" "$ADIR" "$RES/ckpt_dev"
               rm -rf "${ADIR}_smoke"     # the 5-step merged model (23 GB) is not needed after the check
               for f in "$RES/A_train_seed$SEED.json" "$RES"/A_merge_seed${SEED}_ckpt5.json; do
                 [ -e "$f" ] && mv -f "$f" "${f%.json}_smoke.json"; done
               echo "smoke: train=$r1 lora_dev=$r2 merge=$r3"; [ $r1 -eq 0 ] && [ $r2 -eq 0 ] && [ $r3 -eq 0 ] ;;
    sweep_a)   [ -d "${ADIR}_run" ] && [ -z "${RESUME:-}" ] && { echo "${ADIR}_run exists (use RESUME=1 or move it)"; return 1; }
               "$U" -m quantization.v2_run a --config "$CFG" --seed "$SEED" "${RESUME_FLAG[@]}" ;;
    sweep_dev) mapfile -t CK < <(ls -d "${ADIR}_run"/checkpoint-* 2>/dev/null | sort -V)
               [ "${#CK[@]}" -gt 0 ] || { echo "no adapter checkpoints in ${ADIR}_run"; return 1; }
               LA=(); for c in "${CK[@]}"; do LA+=(--lora-dir "$c"); done
               evalv base selection_dev "${LA[@]}" --mode "${MODE:-answer}" ;;
    sweep_merge) case "${ADAPTER:-}" in "${ADIR}_run"/checkpoint-*) ;; *)
                 echo "ADAPTER must be under ${ADIR}_run/ (got '${ADAPTER:-}')"; return 64;; esac
               step=${ADAPTER##*checkpoint-}; step=${step%/}
               "$Q" -m quantization.v2_run merge --config "$CFG" --seed "$SEED" --adapter "$ADAPTER" &&
               evalv A_bf16 selection_dev --out "$RES/eval_vllm_A_bf16_seed${SEED}_ckpt${step}_selection_dev.json" ;;
    self_explain) LIM=(); [ -n "${LIMIT:-}" ] && LIM=(--limit "$LIMIT")
               "$Q" -m quantization.self_explain --config "$CFG" --seed "$SEED" "${LIM[@]}" ;;
    stop)      [ -n "${STOP:-}" ] && [ -f "$WORK/state/$STOP.running" ] || { echo "set STOP=<stage[_TAG]> of a running stage"; return 1; }
               pid=$(cat "$WORK/state/$STOP.running"); pg=$(ps -o pgid= -p "$pid" | tr -d ' ')
               [ -n "$pg" ] && kill -TERM -- "-$pg" && echo "sent TERM to process group $pg ($STOP)"
               rm -f "$WORK/state/$STOP.running" ;;
    sweep_report) "$Q" -m quantization.sweep_report ;;
    final_base) evalv base final_holdout --out "$RES/eval_vllm_base_seed${SEED}_${GPU_TAG}.json" ;;
    final_A)   case "$CFG" in quantization/configs/sweep/*)   # one sweep model is measured on 2022-2024, once
                 if ls quantization/results/kormedmcqa_midm/sweep/*/eval_vllm_*_"${GPU_TAG}".json >/dev/null 2>&1 \
                    && [ -z "${FORCE_FINAL:-}" ]; then
                   echo "a sweep model already has a 2022-2024 result; set FORCE_FINAL=1 only if that run was invalid"; return 64
                 fi ;; esac
               evalv A_bf16 final_holdout --out "$RES/eval_vllm_A_bf16_seed${SEED}_${GPU_TAG}.json" ;;
    final_B)   evalv B_int4_ptq final_holdout --out "$RES/eval_vllm_B_int4_ptq_seed${SEED}_${GPU_TAG}.json" ;;
    final_C)   evalv C_int4_qat final_holdout --out "$RES/eval_vllm_C_int4_qat_seed${SEED}_${GPU_TAG}.json" ;;
    compare)   "$Q" - "$RES" "$SEED" "$GPU_TAG" <<'PY'
import json, sys
from quantization.kormedmcqa import paired_compare
res, seed, tag = sys.argv[1:4]
f = lambda m: json.load(open(f"{res}/eval_vllm_{m}_seed{seed}_{tag}.json"))
pairs = [("base", "A_bf16"), ("A_bf16", "B_int4_ptq"), ("A_bf16", "C_int4_qat"), ("B_int4_ptq", "C_int4_qat")]
out = {"gpu_tag": tag, "split": "final_holdout", "pairs": []}
for a, b in pairs:
    A, B = f(a), f(b)
    r = paired_compare(A["items"], B["items"])
    r.update({"a": a, "b": b, "acc_a": A["accuracy"], "acc_b": B["accuracy"]})
    out["pairs"].append(r)
    print(json.dumps(r, ensure_ascii=False))
json.dump(out, open(f"{res}/compare_final_seed{seed}_{tag}.json", "w"), ensure_ascii=False, indent=2)
PY
               ;;
    bench_A)   "$Q" quantization/v2_bench.py --config "$CFG" --model-dir "$ART/A_bf16_seed$SEED" --method A_bf16 \
                 --precision bf16 --max-model-len 2048 --out "$RES/bench_A_bf16_seed${SEED}_${GPU_TAG}.json" ;;
    bench_B)   "$Q" quantization/v2_bench.py --config "$CFG" --model-dir "$ART/B_int4_ptq_seed$SEED" --method B_int4_ptq \
                 --precision int4 --max-model-len 2048 --out "$RES/bench_B_int4_ptq_seed${SEED}_${GPU_TAG}.json" ;;
    bench_C)   "$Q" quantization/v2_bench.py --config "$CFG" --model-dir "$ART/C_int4_qat_seed$SEED" --method C_int4_qat \
                 --precision int4 --max-model-len 2048 --out "$RES/bench_C_int4_qat_seed${SEED}_${GPU_TAG}.json" ;;
    online_A)  online A_bf16 "$ADIR" "" ;;
    online_B)  online B_int4_ptq "$ART/B_int4_ptq_seed$SEED" torchao ;;
    *) echo "unknown stage: $STAGE"; return 64 ;;
  esac
}

t0=$(date +%s)
run; rc=$?
dt=$(( $(date +%s) - t0 ))
rm -f "$ST.running"
if [ $rc -eq 0 ]; then
  echo "$(date -u +%FT%TZ) ${dt}s" >"$ST.done"; echo "== $STAGE done in ${dt}s"
else
  echo "$(date -u +%FT%TZ) ${dt}s rc=$rc" >"$ST.failed"; echo "== $STAGE FAILED rc=$rc after ${dt}s"
fi
exit $rc
