#!/usr/bin/env bash
# Recipe-sweep queue for the one-GPU VM: runs stage.sh stages one after another, waits until the
# GPU is empty before each, checks free disk, and stops at the first failure (the GPU neither idles
# between stages nor runs two jobs at once). Selection uses the 2021 questions only.
#
# Copy it OUT of the repo before running (a running bash reads its script as it goes):
#   cp ~/work/repo/scripts/gpu/sweep_queue.sh ~/work/sweep_queue.sh
#   setsid nohup bash ~/work/sweep_queue.sh q0 </dev/null >>~/work/logs/queue_q0.log 2>&1 &
# Stop the whole queue and its Python/vLLM children:  kill -TERM -- -<pid printed by setsid>
#
#   q0     ~15 min  smoke of the new paths (S2 sequential views, S8 per-example loss + explanations,
#                   vLLM+LoRA answer/explain scoring) and a 100-item explanation pilot (keep rate)
#   q1     ~3 h     reference adapter through vLLM+LoRA, S0 (seed 43) + its merged score, S1, S2,
#                   explanations for all 3,401 training items, S8 (+ explain-mode scoring)
#   q2     ~2 h     S3, S4, S5 (set their `extends` to the better of S1/S2 first), sweep_report
#   final           merge the picked checkpoint: PICK_CFG=<sweep config> PICK_CKPT=<checkpoint dir>
set -uo pipefail
cd "$HOME/work/repo" || exit 1
export HF_HOME=${HF_HOME:-$HOME/work/hf} VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONUNBUFFERED=1
ST="bash scripts/gpu/stage.sh"
SW=quantization/configs/sweep
RS=quantization/results/kormedmcqa_midm/sweep
AR=quantization/artifacts/kormedmcqa_midm/sweep
Q=$HOME/venv-q/bin/python
U=$HOME/venv-unsloth/bin/python

gpu_free() {   # wait up to 5 min for < 2 GB used and no compute process on the GPU
  local used apps
  for _ in $(seq 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
    [ "$used" -lt 2000 ] && [ "$apps" -eq 0 ] && return 0
    sleep 5
  done
  echo "GPU not free: ${used} MiB used, ${apps} process(es)"; nvidia-smi; return 1
}
disk_ok() {    # disk_ok <GB>: stop before a stage that could fill the OS disk mid-save
  local free; free=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc 0-9)
  [ "$free" -ge "$1" ] || { echo "only ${free}G free on \$HOME, need $1G"; exit 1; }
}
step() {       # step <stage> VAR=value ...  (foreground; the queue stops if the stage fails)
  local stage=$1; shift
  gpu_free || exit 1
  echo "$(date -u +%FT%TZ) >> $stage $*"
  env "$@" $ST "$stage" || { echo "$(date -u +%FT%TZ) queue STOPPED at $stage $*"; exit 1; }
}
arm() {        # arm <config name> [seed]: train, then score every saved adapter on 2021
  local a=$1 s=${2:-42}
  [ -d "$AR/$a/A_bf16_seed${s}_run" ] && { echo "$a: run dir exists; move it aside first"; exit 1; }
  disk_ok 40
  step sweep_a   CONFIG=$SW/$a.yaml SEED=$s TAG=$a
  step sweep_dev CONFIG=$SW/$a.yaml SEED=$s TAG=$a MODE=answer
}
scratch_s8() { # scratch config: S8 on the 100-item pilot explanations, outside results/.../sweep
  mkdir -p "$HOME/work/sweep_smoke"
  cat > "$HOME/work/sweep_smoke/s8_pilot.yaml" <<EOF
extends: $HOME/work/repo/$SW/s8_self_explain.yaml
data:
  self_explain:
    path: "$HOME/work/repo/quantization/artifacts/kormedmcqa_midm/self_explain/train_explanations_seed42_n100.jsonl"
paths:
  results_dir: "$HOME/work/sweep_smoke/results"
  method_a_dir: "$HOME/work/sweep_smoke/s8/A_bf16"
EOF
}

case "${1:?usage: sweep_queue.sh q0|q1|q2|final}" in
  q0)
    disk_ok 60
    step sweep_smoke CONFIG=$SW/s2_perm4.yaml TAG=s2_perm4          # sequential views, final adapter, merge
    step self_explain CONFIG=$SW/s8_self_explain.yaml TAG=pilot LIMIT=100
    "$Q" - "$RS/s8_self_explain/self_explain_stats_seed42_n100.json" <<'PY' || exit 1
import json, sys
s = json.load(open(sys.argv[1]))
print("pilot kept", s["n_kept"], "of", s["n_train"], s["kept_by_role"], "truncated", s["samples_truncated"],
      "no final line", s["samples_without_final_line"], "chars", s["explanation_chars_mean"])
sys.exit(0 if s["n_kept"] >= 25 else 1)      # under 25% kept: fix the prompt before the full run
PY
    scratch_s8; gpu_free || exit 1
    rm -rf "$HOME/work/sweep_smoke/s8"
    "$U" -m quantization.v2_run a --config "$HOME/work/sweep_smoke/s8_pilot.yaml" --seed 42 --max-steps 5 || exit 1
    "$Q" -m quantization.eval_vllm --config "$HOME/work/sweep_smoke/s8_pilot.yaml" --method base --split selection_dev \
      --eval-size 20 --mode explain --lora-dir "$HOME/work/sweep_smoke/s8/A_bf16_seed42_run/checkpoint-5" || exit 1 ;;
  q1)
    # the reference adapter scored the same way as the arms (vLLM + LoRA): measures the path gap
    if ls -d quantization/artifacts/kormedmcqa_midm/A_bf16_unsloth_seed42_run/checkpoint-* >/dev/null 2>&1; then
      step sweep_dev CONFIG=quantization/configs/kormedmcqa_midm_unsloth.yaml TAG=reference MODE=answer
    else
      echo "reference adapter checkpoints not found; the path gap will come from S0 (adapter vs merged)"
    fi
    arm s0_seed43 43
    step a_dev CONFIG=$SW/s0_seed43.yaml SEED=43 TAG=s0_seed43     # S0's merged final model on 2021
    arm s1_ep4
    arm s2_perm4
    step self_explain CONFIG=$SW/s8_self_explain.yaml TAG=s8_self_explain
    "$Q" - "$RS/s8_self_explain/self_explain_stats_seed42.json" <<'PY' || exit 1
import json, sys
s = json.load(open(sys.argv[1]))
print("self_explain kept", s["n_kept"], "of", s["n_train"], s["kept_by_role"])
sys.exit(0 if s["n_kept"] >= 1000 else 1)      # too few explanations: do not spend GPU time on S8
PY
    arm s8_self_explain
    step sweep_dev CONFIG=$SW/s8_self_explain.yaml SEED=42 TAG=s8_self_explain_explain MODE=explain
    step sweep_report ;;
  q2)
    arm s3_lr1e-4
    arm s4_lr4e-4
    arm s5_roles_t2
    step sweep_report ;;
  final)  # after sweep_report and the user's pick
    disk_ok 30
    step sweep_merge CONFIG="${PICK_CFG:?set PICK_CFG}" SEED="${PICK_SEED:-42}" TAG=final ADAPTER="${PICK_CKPT:?set PICK_CKPT}" ;;
  *) echo "unknown queue $1"; exit 64 ;;
esac
echo "$(date -u +%FT%TZ) queue $1 done"
