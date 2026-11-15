#!/usr/bin/env bash
# One-time GPU VM setup for the KorMedMCQA x Mi:dm run (idempotent: safe to re-run).
#
#   bash scripts/gpu/bootstrap.sh            # from the repo root on the VM, as the VM user
#
# Creates ~/venv-q (B/C/eval/bench, the recorded A100 stack) and ~/venv-unsloth (Method A),
# runs the CPU tests in venv-q, and writes an environment record to
# quantization/results/kormedmcqa_midm/env_<gpu>.json (e.g. env_h100.json). Everything lives on the OS disk
# (~/work); never under /mnt, which is wiped on deallocation.
# Output is kept short so it fits an `az vm run-command` reply; full log: ~/work/logs/bootstrap.log
set -uo pipefail

WORK=${WORK:-$HOME/work}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
LOG=$WORK/logs/bootstrap.log
mkdir -p "$WORK/logs" "$WORK/state" "$WORK/hf"
exec 3>&1 >>"$LOG" 2>&1
say() { echo "$*" >&3; echo "== $*"; }
die() { say "FAIL: $*"; touch "$WORK/state/bootstrap.failed"; exit 1; }
rm -f "$WORK/state/bootstrap.done" "$WORK/state/bootstrap.failed"

say "bootstrap start $(date -u +%FT%TZ) repo=$REPO"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
  || die "nvidia-smi failed (no GPU driver)"
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
[ "${DRV:-0}" -ge 580 ] || die "driver R$DRV < R580: torch cu130 will not run (upgrade the driver or use cu128 and record it)"

# `venv --help` works without ensurepip, so test ensurepip itself (ubuntu-hpc 2204 lacks it)
if ! python3.10 -c "import ensurepip" >/dev/null 2>&1; then
  sudo apt-get update -y && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3.10-venv \
    || die "python3.10-venv install"
fi
for v in venv-q venv-unsloth; do   # drop a half-made venv (no pip) left by an earlier failure
  if [ -d "$HOME/$v" ] && [ ! -x "$HOME/$v/bin/pip" ]; then rm -rf "${HOME:?}/$v"; fi
done

if [ ! -x "$HOME/venv-q/bin/python" ]; then python3.10 -m venv "$HOME/venv-q" || die "venv-q"; fi
"$HOME/venv-q/bin/pip" install -q -U pip || die "pip upgrade"
"$HOME/venv-q/bin/pip" install -q --index-url https://download.pytorch.org/whl/cu130 torch==2.11.0 \
  || die "torch 2.11.0 cu130"
"$HOME/venv-q/bin/pip" install -q -r "$REPO/requirements/gpu.txt" \
  || die "requirements/gpu.txt"
"$HOME/venv-q/bin/pip" install -q -r "$REPO/requirements/cpu.txt" || die "requirements/cpu.txt"
TV=$("$HOME/venv-q/bin/python" -c "import torch; print(torch.__version__)")
case "$TV" in *cu130*) ;; *) die "venv-q torch is $TV, not the cu130 build (a later pin replaced it)";; esac
"$HOME/venv-q/bin/python" -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))" \
  || die "torch cannot see the GPU"
(cd "$REPO" && "$HOME/venv-q/bin/python" -m pytest -q) || die "CPU tests failed in venv-q"

if [ ! -x "$HOME/venv-unsloth/bin/python" ]; then python3.10 -m venv "$HOME/venv-unsloth" || die "venv-unsloth"; fi
"$HOME/venv-unsloth/bin/pip" install -q -U pip || die "pip upgrade (unsloth)"
# Pin the HF stack to venv-q's versions. Unsloth's own resolve pulled transformers 5.17, which
# fails to load the Mi:dm tokenizer ("argument of type 'tokenizers.AddedToken' is not iterable",
# 2026-11-11), and A's merged dir is read by venv-q (4.57.6) anyway. unsloth 2026.11.3 accepts these.
"$HOME/venv-unsloth/bin/pip" install -q -r "$REPO/requirements/unsloth.txt" || die "requirements/unsloth.txt"
"$HOME/venv-unsloth/bin/python" -c "import unsloth, torch; assert torch.cuda.is_available()" \
  || die "unsloth import / GPU check"

TAG=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | grep -o -E "H100|A100|L40S|A10" | head -1 | tr "A-Z" "a-z")
OUT=$REPO/quantization/results/kormedmcqa_midm/env_${TAG:-gpu}.json
mkdir -p "$(dirname "$OUT")"
IMDS=$(curl -s -m 5 -H Metadata:true \
  "http://169.254.169.254/metadata/instance/compute?api-version=2021-02-01" || echo '{}')
"$HOME/venv-q/bin/python" - "$OUT" "$IMDS" "$HOME/venv-unsloth/bin/python" <<'PY' || die "env record"
import json, subprocess, sys
from importlib import metadata as md

out, imds, unsloth_py = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    c = json.loads(imds or "{}")
except Exception:
    c = {}
def ver(names):
    r = {}
    for n in names:
        try:
            r[n] = md.version(n)
        except md.PackageNotFoundError:
            r[n] = None
    return r
smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
cuda = next((l.split("CUDA Version:")[1].split()[0] for l in subprocess.run(
    ["nvidia-smi"], capture_output=True, text=True).stdout.splitlines() if "CUDA Version:" in l), None)
uns = subprocess.run([unsloth_py, "-c",
    "import json;from importlib import metadata as m;print(json.dumps({n:m.version(n) for n in "
    "['unsloth','unsloth_zoo','torch','transformers','peft','trl','datasets']}))"],
    capture_output=True, text=True).stdout.strip()
rec = {"schema": "gpu_environment/v1",
       "vm": {k: c.get(k) for k in ["vmSize", "location", "priority", "evictionPolicy"]},
       "image": (c.get("storageProfile") or {}).get("imageReference"),
       "gpu": smi, "nvidia_driver": smi.split(", ")[1] if ", " in smi else None, "cuda_driver_api": cuda,
       "python": sys.version.split()[0],
       "venv_q": ver(["torch", "torchao", "transformers", "peft", "trl", "bitsandbytes",
                      "accelerate", "datasets", "vllm", "huggingface-hub"]),
       "venv_unsloth": json.loads(uns) if uns.startswith("{") else uns}
json.dump(rec, open(out, "w"), indent=2)
print(json.dumps({"gpu": rec["gpu"], "cuda": cuda, "vm": rec["vm"],
                  "torch": rec["venv_q"]["torch"], "vllm": rec["venv_q"]["vllm"],
                  "unsloth": (rec["venv_unsloth"] or {}).get("unsloth") if isinstance(rec["venv_unsloth"], dict) else None}))
PY
say "env record: $OUT"
grep '^{"gpu"' "$LOG" | tail -1 >&3
touch "$WORK/state/bootstrap.done"
say "bootstrap done $(date -u +%FT%TZ)"
