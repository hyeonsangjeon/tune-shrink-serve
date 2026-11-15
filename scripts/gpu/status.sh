#!/usr/bin/env bash
# Short GPU VM status (fits an `az vm run-command` reply, which keeps ~4 KB of output).
#   bash scripts/gpu/status.sh [lines]     # lines = log tail per running/failed stage (default 6)
set -uo pipefail
WORK=${WORK:-$HOME/work}
N=${1:-6}
echo "time $(date -u +%FT%TZ)  disk_free $(df -h --output=avail "$HOME" | tail -1 | tr -d ' ')"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null \
  | sed 's/^/gpu  /'
shopt -s nullglob
for f in "$WORK"/state/*.running "$WORK"/state/*.done "$WORK"/state/*.failed; do
  s=$(basename "$f"); name=${s%.*}; kind=${s##*.}
  case $kind in
    running) if kill -0 "$(cat "$f")" 2>/dev/null; then echo "RUN  $name (pid $(cat "$f"))";
             else echo "DEAD $name (pid $(cat "$f") gone, no done/failed marker)"; fi ;;
    done)    echo "ok   $name  $(cat "$f")" ;;
    failed)  echo "FAIL $name  $(cat "$f")" ;;
  esac
done
for f in "$WORK"/state/*.running "$WORK"/state/*.failed; do
  name=$(basename "${f%.*}")
  echo "--- $name.log (last $N)"
  tail -n "$N" "$WORK/logs/$name.log" 2>/dev/null | cut -c1-300
done
