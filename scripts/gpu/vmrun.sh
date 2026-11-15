#!/usr/bin/env bash
# Run a command on the GPU VM as its login user through `az vm run-command` (no inbound port).
# Use this when SSH is not available (e.g. a security automation strips internet SSH rules).
#
#   SUB=<subscription> RG=<resource group> VM=<vm name> scripts/gpu/vmrun.sh '<command>'
#   ... vmrun.sh --put <local file> <remote path>      # copy a small text file (< ~200 KB)
#
# The command travels base64-encoded and runs under `bash -l` as $VMUSER, so quotes, newlines
# and heredocs survive (run-command itself may use /bin/sh). The reply keeps only the last ~4 KB
# of output, so prefer scripts/gpu/status.sh and log tails. Calls are serialized per VM.
set -euo pipefail
: "${SUB:?set SUB}" "${RG:?set RG}" "${VM:?set VM}"
VMUSER=${VMUSER:-azureuser}

if [ "${1:-}" = "--put" ]; then
  src=${2:?local file}; dst=${3:?remote path}
  # Write to a temp file and rename: a running `bash stage.sh` keeps reading its old inode.
  # Overwriting in place made a running stage die after training (2026-11-10, C stage).
  cmd="mkdir -p \"\$(dirname '$dst')\" && base64 -d > '$dst.tmp.\$\$' <<'B64'
$(base64 <"$src")
B64
chmod --reference='$dst' '$dst.tmp.\$\$' 2>/dev/null || chmod 755 '$dst.tmp.\$\$'
mv -f '$dst.tmp.\$\$' '$dst' && md5sum '$dst'"
else
  cmd=${1:?command}
fi
enc=$(printf '%s\n' "$cmd" | base64 | tr -d '\n')
az vm run-command invoke --subscription "$SUB" -g "$RG" -n "$VM" --command-id RunShellScript \
  --scripts "echo $enc | base64 -d | sudo -u $VMUSER -H bash -l" --query "value[0].message" -o tsv
