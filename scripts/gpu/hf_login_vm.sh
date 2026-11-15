#!/usr/bin/env bash
# Put a Hugging Face write token on the GPU VM (for scripts/gpu/stage.sh publish_A / publish_C)
# without SSH. Run it yourself: the token is read from a hidden prompt, never echoed, and only
# written to the VM's HF_HOME (~/work/hf/token, mode 600), where stage.sh's publish stages look.
#
#   SUB=<subscription> RG=<resource group> VM=<vm name> scripts/gpu/hf_login_vm.sh
#   SUB=... RG=... VM=... scripts/gpu/hf_login_vm.sh --logout   # delete the token from the VM
#
# Use a token you can revoke right after the upload: `az vm run-command` keeps the script it ran
# on the VM (root-only, under /var/lib/waagent), so revoke the token on huggingface.co afterwards.
set -euo pipefail
: "${SUB:?set SUB}" "${RG:?set RG}" "${VM:?set VM}"
HERE=$(cd "$(dirname "$0")" && pwd)
HFH=/home/${VMUSER:-azureuser}/work/hf

if [ "${1:-}" = "--logout" ]; then
  "$HERE/vmrun.sh" "rm -f $HFH/token $HFH/stored_tokens && echo 'token removed from VM'"
  exit 0
fi

read -r -s -p "Hugging Face token (input hidden): " TOKEN; echo
[ -n "$TOKEN" ] || { echo "no token entered"; exit 1; }
enc=$(printf '%s' "$TOKEN" | base64 | tr -d '\n'); unset TOKEN
"$HERE/vmrun.sh" "umask 077 && mkdir -p $HFH && echo $enc | base64 -d > $HFH/token && \
HF_HOME=$HFH \$HOME/venv-q/bin/hf auth whoami 2>&1 | head -3"
unset enc
