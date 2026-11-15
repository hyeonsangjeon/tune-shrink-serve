#!/usr/bin/env bash
# Open an Azure Bastion tunnel to the GPU VM's SSH port, then use plain ssh/scp on localhost.
#
#   SUB=<subscription name> RG=<rg> BASTION=<bastion name> VM=<vm name> [PORT=2222] scripts/gpu/bastion_tunnel.sh
#   ssh -p 2222 azureuser@127.0.0.1            # in another terminal while the tunnel runs
#
# Why the separate config dir: `az network bastion tunnel` asks the Bastion for a session token
# with the CLI's *default* account. When the default subscription is in another tenant the
# Bastion answers 403 "Bastion cannot be located by given credentials". This script points
# AZURE_CONFIG_DIR at a private copy of azureProfile.json whose default is $SUB and symlinks the
# rest (credential cache included, nothing copied), so the global `az` default is untouched.
set -euo pipefail
: "${SUB:?set SUB}" "${RG:?set RG}" "${BASTION:?set BASTION}" "${VM:?set VM}"
PORT=${PORT:-2222}
CFG=${TMPDIR:-/tmp}/azcfg-bastion-$(printf '%s' "$SUB" | tr -c 'A-Za-z0-9' '_')
SRC=${AZURE_CONFIG_DIR:-$HOME/.azure}

rm -rf "$CFG"; mkdir -p "$CFG"
for f in "$SRC"/* "$SRC"/.[!.]*; do
  [ -e "$f" ] || continue
  b=$(basename "$f"); [ "$b" = "azureProfile.json" ] || ln -s "$f" "$CFG/$b"
done
python3 - "$SRC/azureProfile.json" "$CFG/azureProfile.json" "$SUB" <<'PY'
import json, sys
src, dst, sub = sys.argv[1:4]
d = json.loads(open(src, encoding="utf-8-sig").read())
hits = 0
for s in d.get("subscriptions", []):
    s["isDefault"] = s.get("name") == sub or s.get("id") == sub
    hits += s["isDefault"]
if hits != 1:
    sys.exit(f"subscription {sub!r} not found (or not unique) in {src}; run `az login` for its tenant")
json.dump(d, open(dst, "w"))
PY
export AZURE_CONFIG_DIR=$CFG
VMID=$(az vm show -g "$RG" -n "$VM" --query id -o tsv)
echo "tunnel: localhost:$PORT -> $VM:22 via $BASTION (Ctrl+C to close)"
exec az network bastion tunnel -g "$RG" -n "$BASTION" --target-resource-id "$VMID" --resource-port 22 --port "$PORT"
