"""Interactive Hugging Face login on the GPU VM, tolerant of terminal paste artefacts.

Run it yourself over SSH (the token never leaves your keyboard):
    ssh -t -p 2222 azureuser@127.0.0.1 'HF_HOME=~/work/hf ~/venv-q/bin/python ~/work/repo/scripts/gpu/hf_login.py'

`hf auth login` failed on 2026-11-11 with "400 Bad request" when the token was pasted into the
hidden prompt: some terminals wrap pastes in bracketed-paste escape codes (ESC[200~ ... ESC[201~),
which end up inside the token and break the Authorization header. This strips those codes and
whitespace, never prints the token, does not store it as a git credential, and reports the account.
"""
import getpass
import re
import sys

raw = getpass.getpass("Hugging Face token (input hidden): ")
token = re.sub(r"\x1b\[20[01]~", "", raw)
token = re.sub(r"[\x00-\x1f\x7f\s]", "", token)
removed = len(raw) - len(token)
if removed:
    print(f"(removed {removed} hidden/whitespace characters from the pasted value)")
if not token.startswith("hf_"):
    sys.exit(f"that does not look like an HF token (expected it to start with 'hf_', got {len(token)} chars)")

from huggingface_hub import login, whoami  # noqa: E402

login(token=token, add_to_git_credential=False)
print("logged in as:", whoami().get("name"))
