# scripts/

[한국어](README.md) | **English**

## Used on the GPU VM (`gpu/`)

| Script | What it does | Step-by-step guide |
|---|---|---|
| [`gpu/bootstrap.sh`](gpu/bootstrap.sh) | Installs two virtual environments (`~/venv-q`, `~/venv-unsloth`), checks the driver, GPU and tests, records the installed versions | 2 |
| [`gpu/stage.sh`](gpu/stage.sh) | Runs one stage and leaves a log (`~/work/logs/`) and a finished-state record (`~/work/state/`). With `--detach`, it runs in the background | 3-7 |
| [`gpu/status.sh`](gpu/status.sh) | GPU usage, the running stage, recent logs | 2 |
| [`gpu/sweep_queue.sh`](gpu/sweep_queue.sh) | A queue that runs the stages of the experiments to raise the score one after another on a single GPU | 7 |
| [`gpu/bastion_tunnel.sh`](gpu/bastion_tunnel.sh) | (Azure) Opens a Bastion tunnel when SSH is blocked | 2 |
| [`gpu/vmrun.sh`](gpu/vmrun.sh) | (Azure) Sends commands with `az vm run-command`, without any inbound port | 2 |
| [`gpu/hf_login_vm.sh`](gpu/hf_login_vm.sh), [`gpu/hf_login.py`](gpu/hf_login.py) | Hugging Face login on the VM (you enter the token yourself). Needed only when uploading weights | 4 |

## Used on a laptop (no GPU needed)

| Script | What it does | CI |
|---|---|---|
| [`build_evidence_index.py`](build_evidence_index.py) | Checks that the numbers in the docs equal the values in the result JSON (`--check`); rebuilds the evidence index `docs/EVIDENCE.md` (`--emit`) | ✓ |
| [`make_figures.py`](make_figures.py) | Draws the charts (`docs/talk/figures/*.svg`, and `*_en.svg` for the English docs) from the result JSON. `--check` checks that redrawing gives the same output | ✓ |
| [`scan_secrets.py`](scan_secrets.py) | Checks that no keys, tokens, endpoint addresses or personal paths got into a commit | ✓ |
| [`check_asset_ledger.py`](check_asset_ledger.py) | Checks that every committed file related to data or models is in the license record (`docs/asset_ledger.yaml`) | ✓ |
| [`make_slides.py`](make_slides.py) | Builds the talk slides and speaker script from the `docs/talk/` docs (`build/slides/`; needs `requirements/slides.txt`). Also checks that every number on the slides appears in the docs. `--lang en` builds the English deck (translations in [`slides_en.json`](slides_en.json); numbers checked against the English docs), `--high-contrast` turns gray text black (the deck used on the talk-room projector) | |
