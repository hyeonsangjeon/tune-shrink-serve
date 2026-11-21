# 2. Prepare the GPU VM
[한국어](02_gpu_setup.md) | **English**

Create a Linux VM with one GPU and install two virtual environments. Below is an example based on the Azure setup used this time. Another cloud or your own server also works, as long as it meets the conditions in the table.

| Item | This run | Notes |
|---|---|---|
| GPU | 1 × H100 NVL 94GB | On an 80GB-class GPU, reduce the QAT batch ([doc 4](04_int4_en.md#qat-memory)) |
| Driver | R580 or later (580.178.04, CUDA 13.0) | Required by the torch cu130 build |
| OS | Ubuntu 22.04, Python 3.10 | If `python3.10-venv` is missing, the install script installs it |
| Disk | 512GB or more, a managed disk attached to the VM | This run used about 320GB besides the OS. The local temporary NVMe is wiped when the VM is deallocated, so it is not used |
| Outbound access | huggingface.co, pypi.org, download.pytorch.org | The model (about 23GB) and the dataset are not gated, so they download without an HF token |
| Cost | $6.98 per hour on demand (Azure H100 NVL, list price at the time of this run) | When you are not using the VM, **deallocate** it. If you only stop it, billing continues |

## Creating it on Azure

- **Size:** `Standard_NC40ads_H100_v5` (1 × H100 NVL 94GB)
- **Image:** `microsoft-dsvm:ubuntu-hpc:2204`. It includes the driver and CUDA (R580 from the 2025-11 release onward).
- **Security type:** this image does not support Trusted Launch, so create the VM with `--security-type Standard`.
- **Organization subscriptions:** the two items below may need a separate subscription feature registration. Ask your administrator.
  - Creating a public IP: `Microsoft.Network/AllowBringYourOwnPublicIpAddress`
  - `--security-type Standard`: `Microsoft.Compute/UseStandardSecurityType`
- **On-demand vs Spot:** on-demand is better if you do not want the VM reclaimed during training. With Spot, continue training with `RESUME=1` (LoRA leaves a checkpoint every 100 steps, QAT every 150 steps).

## Connecting

- **If SSH is open,** use it as is.
- **If your organization's security policy deletes rules that allow inbound SSH from the internet,** do not add the rule back; use one of the two options below.
  - **Azure Bastion (Standard, tunneling enabled):** open a tunnel with [`scripts/gpu/bastion_tunnel.sh`](../../scripts/gpu/bastion_tunnel.sh), then `ssh -p 2222 azureuser@127.0.0.1`.
    - If the NSG on the Bastion subnet has only the default rules, you must add the [NSG rules from the Bastion docs](https://learn.microsoft.com/azure/bastion/bastion-nsg) before it can connect.
    - If the az CLI's default subscription is in another tenant, it fails with `403 Bastion cannot be located by given credentials`. The script changes the default subscription only in that shell.
  - **`az vm run-command`:** needs no inbound port. Send a command with [`scripts/gpu/vmrun.sh`](../../scripts/gpu/vmrun.sh) `'<command>'`, and move a small file with `vmrun.sh --put <file> <path>`.

## Install

Inside the VM, put the repo in `~/work/repo` and run:

```bash
mkdir -p ~/work && git clone https://github.com/hyeonsangjeon/tune-shrink-serve.git ~/work/repo && cd ~/work/repo
bash scripts/gpu/bootstrap.sh
```

What [`bootstrap.sh`](../../scripts/gpu/bootstrap.sh) does:
- Creates two virtual environments.
  - `~/venv-q`: for INT4 conversion, scoring, serving measurements and transformers+PEFT training ([`requirements/gpu.txt`](../../requirements/gpu.txt))
  - `~/venv-unsloth`: for Unsloth training ([`requirements/unsloth.txt`](../../requirements/unsloth.txt)). It is kept separate because Unsloth changes the installed transformers
- Stops if the driver is older than R580, if torch is not the cu130 build, or if it cannot see the GPU.
- Runs the CPU tests.
- Records the installed versions in `quantization/results/kormedmcqa_midm/env_<gpu>.json` (e.g. `env_h100.json`).

It is fine to run it more than once. The log goes to `~/work/logs/bootstrap.log`.

## Verify

```bash
cat quantization/results/kormedmcqa_midm/env_h100.json   # GPU name, driver, versions in both virtual environments
bash scripts/gpu/status.sh                                # GPU usage, running stage, recent log
```

It is fine if `venv_q.torch` ends with `+cu130` and the versions match [`env_h100.json`](../../quantization/results/kormedmcqa_midm/env_h100.json) (this run).

## When it fails

- **`venv-q torch is ..., not the cu130 build`:** another package replaced torch with a different build. Delete `~/venv-q` and run again.
- **`torch cannot see the GPU`:** check the driver (`nvidia-smi`). If the image has a driver older than R580, create the VM from a newer image.
- **Unsloth cannot read the Mi:dm tokenizer:** use the pinned versions in `requirements/unsloth.txt` as they are. Details are in [doc 8](08_troubleshooting_en.md).

## Running a stage (same for step 3 onward)

```bash
bash scripts/gpu/stage.sh <stage>             # runs in the foreground
bash scripts/gpu/stage.sh <stage> --detach    # runs in the background (keeps going if SSH drops)
bash scripts/gpu/status.sh                    # progress
```

- The log goes to `~/work/logs/<stage>.log`, and whether it finished is recorded in `~/work/state/<stage>.done` or `.failed`.
- To record prices with the results, put `PRICES="--price ondemand=6.98 --price spot=3.537"` in front.

Next: [3. LoRA training](03_finetune_en.md)
