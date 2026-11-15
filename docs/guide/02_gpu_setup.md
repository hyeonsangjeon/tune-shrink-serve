# 2. GPU VM 준비

GPU 한 장짜리 Linux VM을 만들고, 가상환경 두 개를 설치합니다. 아래는 이번에 쓴 Azure 기준 예시입니다. 다른 클라우드나 자기 서버여도 표의 조건만 맞으면 됩니다.

| 항목 | 이번 실행 | 메모 |
|---|---|---|
| GPU | H100 NVL 94GB 1장 | 80GB급이면 QAT 배치를 줄입니다([4번 문서](04_int4.md#qat-메모리)) |
| 드라이버 | R580 이상 (580.178.04, CUDA 13.0) | torch cu130 빌드가 요구합니다 |
| OS | Ubuntu 22.04, Python 3.10 | `python3.10-venv`가 없으면 설치 스크립트가 넣습니다 |
| 디스크 | 512GB 이상, VM에 붙는 관리 디스크 | 이번 실행은 OS 외에 약 320GB를 썼습니다. 로컬 임시 NVMe는 VM을 할당 해제하면 지워지므로 쓰지 않습니다 |
| 나가는 연결 | huggingface.co, pypi.org, download.pytorch.org | 모델(약 23GB)과 데이터셋은 접근 제한이 없어 HF 토큰 없이 받습니다 |
| 비용 | 온디맨드 시간당 $6.98 (Azure H100 NVL, 이번 정가) | 쓰지 않을 때는 VM을 **할당 해제**합니다. 멈춤만 하면 계속 청구됩니다 |

## Azure에서 만들 때

- **크기:** `Standard_NC40ads_H100_v5`(H100 NVL 94GB 1장)
- **이미지:** `microsoft-dsvm:ubuntu-hpc:2204`. 드라이버와 CUDA가 들어 있습니다(2025-11판부터 R580).
- **보안 유형:** 이 이미지는 Trusted Launch를 지원하지 않으므로 `--security-type Standard`로 만듭니다.
- **조직 구독:** 아래 두 가지에 구독 기능 등록이 따로 필요할 수 있습니다. 관리자에게 요청하세요.
  - 공용 IP 만들기: `Microsoft.Network/AllowBringYourOwnPublicIpAddress`
  - `--security-type Standard`: `Microsoft.Compute/UseStandardSecurityType`
- **온디맨드와 Spot:** 학습 도중 회수되지 않으려면 온디맨드가 낫습니다. Spot이면 `RESUME=1`로 이어서 학습합니다(LoRA는 100 step, QAT는 150 step마다 체크포인트를 남깁니다).

## 접속

- **SSH가 열리면** 그대로 씁니다.
- **조직 보안 정책이 인터넷에서 들어오는 SSH 규칙을 지우면**, 규칙을 다시 넣지 말고 아래 둘 중 하나를 씁니다.
  - **Azure Bastion(Standard, 터널링 켬):** [`scripts/gpu/bastion_tunnel.sh`](../../scripts/gpu/bastion_tunnel.sh)로 터널을 연 뒤 `ssh -p 2222 azureuser@127.0.0.1`.
    - Bastion 서브넷의 NSG가 기본 규칙만 있으면 [Bastion 문서의 NSG 규칙](https://learn.microsoft.com/azure/bastion/bastion-nsg)을 넣어야 연결됩니다.
    - az CLI의 기본 구독이 다른 테넌트에 있으면 `403 Bastion cannot be located by given credentials`로 실패합니다. 스크립트는 그 셸에서만 기본 구독을 바꿉니다.
  - **`az vm run-command`:** 들어오는 포트가 필요 없습니다. [`scripts/gpu/vmrun.sh`](../../scripts/gpu/vmrun.sh) `'<명령>'`으로 명령을 보내고, `vmrun.sh --put <파일> <경로>`로 작은 파일을 옮깁니다.

## 설치

VM 안에서 레포를 `~/work/repo`에 두고 실행합니다.

```bash
mkdir -p ~/work && git clone https://github.com/hyeonsangjeon/tune-shrink-serve.git ~/work/repo && cd ~/work/repo
bash scripts/gpu/bootstrap.sh
```

[`bootstrap.sh`](../../scripts/gpu/bootstrap.sh)가 하는 일:
- 가상환경 두 개를 만듭니다.
  - `~/venv-q`: INT4 변환, 채점, 서빙 측정, transformers+PEFT 학습용([`requirements/gpu.txt`](../../requirements/gpu.txt))
  - `~/venv-unsloth`: Unsloth 학습용([`requirements/unsloth.txt`](../../requirements/unsloth.txt)). Unsloth가 transformers를 바꿔 놓기 때문에 따로 둡니다
- 드라이버가 R580 미만이거나, torch가 cu130 빌드가 아니거나, GPU를 못 보면 멈춥니다.
- CPU 테스트를 돌립니다.
- 설치된 버전을 `quantization/results/kormedmcqa_midm/env_<gpu>.json`(예: `env_h100.json`)에 남깁니다.

여러 번 돌려도 됩니다. 로그는 `~/work/logs/bootstrap.log`에 남습니다.

## 확인

```bash
cat quantization/results/kormedmcqa_midm/env_h100.json   # GPU 이름, 드라이버, 두 가상환경의 버전
bash scripts/gpu/status.sh                                # GPU 사용량, 도는 단계, 최근 로그
```

`venv_q.torch`가 `+cu130`으로 끝나고, 버전이 [`env_h100.json`](../../quantization/results/kormedmcqa_midm/env_h100.json)(이번 실행)과 같으면 됩니다.

## 안 될 때

- **`venv-q torch is ..., not the cu130 build`:** 다른 패키지가 torch를 바꿔 깔았습니다. `~/venv-q`를 지우고 다시 돌립니다.
- **`torch cannot see the GPU`:** 드라이버를 확인합니다(`nvidia-smi`). 이미지가 R580 미만이면 새 이미지로 VM을 만듭니다.
- **Unsloth가 Mi:dm 토크나이저를 못 읽음:** `requirements/unsloth.txt`의 고정 버전을 그대로 씁니다. 자세한 것은 [8번 문서](08_troubleshooting.md).

## 단계 돌리는 법 (3번부터 공통)

```bash
bash scripts/gpu/stage.sh <단계>              # 앞에서 돌림
bash scripts/gpu/stage.sh <단계> --detach     # 백그라운드로 돌림 (SSH가 끊겨도 계속)
bash scripts/gpu/status.sh                    # 진행 상황
```

- 로그는 `~/work/logs/<단계>.log`, 끝났는지는 `~/work/state/<단계>.done` 또는 `.failed`에 남습니다.
- 가격을 결과에 함께 남기려면 `PRICES="--price ondemand=6.98 --price spot=3.537"`을 앞에 붙입니다.

다음: [3. LoRA 학습](03_finetune.md)
