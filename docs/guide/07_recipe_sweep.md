# 7. (선택) 점수 올리기 실험 — 레시피 스윕

**한국어** | [English](07_recipe_sweep_en.md)

Unsloth로 학습한 BF16 LoRA 모델(아래와 [5번 결과 문서](../talk/05_recipe_sweep.md)에서 "지금 LoRA")의 학습 방법을 한 가지씩 바꿔, 2021년 문제로 정답률이 오르는지 봅니다. 다른 모델의 출력은 쓰지 않습니다. 발표 본편의 숫자를 다시 만드는 데는 필요 없습니다.

| 필요한 것 | 걸린 시간 (H100 NVL) | 결과 문서 |
|---|---|---|
| [3번](03_finetune.md)의 Unsloth 학습(`unsloth_A`)과 그 체크포인트 | 큐 q0 약 15분, q1 약 3시간, q2 약 2시간(스크립트에 적힌 예상) | [5. 점수 올리기](../talk/05_recipe_sweep.md) |

## 무엇을 돌리나

설정은 [`quantization/configs/sweep/`](../../quantization/configs/sweep/)에 있고, 모두 Unsloth 설정(`kormedmcqa_midm_unsloth.yaml`)을 물려받습니다. 각 설정 파일의 머리 주석에 그 실험의 설명이 있습니다. 파일 이름의 `s0`~`s8`은 실험 번호입니다.

| 설정 파일 | 바꾼 것 |
|---|---|
| `s0_seed43.yaml` | 레시피는 그대로, seed만 43. 레시피를 비교하기 전에 seed만 바꿨을 때 점수가 얼마나 달라지는지 잽니다 |
| `s1_ep4.yaml` | epoch 2 → 4. epoch마다 어댑터를 남깁니다 |
| `s2_perm4.yaml` | 4 epoch + epoch마다 보기 순서를 바꿔 학습(정답 번호도 따라 바꿈) |
| `s3_lr1e-4.yaml` | 보기 섞기 + 학습률 1e-4 |
| `s4_lr4e-4.yaml` | 보기 섞기 + 학습률 4e-4 |
| `s5_roles_t2.yaml` | 보기 섞기 + 직군별로 보는 횟수 조정(온도 2) |
| `s6_plw01.yaml` (선택, 이번에 안 돌림) | 보기 섞기 + 질문과 보기에도 0.1 무게로 손실 |
| `s7_neftune5.yaml` (선택, 이번에 안 돌림) | 보기 섞기 + NEFTune 잡음 alpha 5 |
| `s8_self_explain.yaml` | 2 epoch + 튜닝 전 Mi:dm이 직접 쓴 해설을 학습 예시로 추가(최대 길이 1,536토큰). 서빙은 그대로 번호만 답함 |
| `refit_train_dev.yaml` (이번에 안 돌림) | 고른 레시피를 2012~2021년 문제(4,460문항)로 다시 학습. 2021년 문제가 학습에 들어가므로 2022~2024년 문제로만 한 번 잴 수 있습니다 |

`s3`·`s4`·`s5`는 `s1`과 `s2` 중 2021년 점수가 높은 쪽을 물려받습니다(`extends`, 지금은 `s2_perm4.yaml`). `s1`이 더 높으면 q2를 돌리기 전에 세 설정의 `extends`를 바꿉니다.

## 고르는 규칙과 통과 기준

규칙은 실행 전에 [`quantization/sweep_report.py`](../../quantization/sweep_report.py)에 정해 두었습니다.

- **2021년 문제로만 고릅니다.** `sweep_report`는 2022~2024년 결과를 읽지 않습니다.
- **실험 안에서:** 2021년 점수가 가장 높은 체크포인트를 고릅니다. 가장 높은 점수와 0.5%p 안이면 앞쪽(덜 학습한 쪽)을 고릅니다.
- **통과 기준:** 지금 LoRA보다 2.0%p 이상 높고, 같은 문항 비교(McNemar)의 p가 0.05 미만이어야 합니다.
- **2022~2024년 문제는 통과한 모델 하나만, 한 번 잽니다.**
- **p는 명목값입니다.** 2021년 점수를 몇 번 봤는지는 `dev_scorings_looked_at`에 남습니다.

## 순서

큐 스크립트 [`scripts/gpu/sweep_queue.sh`](../../scripts/gpu/sweep_queue.sh)가 `stage.sh` 단계를 차례로 돌립니다. 단계마다 GPU가 비었는지와 디스크 여유를 확인하고, 하나라도 실패하면 멈춥니다. bash는 스크립트를 읽어 가며 실행하므로, 큐 스크립트는 레포 밖으로 복사해서 돌립니다.

```bash
cp ~/work/repo/scripts/gpu/sweep_queue.sh ~/work/sweep_queue.sh
setsid nohup bash ~/work/sweep_queue.sh q0 </dev/null >>~/work/logs/queue_q0.log 2>&1 &
# q0가 끝나면 같은 방식으로 q1, 그다음 q2
```

| 큐 | 하는 일 |
|---|---|
| `q0` | 새 코드 경로 시험(보기 섞기의 순서대로 읽기, 해설 학습의 손실, vLLM+어댑터 채점), 해설 100문항 파일럿(남은 해설이 25개 미만이면 멈춤) |
| `q1` | 지금 LoRA 어댑터를 같은 방식으로 채점, `s0`과 그 합친 모델, `s1`, `s2`, 학습 문항 3,401개 해설 만들기(1,000개 미만이면 멈춤), `s8`(해설 모드 채점 포함), `sweep_report` |
| `q2` | `s3`, `s4`, `s5`, `sweep_report` |
| `final` | 통과한 체크포인트를 합치고 2021년 문제로 다시 잼: `PICK_CFG=<스윕 설정> PICK_CKPT=<체크포인트 폴더>`. 그다음 `CONFIG=<스윕 설정> SEED=<seed> bash scripts/gpu/stage.sh final_A`로 2022~2024년 문제를 한 번 잽니다 |

큐가 끝나면 지금 LoRA의 어댑터와 같은 방식끼리 비교하도록 정리를 한 번 더 돌립니다(GPU 필요 없음).

```bash
python -m quantization.sweep_report --reference-lora quantization/results/kormedmcqa_midm/unsloth_A/ckpt_dev/eval_vllm_lora_A_bf16_unsloth_seed42_run_checkpoint-426_selection_dev.json
```

이번 실행(2026-11-13)에서는 통과한 실험이 없어 `final`과 2022~2024년 측정을 돌리지 않았습니다.

## 확인

파일 위치는 `quantization/results/kormedmcqa_midm/sweep/` 기준입니다.

| 단계 | 남는 파일 | 볼 것 |
|---|---|---|
| `sweep_a` | `<실험>/A_train_seed<S>.json` | `recipe`가 그 설정과 같은지(학습률, epoch, 최대 길이), `global_step`. 보기를 섞거나 해설을 더한 실험은 `augment`의 `n_views`, `n_explain_items`, `gold_position_share` |
| `sweep_dev` | `<실험>/ckpt_dev/eval_vllm_lora_*_selection_dev.json` | 체크포인트마다 하나씩, `n_eval` 1,059, `n_unparsed` 0에 가까움 |
| `self_explain` | `s8_self_explain/self_explain_stats_seed42.json` | `n_kept`(해설이 남은 문항 수), `kept_by_role`. 해설 원문은 `quantization/artifacts/` 아래에 남고 커밋하지 않습니다 |
| `sweep_report` | `summary_dev.json` | 실험마다 `picked`와 `passes_gate`, `skipped_not_full_2021`이 비었는지, `dev_scorings_looked_at` |

## 안전장치와 멈추는 법

- **스윕 설정이 아니면 스윕 단계가 멈춥니다.** 발표용 모델과 결과를 덮어쓰지 않기 위해서입니다.
- **설정 이름의 seed와 `SEED`가 다르면 멈춥니다.** `s0_seed43.yaml`은 `SEED=43`으로만 돕니다.
- **학습 폴더가 이미 있으면 새로 학습하지 않습니다.** 이어 하려면 `RESUME=1`, 처음부터 하려면 폴더를 옮깁니다.
- **2022~2024년 문제는 판정 방식이 달라도 한 번만 잽니다.** 어댑터 채점(`--lora-dir`)은 2021년 문제에만 씁니다.
- **멈출 때는 프로세스 그룹 전체를 멈춥니다:** `STOP=<단계[_TAG]> bash scripts/gpu/stage.sh stop`(예: `STOP=sweep_a_s3_lr1e-4`). 큐와 그 아래 Python·vLLM 프로세스가 함께 멈춥니다. 결과는 `~/work/logs/stop.log`에 남습니다.

다음: [8. 안 될 때와 커밋할 것](08_troubleshooting.md)
