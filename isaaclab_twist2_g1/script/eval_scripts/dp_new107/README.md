# dp_new107：Diffusion motion-token 107D 测试接口

## 测试内容

本目录基于 SONIC raw107 评测接口适配 Diffusion Policy checkpoint。它保留统一的场景、机器人、
HTTP server、视频、成功判定和结果汇总，只替换模型契约与时序校验。并行入口会在仿真前运行
`dp107_contract.py`，并在 `10000-15000` 范围内自动选择受文件锁保护的空闲端口。

## 支持的任务

| 场景 | 任务入口 | YAML | 仿真 task |
| --- | --- | --- | --- |
| DoubleDesk | `HOI_double_desk_run_vla_eval_parallel.sh` | `doubledesk_sonic_test.yaml` | `Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody` |
| Football | `HOI_football_run_vla_eval_parallel.sh` | `football_single_sonic_test.yaml` | `Isaac-Move-Football-Single-G129-Dex3-Wholebody` |
| PickPlace Box | `HOI_pp_box_run_vla_eval_parallel.sh` | `pp_box_sonic_test.yaml` | `Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby` |
| Boxing | `HSI_boxing_run_vla_eval_parallel.sh` | `boxing_sonic_test.yaml` | `Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody` |
| Open Door | `HSI_open_door_run_vla_eval_parallel.sh` | `open_door_sonic_test.yaml` | `Isaac-Move-Open-Door-G129-Dex3-Wholebody` |
| Sit Sofa | `HSI_sit_sofa_run_vla_eval_parallel.sh` | `sit_sofa_sonic_test.yaml` | `Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody` |
| Vision Navigation | `HSI_vision_navi_run_vla_eval_parallel.sh` | `vision_navi_sonic_test.yaml` | `Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody` |

## 模型输入输出

```text
policy/horizon       diffusion；horizon=32，n_action_steps=25，drop_n_last_frames=6
输入图像              front RGB [3,224,224]
输入状态/历史          2 x (joint_pos29 + joint_vel29 + ang_vel_b3 + gravity3) = [2,64]
输出                  [25,107]
输出布局               applied_action29 + motion_token64 + left_hand7 + right_hand7
```

64D `motion_token` 默认进入 SONIC native decoder；双手 7+7D 连续输出直接进入 Dex3。

已验证的示例 checkpoint：

```text
/DATA/disk0/fym/vla/outputs/diffusion_humanoidarena_sonic_20260812_145643/HOI_double_desk/checkpoints/200000/pretrained_model
```

## 运行方式

```bash
MODEL_PATH=/DATA/disk0/fym/vla/outputs/diffusion_humanoidarena_sonic_20260812_145643/HOI_double_desk/checkpoints/200000/pretrained_model \
GPU_ID=0 \
SEEDS_OVERRIDE="0 1 2" \
REPEATS_PER_SEED=20 \
bash isaaclab_twist2_g1/script/eval_scripts/dp_new107/HOI_double_desk_run_vla_eval_parallel.sh
```

该入口要求单 GPU、`NUM_WORKERS=1`。多 GPU 时按 GPU 拆分 seed 并启动多个独立进程：

```bash
GPU_ID=0 SEEDS_OVERRIDE="0 1" MODEL_PATH=/path/to/pretrained_model \
  bash isaaclab_twist2_g1/script/eval_scripts/dp_new107/HOI_double_desk_run_vla_eval_parallel.sh &
GPU_ID=1 SEEDS_OVERRIDE="2 3" MODEL_PATH=/path/to/pretrained_model \
  bash isaaclab_twist2_g1/script/eval_scripts/dp_new107/HOI_double_desk_run_vla_eval_parallel.sh &
wait
```

常用参数包括 `MODEL_PATH`/`MODEL_PATHS_CSV`/`MODEL_PATHS_FILE`/`MODEL_ROOT`、
`SEEDS_OVERRIDE`、`REPEATS_PER_SEED`、`ENV_CONFIG_YAML`、`TASK_NAME`、`MAX_STEPS`、
`SERVER_PORT=auto`、`SERVER_PORT_BASE/MAX`、`DRY_RUN=1`。`TASK_NAME` 默认由 YAML 自动读取；
手动映射时二者必须指向同一场景。

源码覆盖示例：

```bash
SERVER_PYTHON=/root/miniconda3/envs/lerobot_new/bin/python \
SERVER_LEROBOT_SRC=/DATA/disk0/fym/vla/src \
SERVER_CHECKPOINT_REF_REMAP="/old/src=/DATA/disk0/fym/vla/src" \
MODEL_PATH=/path/to/pretrained_model \
bash isaaclab_twist2_g1/script/eval_scripts/dp_new107/HOI_double_desk_run_vla_eval_parallel.sh
```

## 公平性边界

- 不使用物体位姿、奖励、接触、成功状态或其他 oracle 信号修正动作。
- 不增加 IK、规划、抓取规则、额外滤波或按任务后处理。
- native decoder 与 `direct_raw` 是不同执行边界，必须单独披露和统计。
- Diffusion 的 2 帧历史、32 步 horizon、25 步执行队列是模型契约；与 10 帧 ACT/FM/Stream
  比较时不能把差异归因于策略架构本身。
- 固定任务 YAML、checkpoint 选择、图像处理、`MAX_STEPS`、seed/repeat、随机化和成功判定。

