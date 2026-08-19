# stream_new107：Stream motion-token 107D 测试接口

## 测试内容

本目录基于 SONIC raw107 流程适配 Stream motion-token checkpoint。适配层负责 64D 本体状态、
front RGB、Stream 在线 history/action queue、107D 动作拆分、SONIC decoder、Dex3 手部目标、
结果记录和失败恢复。parallel 入口会校验 contract，并自动在 `10000-15000` 中寻找空闲端口。

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
policy/频率           Stream；50 Hz
输入图像              front RGB [3,224,224]
输入状态/历史          10 x (joint_pos29 + joint_vel29 + ang_vel_b3 + gravity3) = [10,64]
输出 chunk            [25,107]
输出布局               applied_action29 + motion_token64 + left_hand7 + right_hand7
```

默认将 `motion_token` 交给 SONIC native decoder；手部 7+7D 连续目标直接控制 Dex3。前 29D
`applied_action` 的 `direct_raw` 路径仅用于映射审计。

已验证的示例 checkpoint：

```text
/DATA/disk0/fym/vla/outputs/stream_humanoidarena_lerobot/HOI_double_desk_float32_20260812_151035/checkpoints/200000/pretrained_model
```

## 运行方式

Stream float32 server 与 RTX/PhysX 通常不能安全共享一张 24 GiB GPU。`GPU_ID` 指定模型 server
GPU，`ISAAC_GPU_ID` 指定仿真 GPU：

```bash
MODEL_PATH=/DATA/disk0/fym/vla/outputs/stream_humanoidarena_lerobot/HOI_double_desk_float32_20260812_151035/checkpoints/200000/pretrained_model \
GPU_ID=0 \
ISAAC_GPU_ID=1 \
SEEDS_OVERRIDE="0 1 2" \
REPEATS_PER_SEED=20 \
bash isaaclab_twist2_g1/script/eval_scripts/stream_new107/HOI_double_desk_run_vla_eval_parallel.sh
```

该入口要求每个进程一个 server GPU、`NUM_WORKERS=1`。多 GPU 并行时启动多个进程，为每个进程
分配不同的 `GPU_ID`/`ISAAC_GPU_ID` 和不重叠 seed。仅在显存已经确认足够时才能显式设置
`ALLOW_SHARED_GPU=1`。

常用参数：

- `MODEL_PATH`，或批量选择器 `MODEL_PATHS_CSV`/`MODEL_PATHS_FILE`/`MODEL_ROOT`/`MODEL_GLOB`。
- `SEEDS_OVERRIDE`、`REPEATS_PER_SEED`；单入口别名为 `EVAL_SEEDS`。
- `ENV_CONFIG_YAML` 映射场景，`TASK_NAME` 默认从 YAML 读取；手动覆盖必须保持一致。
- `GPU_ID`、`ISAAC_GPU_ID`、`ALLOW_SHARED_GPU`、`GPU_LOCK`。
- `SERVER_PORT=auto`、`SERVER_PORT_BASE/MAX`、`MAX_STEPS`、`DRY_RUN=1`。

源码或 checkpoint 内引用路径有问题时：

```bash
SERVER_PYTHON=/root/miniconda3/envs/lerobot_new/bin/python \
SERVER_LEROBOT_SRC=/DATA/disk0/fym/vla/src \
SERVER_CHECKPOINT_REF_REMAP="/share/old/model=/DATA/disk0/fym/model" \
MODEL_PATH=/path/to/pretrained_model GPU_ID=0 ISAAC_GPU_ID=1 \
bash isaaclab_twist2_g1/script/eval_scripts/stream_new107/HOI_double_desk_run_vla_eval_parallel.sh
```

## 公平性边界

- 不读取物体位姿、奖励、接触、目标状态或成功标签来修正动作。
- 不添加 IK、规划、抓取 latch、动作平滑或任务规则。
- native decoder 是必须披露的机器人接口；`direct_raw` 是消融，不能混合统计。
- Stream 的在线 10 帧 history、25 步 queue、图像拉伸和 action-delta 行为属于模型接口。
- 跨模型比较必须固定任务/YAML、checkpoint 筛选、`MAX_STEPS`、seed/repeat、视觉随机化和成功
  判定。GPU 拆分不得改变 seed 集合或为单个模型提供额外重复次数。

