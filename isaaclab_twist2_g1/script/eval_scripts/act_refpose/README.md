# act_refpose：ACT RefPose 52D 测试接口

## 测试内容

本目录基于 SONIC 仿真评测流程，为 ACT RefPose v3.1 52D checkpoint 提供适配。它采集 10 帧
64D 本体历史和 front RGB，将 52D chunk 的身体部分送入 semantic RefPose runtime，并把左右
Dex3 连续关节目标与动作步严格对齐。目录还负责 checkpoint/health 校验、HTTP server、并行调度、
视频和结果汇总。

推荐使用 parallel 入口。每个 worker 会在 `SERVER_PORT_BASE` 到 `SERVER_PORT_MAX` 中自动寻找
空闲端口，端口选择和 server readiness 位于跨进程锁内，避免连接到其他评测留下的 server。

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

所有 YAML 位于 `tasks/common_test_config/base_test/`。任务包装脚本设置 YAML 和当前默认
`MAX_STEPS`，`TASK_NAME` 从 YAML 自动读取。

## 模型输入输出

`refpose52_contract.py` 和 server `/health` 双重校验：

```text
policy/频率           ACT；50 Hz
输入图像              front RGB [3,224,224]
输入状态/历史          10 x (joint_pos29 + joint_vel29 + ang_vel_b3 + gravity3) = [10,64]
输出 chunk            [25,52]
输出布局               root_local_xy_delta2 + root_z1 + root_rot6d6 + joint_pos29
                      + left_hand7 + right_hand7
```

默认预测 25 步、执行前 `ACT_REFPOSE_EXECUTE_STEPS=5` 步后重新规划；身体前 38D 进入 semantic
RefPose runtime，手部 7+7D 连续目标直接控制 Dex3。

已验证的示例 checkpoint：

```text
/DATA/disk0/fym/vla/outputs/act_refpose_humanoidarena_lerobot/HOI_double_desk_20260812_161940/checkpoints/200000/pretrained_model
```

## 运行方式

建议显式区分 Isaac 和 LeRobot Python：

```bash
AUTO_ACTIVATE_CONDA=0 \
EVAL_PYTHON=/root/miniconda3/envs/unitree_sim_env/bin/python \
SERVER_PYTHON=/root/miniconda3/envs/lerobot/bin/python \
LEROBOT_VLA_SRC=/DATA/disk0/fym/vla/src \
MODEL_PATH=/DATA/disk0/fym/vla/outputs/act_refpose_humanoidarena_lerobot/HOI_double_desk_20260812_161940/checkpoints/200000/pretrained_model \
SERVER_GPU_IDS=0,1 \
NUM_WORKERS=1 \
SEEDS_OVERRIDE="0 1 2" \
REPEATS_PER_SEED=20 \
bash isaaclab_twist2_g1/script/eval_scripts/act_refpose/HOI_double_desk_run_vla_eval_parallel.sh
```

`SERVER_GPU_IDS` 是物理 GPU 列表，`NUM_WORKERS` 是每张 GPU 的 worker 数；以上配置提供两个
并发槽位。`ISAAC_DEVICE=cuda` 时，每个仿真跟随对应 server GPU。若显存不足，减少 GPU 上其他
进程或保持 `NUM_WORKERS=1`。

常用参数：

- `MODEL_PATH`；批量时使用 `MODEL_PATHS_CSV`、`MODEL_PATHS_FILE`、`MODEL_ROOT`、`MODEL_GLOB`。
- `SEEDS_OVERRIDE="0 1 2"`、`REPEATS_PER_SEED=N`；单入口也接受 `EVAL_SEEDS`。
- `ENV_CONFIG_YAML=...` 映射场景；`TASK_NAME` 默认从 YAML 读取。手动覆盖时二者必须一致。
- `ACT_REFPOSE_HISTORY_STEPS=10`：contract 固定值。
- `ACT_REFPOSE_EXECUTE_STEPS=5`：可在 1 到 25 间做明确披露的时序消融。
- `SERVER_PORT_BASE=10000 SERVER_PORT_MAX=15000`：自动端口范围。
- `MAX_STEPS`、`PERSISTENT_SIM`、`RECORD_VIDEO_EVERY_N`、`RESUME_LATEST`、`DRY_RUN=1`。

如果 server 找到了错误的 `lerobot`/训练 fork，用 `LEROBOT_VLA_SRC` 覆盖，不要只修改当前
shell 的普通 `PYTHONPATH`：

```bash
LEROBOT_VLA_SRC=/path/to/training/repository/src \
SERVER_PYTHON=/path/to/lerobot-env/bin/python \
MODEL_PATH=/path/to/pretrained_model \
bash isaaclab_twist2_g1/script/eval_scripts/act_refpose/HOI_double_desk_run_vla_eval_parallel.sh
```

## 公平性边界

- 不读取物体位姿、奖励、接触、目标状态或成功标签来改变动作。
- 不增加 IK、规划、抓取 latch、任务规则或额外动作平滑。
- semantic RefPose runtime、root 旋转布局、10 帧历史和“预测 25/执行 5”均属于需要披露的
  机器人接口与时序，不能与 raw107/native-decoder 结果视为只差模型。
- `ACT_REFPOSE_EXECUTE_STEPS`、图像处理、`MAX_STEPS`、YAML、seed/repeat、随机化和成功判定
  必须在对比模型之间保持一致。
- 52D 连续手部输出与 40D 二值手部接口能力不同，涉及操作任务时必须分组报告。

