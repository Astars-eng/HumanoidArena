# act_refpose_new：ACT RefPose 40D 测试接口

## 测试内容

本目录是基于 SONIC 的独立 ACT RefPose v3.1 40D 评测适配，不修改 `act_refpose` 或共享 SONIC
provider。它维护 10 帧本体历史，调用专用 HTTP server，执行 semantic RefPose body action，
并记录完整预测 chunk、实际执行前缀和重规划边界指标。

parallel 入口支持自动端口：每个 worker 从 `SERVER_PORT_BASE` 开始，在 `SERVER_PORT_MAX` 以内
寻找空闲端口，并通过文件锁与 `/health` 的 instance token、模型路径和 shape 校验，避免误连旧
server。

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

YAML 位于 `tasks/common_test_config/base_test/`；包装脚本选择 YAML 和 `MAX_STEPS`，`TASK_NAME`
默认从 YAML 自动读取。

## 模型输入输出

专用 server 在加载时校验：

```text
policy/频率           ACT；50 Hz
输入图像              front RGB [3,224,224]
输入状态/历史          10 x (joint_pos29 + joint_vel29 + ang_vel_b3 + gravity3) = [10,64]
输出 chunk            [25,40]
输出布局               root_local_xy_delta2 + root_z1 + root_rot6d6 + joint_pos29
                      + hand_binary2
rotation layout       row
```

默认预测 25 步并执行前 `ACT_REFPOSE_EXECUTE_STEPS=5` 步，然后使用最新图像与状态历史重规划。
episode 起点不足的历史用最早可用状态补齐。

已验证配置为上述 40D 契约的示例 checkpoint：

```text
/DATA/disk0/fym/vla/outputs/act_refpose_humanoidarena_lerobot_fym_0807_complete/HOI_double_desk/checkpoints/200000/pretrained_model
```

## 运行方式

```bash
AUTO_ACTIVATE_CONDA=0 \
EVAL_PYTHON=/root/miniconda3/envs/unitree_sim_env/bin/python \
SERVER_PYTHON=/root/miniconda3/envs/lerobot/bin/python \
LEROBOT_VLA_SRC=/DATA/disk0/fym/vla/src \
MODEL_PATHS_CSV=/DATA/disk0/fym/vla/outputs/act_refpose_humanoidarena_lerobot_fym_0807_complete/HOI_double_desk/checkpoints/200000/pretrained_model \
SERVER_GPU_IDS=0,1 \
NUM_WORKERS=1 \
SEEDS_OVERRIDE="0 1 2" \
REPEATS_PER_SEED=20 \
bash isaaclab_twist2_g1/script/eval_scripts/act_refpose_new/HOI_double_desk_run_vla_eval_parallel.sh
```

`SERVER_GPU_IDS` 指定物理 GPU，`NUM_WORKERS` 表示每张 GPU 的 worker 数；示例总并发为 2。
`ISAAC_DEVICE=cuda` 时仿真跟随 worker 的 server GPU。

常用参数：

- parallel 入口使用 `MODEL_PATHS_CSV`（单个或逗号分隔的多个模型），也支持
  `MODEL_PATHS_FILE`、`MODEL_ROOT`、`MODEL_GLOB`；`run_vla_eval.sh` 单入口直接使用
  `MODEL_PATH`。
- `SEEDS_OVERRIDE`、`REPEATS_PER_SEED`；总 episode 数为模型数 × seed 数 × repeat。
- `ENV_CONFIG_YAML` 映射场景；`TASK_NAME` 默认自动读取，手动覆盖时必须与 YAML 一致。
- `ACT_REFPOSE_HISTORY_STEPS=10`：固定 contract。
- `ACT_REFPOSE_EXECUTE_STEPS=5`：允许 1-25 的显式时序消融。
- `ACT_REFPOSE_RECORD_FULL_CHUNKS=1`：保留完整 chunk trace。
- `SERVER_PORT_BASE/MAX`、`MAX_STEPS`、`PERSISTENT_SIM`、`RESUME_LATEST`、`DRY_RUN=1`。

遇到训练源码不匹配、缺少扩展 config 字段或导入到 pip 版 LeRobot 时，覆盖源码根目录：

```bash
LEROBOT_VLA_SRC=/path/to/training/repository/src \
SERVER_PYTHON=/path/to/lerobot-env/bin/python \
MODEL_PATHS_CSV=/path/to/pretrained_model \
bash isaaclab_twist2_g1/script/eval_scripts/act_refpose_new/HOI_double_desk_run_vla_eval_parallel.sh
```

## 公平性边界

- 不读取物体位姿、奖励、接触、目标状态或成功标签来修正动作。
- 不添加任务控制器、IK、规划、抓取规则或额外平滑。
- semantic RefPose runtime、row rotation layout、历史长度和执行前缀属于接口；必须与结果一起披露。
- 40D 的 `hand_binary2` 与 52D 的连续 Dex3 14D、raw107 的连续手部头不是同等动作空间，操作任务
  不能不加说明地横向汇总。
- 对比时固定 checkpoint 选择、YAML/task、图像几何、`ACT_REFPOSE_EXECUTE_STEPS`、
  `MAX_STEPS`、seed/repeat、视觉随机化和成功判定。
