# act_new107：ACT motion-token 107D 测试接口

## 测试内容

本目录基于 `script/eval_scripts/sonic` 的仿真、场景和成功判定逻辑，为 ACT motion-token
checkpoint 提供独立评测适配。适配层负责 checkpoint 契约校验、LeRobot HTTP server、SONIC
body decoder、Dex3 手部控制、任务调度、视频与结果汇总；不负责训练。

推荐使用 `*_run_vla_eval_parallel.sh` 或 `run_vla_eval_parallel.sh`。并行入口会在启动仿真前校验
checkpoint，并在 `10000-15000` 范围内自动寻找空闲 server 端口；端口选择受跨进程文件锁保护。

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

YAML 均位于 `tasks/common_test_config/base_test/`。包装脚本会选择 YAML 和默认 `MAX_STEPS`，
`TASK_NAME` 默认从 YAML 的 `task_name` 自动读取。

## 模型输入输出

`raw107_contract.py` 要求 ACT checkpoint 满足：

```text
控制频率/图像       50 Hz；front RGB [3,224,224]
状态历史            10 帧
输入状态            joint_pos(29) + joint_vel(29) + ang_vel_b(3) + gravity(3) = 64
输出 chunk          25 x 107
输出布局             applied_action(29) + motion_token(64) + left_hand(7) + right_hand(7)
```

默认 `SONIC_RAW107_BODY_SOURCE=native_decoder`，即用 64D `motion_token` 和实时本体反馈产生
29D body target；左右手 7+7D 连续目标直接写入 Dex3。`direct_raw` 只用于动作映射消融。

`SONIC_RAW107_HAND_MODE=policy`（默认）执行 checkpoint 的左右手 7+7D 输出；设为 `open` 时仍做
完整 107D 推理并保持 body/motion-token 路径不变，但执行端丢弃最后 14D，用 Dex3 标准 open pose
覆盖双手。该选项用于手部输出消融，不应在同一结果目录中混用两种模式。

`ACT_NEW107_EXECUTE_STEPS` 控制每个 25 步预测 chunk 实际执行的前缀长度，范围为 1–25，默认 25。
例如设为 5 时，server 的 ACT action queue 每 5 个控制周期耗尽并重新预测，同时每个控制周期仍会
更新模型的 10 帧状态历史。`SONIC_RAW107_SMOOTH_ALPHA` 对 `direct_raw` 转换后的 29D 关节目标应用
EMA：`smoothed = alpha * current + (1-alpha) * previous`；默认 1.0（关闭），推荐先以 0.2 做稳定性测试。

已用本目录 contract 验证的示例 checkpoint：

```text
/DATA/disk0/fym/vla/outputs/act_humanoidarena_sonic_20260812_145530/HOI_double_desk/checkpoints/200000/pretrained_model
```

## 运行方式

从 HumanoidArena 仓库根目录运行：

```bash
MODEL_PATH=/DATA/disk0/fym/vla/outputs/act_humanoidarena_sonic_20260812_145530/HOI_double_desk/checkpoints/200000/pretrained_model \
GPU_ID=0 \
SEEDS_OVERRIDE="0 1 2" \
REPEATS_PER_SEED=20 \
bash isaaclab_twist2_g1/script/eval_scripts/act_new107/HOI_double_desk_run_vla_eval_parallel.sh
```

该目录为单 GPU 安全模式：`NUM_WORKERS` 必须为 `1`，且一次运行只能指定一个 `GPU_ID`。
需要多 GPU 并行时，在不同 GPU 上启动独立进程，并拆分 seed：

```bash
GPU_ID=0 SEEDS_OVERRIDE="0 1" MODEL_PATH=/path/to/pretrained_model \
  bash isaaclab_twist2_g1/script/eval_scripts/act_new107/HOI_double_desk_run_vla_eval_parallel.sh &
GPU_ID=1 SEEDS_OVERRIDE="2 3" MODEL_PATH=/path/to/pretrained_model \
  bash isaaclab_twist2_g1/script/eval_scripts/act_new107/HOI_double_desk_run_vla_eval_parallel.sh &
wait
```

常用参数：

- `MODEL_PATH`：单个 checkpoint；建议直接指向 `pretrained_model`。
- `MODEL_PATHS_CSV`、`MODEL_PATHS_FILE`、`MODEL_ROOT`、`MODEL_GLOB`：批量选择 checkpoint。
- `SEEDS_OVERRIDE="0 1 2"`：并行入口的 seed 列表；`run_vla_eval.sh` 也接受 `EVAL_SEEDS` 别名。
- `REPEATS_PER_SEED=N`：每个 seed 的重复次数；总 episode 数为模型数 × seed 数 × N。
- `ENV_CONFIG_YAML=...`：映射场景；`TASK_NAME` 通常由 YAML 自动读取。手动覆盖时必须保证二者一致。
- `SERVER_PORT=auto`：默认自动选端口；可用 `SERVER_PORT_BASE/MAX` 调整范围。
- `DRY_RUN=1`：只做模型发现、contract 和参数预检，不启动评测。
- `SONIC_RAW107_HAND_MODE=policy|open`：执行模型手部输出，或固定双手为标准张开姿态。
- `ACT_NEW107_EXECUTE_STEPS=1..25`：每次预测 25 步，但只执行指定前缀后重新预测。
- `SONIC_RAW107_SMOOTH_ALPHA=(0,1]`：`direct_raw` 关节目标 EMA 系数，数值越小越平滑、滞后越大。

若 server 导入了错误的 LeRobot 源码，显式覆盖训练 fork：

```bash
SERVER_PYTHON=/root/miniconda3/envs/lerobot_new/bin/python \
SERVER_LEROBOT_SRC=/DATA/disk0/fym/vla/src \
SERVER_CHECKPOINT_REF_REMAP="/checkpoint/old/src=/DATA/disk0/fym/vla/src" \
MODEL_PATH=/path/to/pretrained_model \
bash isaaclab_twist2_g1/script/eval_scripts/act_new107/HOI_double_desk_run_vla_eval_parallel.sh
```

`SERVER_LEROBOT_SRC` 覆盖 server 的 Python 源码；`SERVER_CHECKPOINT_REF_REMAP=旧前缀=新前缀`
用于 checkpoint 中固化的本地引用路径。两者作用不同。

## 公平性边界

- 适配器不读取物体位姿、奖励、接触、成功状态或其他特权任务状态来修正动作。
- 不加入 IK、轨迹规划、抓取 latch、任务规则、额外平滑或按任务调参。
- SONIC native decoder 是机器人接口的一部分，正式结果必须披露 `SONIC_RAW107_BODY_SOURCE`；
  `direct_raw` 结果不能与 native-decoder 结果混报。
- ACT 的 10 帧历史和 25 步原生 action queue 属于该接口时序。跨策略比较时必须同时报告输入历史、
  action chunk/执行长度、图像预处理、`MAX_STEPS`、seed、重复次数和视觉随机化配置。
- checkpoint、YAML/task 映射和成功判定必须匹配；不能根据评测结果为单一模型调整场景或控制参数。
