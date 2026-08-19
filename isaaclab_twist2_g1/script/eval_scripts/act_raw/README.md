# act_raw：ACT encoder-token 107D 测试接口

## 测试内容

本目录基于 SONIC 评测流程适配较早的 ACT raw107 checkpoint。它与 `act_new107` 的主要区别是
64D 中间动作字段名为 `action.encoder_token`，不能把两类 checkpoint 混用。目录负责 contract
校验、HTTP policy server、SONIC body 执行、Dex3 手部输出、并行调度和结果记录。

推荐使用任务包装脚本。并行调度器会从 `SERVER_PORT_BASE` 到 `SERVER_PORT_MAX` 自动寻找空闲
端口，并以全局文件锁避免多个 worker 抢占同一端口。

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

YAML 位于 `tasks/common_test_config/base_test/`，包装脚本负责选择 YAML 和 `MAX_STEPS`。

## 模型输入输出

`raw107_contract.py` 要求：

```text
policy               ACT（50 Hz）
输入图像              front RGB [3,224,224]
输入状态/历史          10 x (joint_pos29 + joint_vel29 + ang_vel_b3 + gravity3) = [10,64]
输出 chunk            [25,107]
输出布局               applied_action29 + encoder_token64 + left_hand7 + right_hand7
```

默认用 `encoder_token` 驱动 SONIC native decoder，双手连续 7+7D 直接控制 Dex3；
`SONIC_RAW107_BODY_SOURCE=direct_raw` 仅作为 29D raw head 的映射消融。

已验证的示例 checkpoint：

```text
/DATA/disk0/fym/vla/outputs/act_pp_box_raw_200k_0806_fym/checkpoints/200000/pretrained_model
```

## 运行方式

```bash
MODEL_PATHS_CSV=/DATA/disk0/fym/vla/outputs/act_pp_box_raw_200k_0806_fym/checkpoints/200000/pretrained_model \
SERVER_GPU_IDS=0,1 \
NUM_WORKERS=1 \
SEEDS_OVERRIDE="0 1 2" \
REPEATS_PER_SEED=20 \
bash isaaclab_twist2_g1/script/eval_scripts/act_raw/HOI_pp_box_run_vla_eval_parallel.sh
```

`SERVER_GPU_IDS` 是物理 GPU 列表，`NUM_WORKERS` 是每张 GPU 的 worker 数，因此总并发槽位为
GPU 数 × `NUM_WORKERS`。默认 `ISAAC_DEVICE=cuda` 时，每个 Isaac worker 跟随对应 server GPU。
显存不足时先把 `NUM_WORKERS` 降为 `1` 或减少 GPU 上的其他任务。

常用参数：

- parallel 入口用 `MODEL_PATHS_CSV` 指定一个或多个逗号分隔模型；也支持
  `MODEL_PATHS_FILE`、`MODEL_ROOT`、`MODEL_GLOB`。`run_vla_eval.sh` 单入口才直接使用
  `MODEL_PATH`。路径建议指向 `pretrained_model`。
- `SEEDS_OVERRIDE="0 1 2"`、`REPEATS_PER_SEED=20`：seed 与每 seed 重复次数。
- `ENV_CONFIG_YAML=...`：任务 YAML；`TASK_NAME` 默认从其中读取，可显式覆盖但必须保持一致。
- `SERVER_PORT_BASE=10000 SERVER_PORT_MAX=15000`：自动端口搜索范围。
- `MAX_STEPS`、`RECORD_VIDEO_EVERY_N`、`PERSISTENT_SIM`、`RESUME_LATEST`：评测运行控制。
- `DRY_RUN=1`：只预检，不启动 server 和仿真。

源码或 checkpoint 引用路径不匹配时：

```bash
SERVER_PYTHON=/root/miniconda3/envs/lerobot/bin/python \
SERVER_LEROBOT_SRC=/DATA/disk0/fym/vla/src \
SERVER_CHECKPOINT_REF_REMAP="/old/training/src=/DATA/disk0/fym/vla/src" \
MODEL_PATHS_CSV=/path/to/pretrained_model \
bash isaaclab_twist2_g1/script/eval_scripts/act_raw/HOI_pp_box_run_vla_eval_parallel.sh
```

## 公平性边界

- 不读取物体位姿、奖励、接触或成功状态，不使用任务 oracle 修正动作。
- 不添加 IK、轨迹规划、抓取规则、额外滤波或任务专用动作后处理。
- 必须披露 `native_decoder` 或 `direct_raw`；二者不是同一执行接口。
- ACT history、25 步 queue、图像拉伸方式和 `encoder_token` 语义属于 checkpoint 契约。
- 比较模型时固定 checkpoint 选择规则、YAML/task、`MAX_STEPS`、seed、重复次数、成功判定和
  随机化；不能将本接口与 `motion_token` 接口误认为只差模型架构。
