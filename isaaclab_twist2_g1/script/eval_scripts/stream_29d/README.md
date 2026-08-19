# stream_29d：Stream 29D body-only 测试接口

## 测试内容

本目录基于 SONIC 场景和评测流程，适配 Stream DFS/MuJoCo 29D 身体动作 checkpoint。它采集
本体状态、上一时刻动作和 front RGB，将模型输出从 DFS/MuJoCo 顺序重排到 SONIC 顺序，再按
`raw * action_scale + default_joint_pos` 执行。该接口没有手部输出，Dex3 保持 provider 默认目标。

`raw107_contract.py` 会在启动前校验 29D contract。parallel 入口会在 `10000-15000` 中自动选择
受跨进程锁保护的空闲端口，并支持持久仿真、断点恢复、视频和汇总。

## 支持的任务

本目录没有重复维护 7 个包装脚本，统一通过 `ENV_CONFIG_YAML` 映射任务：

| 场景 | YAML | 自动读取的仿真 task |
| --- | --- | --- |
| DoubleDesk | `doubledesk_sonic_test.yaml` | `Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody` |
| Football | `football_single_sonic_test.yaml` | `Isaac-Move-Football-Single-G129-Dex3-Wholebody` |
| PickPlace Box | `pp_box_sonic_test.yaml` | `Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby` |
| Boxing | `boxing_sonic_test.yaml` | `Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody` |
| Open Door | `open_door_sonic_test.yaml` | `Isaac-Move-Open-Door-G129-Dex3-Wholebody` |
| Sit Sofa | `sit_sofa_sonic_test.yaml` | `Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody` |
| Vision Navigation | `vision_navi_sonic_test.yaml` | `Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody` |

YAML 前缀统一为 `tasks/common_test_config/base_test/`。`TASK_NAME` 和结果场景标签会自动从 YAML
解析；通常不需要手写 task。

## 模型输入输出

```text
policy/频率           Stream；50 Hz
输入图像              front RGB [3,224,224]
输入状态/历史          10 x 93
每帧状态              joint_pos29 + joint_vel29 + ang_vel_b3 + gravity3 + last_action29
输出 chunk            [25,29]
输出                  action.applied_action；DFS/MuJoCo raw body action
手部                  无模型输出
```

已用本目录 contract 验证的示例 checkpoint：

```text
/DATA/disk0/fym/vla/outputs/finetune_stream_humanoidarena_sonic_new_multisource_29d_20260815_153724/checkpoints/070000/pretrained_model
```

## 运行方式

Stream server 和 Isaac RTX/PhysX 建议使用不同 GPU：

```bash
MODEL_PATH=/DATA/disk0/fym/vla/outputs/finetune_stream_humanoidarena_sonic_new_multisource_29d_20260815_153724/checkpoints/070000/pretrained_model \
ENV_CONFIG_YAML=tasks/common_test_config/base_test/doubledesk_sonic_test.yaml \
GPU_ID=0 \
ISAAC_GPU_ID=1 \
SEEDS_OVERRIDE="0 1 2" \
REPEATS_PER_SEED=20 \
bash isaaclab_twist2_g1/script/eval_scripts/stream_29d/run_vla_eval_parallel.sh
```

该入口要求一个 server GPU、`NUM_WORKERS=1`。多 GPU 并行时按 GPU 启动多个进程，给每个进程
不同的 `GPU_ID`/`ISAAC_GPU_ID` 和不重叠 seed。只有确认显存足够时才使用
`ALLOW_SHARED_GPU=1`。

常用参数：

- `MODEL_PATH`，或 `MODEL_PATHS_CSV`/`MODEL_PATHS_FILE`/`MODEL_ROOT`/`MODEL_GLOB`。
- `ENV_CONFIG_YAML`：映射 7 个任务；`TASK_NAME` 默认自动读取。显式覆盖二者时必须保持一致。
- `SEEDS_OVERRIDE="0 1 2"`、`REPEATS_PER_SEED=N`；`run_vla_eval.sh` 也接受 `EVAL_SEEDS`。
- `GPU_ID`：模型 server 的物理 GPU；`ISAAC_GPU_ID`：Isaac Sim 的物理 GPU。
- `SERVER_PORT=auto`、`SERVER_PORT_BASE/MAX`、`MAX_STEPS`、`PERSISTENT_SIM`、`DRY_RUN=1`。
- `SERVER_DISABLE_ACTION_DELTA_REFINER=1`：旁路 delta refiner，仅用于明确标注的消融。
- `SONIC_RAW_STATE_JOINT_ORDER=mujoco`：当前 29D contract 的默认关节顺序，不应随意覆盖。

训练 fork 或 checkpoint 固化引用路径不匹配时：

```bash
SERVER_PYTHON=/root/miniconda3/envs/lerobot_new/bin/python \
SERVER_LEROBOT_SRC=/DATA/disk0/fym/vla/src \
SERVER_CHECKPOINT_REF_REMAP="/share/beingm/yuxuan/model/paligemma_model=/DATA/disk0/fym/paligemma_model" \
MODEL_PATH=/path/to/pretrained_model GPU_ID=0 ISAAC_GPU_ID=1 \
bash isaaclab_twist2_g1/script/eval_scripts/stream_29d/run_vla_eval_parallel.sh
```

## 公平性边界

- 不读取物体位姿、奖励、接触、目标状态或成功标签来改变动作。
- 不添加 IK、轨迹规划、抓取规则、额外平滑或任务专用后处理。
- DFS/MuJoCo→SONIC 重排和 action scale 是必要机器人接口，必须对所有 29D 模型保持一致。
- `SERVER_DISABLE_ACTION_DELTA_REFINER`、历史长度、last-action 定义和插值行为必须随结果披露。
- 本接口没有手部动作能力；在依赖抓取/开门的任务上，不能与 29hand/107D 模型作为同等动作空间
  直接比较。
- 固定 checkpoint、任务 YAML、`MAX_STEPS`、seed/repeat、随机化和成功判定；多 GPU 拆分不能
  改变完整 seed/repeat 集合。

