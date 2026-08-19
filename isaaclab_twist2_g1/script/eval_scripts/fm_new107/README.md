# fm_new107：Flow Matching motion-token 107D 测试接口

## 测试内容

本目录基于 SONIC raw107 流程适配 Flow Matching checkpoint，复用相同场景、机器人接口、HTTP
server、成功判定和结果格式。`fm107_contract.py` 在 server 启动前校验模型，parallel 入口在
`10000-15000` 范围内自动选择空闲端口并以文件锁避免冲突。

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
policy                flow_matching
历史/时序              n_obs_steps=10，chunk_size=25，n_action_steps=25
积分                  midpoint，num_inference_steps=10，sigma_min=0
输入图像              front RGB [3,224,224]
输入状态              [10,64]；每帧为 joint_pos29 + joint_vel29 + ang_vel_b3 + gravity3
输出                  [25,107]
输出布局               applied_action29 + motion_token64 + left_hand7 + right_hand7
```

默认以 `motion_token` 驱动 SONIC native decoder，并把连续 7+7D 手部目标写入 Dex3。

已验证的示例 checkpoint：

```text
/DATA/disk0/fym/vla/outputs/flow_matching_humanoidarena_sonic_20260812_145757/HOI_double_desk/checkpoints/200000/pretrained_model
```

## 运行方式

```bash
MODEL_PATH=/DATA/disk0/fym/vla/outputs/flow_matching_humanoidarena_sonic_20260812_145757/HOI_double_desk/checkpoints/200000/pretrained_model \
GPU_ID=0 \
SEEDS_OVERRIDE="0 1 2" \
REPEATS_PER_SEED=20 \
bash isaaclab_twist2_g1/script/eval_scripts/fm_new107/HOI_double_desk_run_vla_eval_parallel.sh
```

该入口要求单 GPU、`NUM_WORKERS=1`。多 GPU 并行应拆分 seed 后分别启动进程；`GPU_ID` 用环境
变量指定最通用，部分包装脚本也接受首个位置参数作为 GPU 编号。

模型选择优先使用显式 `MODEL_PATH`；也支持 `MODEL_PATHS_CSV`、`MODEL_PATHS_FILE`、
`MODEL_ROOT` 和 `MODEL_GLOB`。常用运行参数为 `SEEDS_OVERRIDE`、`REPEATS_PER_SEED`、
`ENV_CONFIG_YAML`、`TASK_NAME`、`MAX_STEPS`、`SERVER_PORT=auto`、`SERVER_PORT_BASE/MAX`、
`PERSISTENT_SIM`、`RESUME_LATEST` 和 `DRY_RUN=1`。`TASK_NAME` 默认从 YAML 自动加载。

源码覆盖与 checkpoint 路径重映射：

```bash
SERVER_PYTHON=/root/miniconda3/envs/lerobot_new/bin/python \
SERVER_LEROBOT_SRC=/DATA/disk0/fym/vla/src \
SERVER_CHECKPOINT_REF_REMAP="/old/training/src=/DATA/disk0/fym/vla/src" \
MODEL_PATH=/path/to/pretrained_model \
bash isaaclab_twist2_g1/script/eval_scripts/fm_new107/HOI_double_desk_run_vla_eval_parallel.sh
```

## 公平性边界

- 不读取物体状态、奖励、接触或成功标签来修正 policy 输出。
- 不增加任务控制器、IK、规划、抓取规则或额外平滑。
- native decoder、`direct_raw`、积分步数和积分方法均属于需要披露的接口条件。
- Flow Matching 的 10 帧历史与 25 步 queue 必须按 contract 使用；不要只为某个模型修改
  `num_inference_steps`、时序或图像几何后与其他模型直接比较。
- 固定 checkpoint 筛选、任务 YAML、`MAX_STEPS`、seed/repeat、随机化和成功判定。

