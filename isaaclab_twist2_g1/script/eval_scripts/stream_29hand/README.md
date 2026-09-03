# stream_29hand：Stream 29D body + 独立 Dex3 测试接口

## 测试内容

本目录基于 SONIC 的 29D body-only 评测接口，增加与身体动作时间对齐的独立 Dex3 双手输出。
29D 身体动作按 DFS/MuJoCo→SONIC 重排并执行；左右手各 7D 已反归一化的连续关节目标直接写入
Dex3，不经过 body permutation 或 body action scale。

本地 `serve_lerobot_vla_http.py` 设置 `LEROBOT_REQUIRE_HAND_ACTIONS=1`：checkpoint 缺少手部头或
HTTP 响应缺少手部 payload 时会立即失败，不会静默以张开手继续评测。parallel 入口同时提供
contract 校验、`10000-15000` 自动端口、端口锁、持久仿真和结果恢复。

## 支持的任务

本目录通过 `ENV_CONFIG_YAML` 统一映射 7 个任务：

| 场景 | YAML | 自动读取的仿真 task |
| --- | --- | --- |
| DoubleDesk | `doubledesk_sonic_test.yaml` | `Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody` |
| Football | `football_single_sonic_test.yaml` | `Isaac-Move-Football-Single-G129-Dex3-Wholebody` |
| PickPlace Box | `pp_box_sonic_test.yaml` | `Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby` |
| Boxing | `boxing_sonic_test.yaml` | `Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody` |
| Open Door | `open_door_sonic_test.yaml` | `Isaac-Move-Open-Door-G129-Dex3-Wholebody` |
| Sit Sofa | `sit_sofa_sonic_test.yaml` | `Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody` |
| Vision Navigation | `vision_navi_sonic_test.yaml` | `Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody` |

YAML 前缀是 `tasks/common_test_config/base_test/`。`TASK_NAME` 和结果文件的规范场景标签会自动
解析。

## 模型输入输出

`raw107_contract.py` 要求真正训练过独立 hand head 的 Stream checkpoint：

```text
policy/频率           Stream；50 Hz
输入图像              front RGB [3,224,224]
输入状态/历史          10 x 93
每帧状态              joint_pos29 + joint_vel29 + ang_vel_b3 + gravity3 + last_action29
body 输出             action.applied_action [25,29]；DFS/MuJoCo raw
hand 输出             action.left_hand [25,7] + action.right_hand [25,7]
```

不能通过手工修改旧 body-only `config.json` 来启用该入口；模型必须包含训练得到的独立手部头。

已验证的示例 checkpoint：

```text
/DATA/disk0/fym/vla/outputs/finetune_stream_humanoidarena_sonic_new_multisource_29d_hand_chunk_20260817_133613/checkpoints/035000/pretrained_model
```

## 运行方式

```bash
MODEL_PATH=/DATA/disk0/fym/vla/outputs/finetune_stream_humanoidarena_sonic_new_multisource_29d_hand_chunk_20260817_133613/checkpoints/035000/pretrained_model \
ENV_CONFIG_YAML=tasks/common_test_config/base_test/open_door_sonic_test.yaml \
GPU_ID=0 \
ISAAC_GPU_ID=1 \
SEEDS_OVERRIDE="0 1 2" \
REPEATS_PER_SEED=20 \
bash isaaclab_twist2_g1/script/eval_scripts/stream_29hand/run_vla_eval_parallel.sh
```

`GPU_ID` 是模型 server GPU，`ISAAC_GPU_ID` 是 Isaac/RTX/PhysX GPU。该入口强制
`NUM_WORKERS=1` 且一次只接受一个 server GPU；多 GPU 并行时启动多个进程并拆分 seed。模型和
仿真默认禁止共卡，只有显存确认足够时才设置 `ALLOW_SHARED_GPU=1`。

常用参数：

- `MODEL_PATH`，或批量选择器 `MODEL_PATHS_CSV`/`MODEL_PATHS_FILE`/`MODEL_ROOT`/`MODEL_GLOB`。
- `ENV_CONFIG_YAML` 映射任务；`TASK_NAME` 默认自动读取，手动覆盖时必须与 YAML 一致。
- `SEEDS_OVERRIDE`、`REPEATS_PER_SEED`；总 episode 为模型数 × seed 数 × repeat。
- `GPU_ID`、`ISAAC_GPU_ID`、`ALLOW_SHARED_GPU`、`GPU_LOCK`。
- `SERVER_PORT=auto`、`SERVER_PORT_BASE/MAX`、`MAX_STEPS`、`PERSISTENT_SIM`、`DRY_RUN=1`。
- `SERVER_DISABLE_ACTION_DELTA_REFINER=1`：仅用于明确记录的消融。
- `SERVER_ZERO_INFERENCE_NOISE=1`：将 Stream 推理的初始 action noise 设为全零；默认仍为标准高斯采样。
- `SONIC_OUTPUT_DELAY_STEPS` 默认必须保持 `0`；若未来增加延迟，body 和 hand 必须进入同一延迟束。

源码和 checkpoint 引用路径覆盖：

```bash
SERVER_PYTHON=/root/miniconda3/envs/lerobot_new/bin/python \
SERVER_LEROBOT_SRC=/DATA/disk0/fym/vla/src \
SERVER_CHECKPOINT_REF_REMAP="/share/beingm/yuxuan/model/paligemma_model=/DATA/disk0/fym/paligemma_model" \
MODEL_PATH=/path/to/pretrained_model GPU_ID=0 ISAAC_GPU_ID=1 \
bash isaaclab_twist2_g1/script/eval_scripts/stream_29hand/run_vla_eval_parallel.sh
```

## 公平性边界

- 不读取物体位姿、奖励、接触、目标状态或成功标签来修正 body/hand 输出。
- 不添加 IK、规划、抓取 latch、手部阈值规则或额外平滑。
- DFS/MuJoCo→SONIC 重排只作用于 29D body；手部输出不得经过 body scale/permutation。
- body 与 hand 必须来自同一个 policy step 并保持时间对齐；任何 delay/refiner 改动都要披露。
- 该接口比 `stream_29d` 多 14D 手部能力。操作任务的结果不能在未声明动作空间差异的情况下
  与 body-only 模型直接汇总。
- 固定 checkpoint、任务 YAML、`MAX_STEPS`、seed/repeat、随机化、成功判定和视频采样规则。
