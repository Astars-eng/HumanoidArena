# merged107：ACT / Diffusion Policy / Flow Matching 统一评测入口

本目录由当前工作区的 `act_new107` 复制并针对七任务混合训练 checkpoint 改造。三类策略共用同一套仿真、SONIC 控制、成功判定、录像和结果汇总逻辑；差异由 checkpoint 的 `config.json` 和保存的 processor 在 LeRobot HTTP server 内部处理。

## 支持的 checkpoint 合约

| 训练类型 | `config.json` | 当前 100000 checkpoint 时序 |
|---|---|---|
| ACT | `type=act` | `n_obs_steps=1`, `chunk_size=20`, `n_action_steps=20` |
| Diffusion Policy | `type=diffusion` | `n_obs_steps=1`, `horizon=24`, `n_action_steps=20` |
| Flow Matching | `type=multi_task_dit`, `objective=flow_matching` | `n_obs_steps=1`, `horizon=40`, `n_action_steps=20` |

共同接口为：

- 图像：`observation.images.front = [3, 224, 224]`；沿用训练转换时的直接拉伸。
- 状态：`observation.state = [93]`，即 `joint_pos[29] + joint_vel[29] + ang_vel_b[3] + gravity[3] + last_action[29]`。
- 动作：`action = [107]`，即 `applied_action[29] + motion_token[64] + left_hand[7] + right_hand[7]`。
- 关节序：状态和 checkpoint 的 `applied_action` 使用 DFS/MuJoCo 顺序；送入 SONIC 前转换为 SONIC/IsaacLab 顺序。

`raw107_contract.py` 在启动 server 前读取 checkpoint 自身配置并 fail-fast。它不把 ACT、DP 和 Flow Matching 的 horizon/chunk 硬改成同一个值。

## last_action 闭环

部署时必须使用 `last_action`。它不是当前关节目标，也不是当前模型输出，而是上一个控制 tick 实际执行的 29D decoder raw action：

- 每个 episode reset 为全零。
- `native_decoder`（默认）：记录 SONIC decoder 实际输出，再转换到 MuJoCo 顺序作为下一 tick 的 `last_action`。
- `direct_raw`（仅映射审计）：执行 checkpoint 的 `action.applied_action`，并把该 raw action 作为下一 tick 的 `last_action`。
- 每个控制 tick 都更新；策略内部什么时候重新规划，仍由各自保存的 action queue / horizon 决定。

推荐保持 `SONIC_RAW107_BODY_SOURCE=native_decoder`，因为它复现数据采集时带机器人状态反馈的 SONIC GMT decoder。`direct_raw` 只用于检查 29D 映射，不是默认性能评测方式。

## 七任务 prompt

环境 task id 继续用于创建仿真环境、随机化和成功判定；下面的训练原文通过独立的 `policy_task` 字段原样送给策略：

| wrapper | policy prompt |
|---|---|
| `HOI_double_desk_run_vla_eval_parallel.sh` | Put the hammer from the right table into the basket on the left table. |
| `HOI_football_run_vla_eval_parallel.sh` | Kick the football into the goal. |
| `HOI_pp_box_run_vla_eval_parallel.sh` | Move the box from the table onto the shelf. |
| `HSI_boxing_run_vla_eval_parallel.sh` | Strike the green markers on the punching bag. |
| `HSI_open_door_run_vla_eval_parallel.sh` | Open the door. |
| `HSI_sit_sofa_run_vla_eval_parallel.sh` | Sit on the sofa. |
| `HSI_vision_navi_run_vla_eval_parallel.sh` | Avoid obstacles and move to the yellow marked area. |

这对 Flow Matching 必须正确；ACT/DP 即使不消费文本，也可以安全共用入口。server 固定使用 verbatim task，避免把 `football` 改写成训练中没有的 `soccer`。

## 如何运行

先做无仿真的配置检查。默认会检查并列出三个 100000 checkpoint：

```bash
AUTO_ACTIVATE_CONDA=0 DRY_RUN=1 \
  bash isaaclab_twist2_g1/script/eval_scripts/merged107/HOI_double_desk_run_vla_eval_parallel.sh
```

单个模型、单 seed、单次 smoke test：

```bash
MODEL_PATH=/DATA/disk0/fym/vla/outputs/act_humanoidarena_sonic_merged_107_0820_fym/checkpoints/100000/pretrained_model \
SEEDS_OVERRIDE="0" REPEATS_PER_SEED=1 RECORD_VIDEO_EVERY_N=1 GPU_ID=0 \
  bash isaaclab_twist2_g1/script/eval_scripts/merged107/HOI_double_desk_run_vla_eval_parallel.sh
```

把 `MODEL_PATH` 分别换为以下路径即可用完全相同的入口评测 DP 和 Flow Matching：

```text
/DATA/disk0/fym/vla/outputs/dp_humanoidarena_sonic_merged_act107_0820_fym/checkpoints/100000/pretrained_model
/DATA/disk0/fym/vla/outputs/fm_humanoidarena_sonic_merged_act107_0820_fym/checkpoints/100000/pretrained_model
```

不设置 `MODEL_PATH` / `MODEL_PATHS_CSV` 时，一个 wrapper 会依次评测上述 ACT、DP、Flow Matching 三个默认 checkpoint。正式七任务评测分别运行七个 wrapper；每个 YAML 的默认 seeds、repeats 和 max steps 会自动加载。建议先 smoke test，再使用完整默认重复数，因为完整评测 episode 数较多。

`merged107` 默认关闭 GPU 文件锁，因此可以在同一张物理 GPU 上启动多个任务。此时各进程仍会自动选择不同的 HTTP server 端口，但显存和算力由这些进程共享；显存不足时请减少并发。若希望一张 GPU 同时只允许一个评测进程，设置 `GPU_LOCK=1`。

也可以显式指定一批模型：

```bash
MODEL_PATHS_CSV="/path/to/act/pretrained_model,/path/to/dp/pretrained_model,/path/to/fm/pretrained_model" \
GPU_ID=0 bash isaaclab_twist2_g1/script/eval_scripts/merged107/HSI_open_door_run_vla_eval_parallel.sh
```

## 评测判定与产物

评测保留 `act_new107` 的环境逻辑：每个控制步读取环境原始 reward，总 reward `>= 1.0` 判成功；达到任务 `MAX_STEPS` 判 `timeout`。默认跌倒检测在连续 5 个控制步满足硬倾斜，或“软倾斜 + 非脚部关键 body 接触力”时判 `fall`。其他异常会记录为 `sim_stopped`、`sim_error`、`interrupted`、`worker_error` 等，不计成功。

结果默认写入 `merged107/eval_results/<task_tag>_gpu.../`：

- `episodes/*.json`：每个 episode 的 seed、成功、失败原因、步数、reward、prompt、state/action 接口等。
- `summary.jsonl` / `summary.csv`：逐 episode 汇总。
- `summary.json`：总体、每模型、每 seed 成功率和失败原因统计。
- `final.csv`：模型成功率、seed 间标准差、成功平均步数、失败中的跌倒占比。
- `videos/success`、`videos/failure`、`logs`：分类录像和仿真/server 日志。

episode seed 使用 `sha256(task_name|group_seed|repeat_idx)` 确定，因此同一任务上三个模型共享同一组场景随机种子，便于公平横向比较。
