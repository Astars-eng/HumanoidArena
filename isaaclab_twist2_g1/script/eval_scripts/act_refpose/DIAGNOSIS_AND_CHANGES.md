# ACT Ref-pose DoubleDesk 评测：问题诊断与修改说明

## 1. 来源与隔离范围

本目录由 Git 提交 `eca00c5ff9bb5bdc76a4d12b18af71a96d4f5891` 中的
`isaaclab_twist2_g1/script/eval_scripts/sonic` 完整复制得到，未复制工作区中
`sonic` 目录尚未提交的修改。上游原始通用入口默认是 Football，本目录的默认任务改为：

```text
Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody
tasks/common_env_config/doubledesk_sonic.yaml
```

共享的 40D ref-pose schema、SONIC controller 和动作字段解析没有改动。所有新增的
时序逻辑只在本目录的 `sim_eval_vla.py` 创建 action provider 后，对该评测进程中的单个
provider 实例生效，不影响原 `sonic` 入口。

## 2. 旧结果诊断

旧结果目录：

```text
isaaclab_twist2_g1/script/eval_scripts/sonic/eval_results/single_20260806_000853
```

主要证据如下：

- 40D 动作字段在训练数据和仿真端一致：`XY(2) + Z(1) + rot6d(6) + joints(29) + hands(2)`。
- 每个 25x40 chunk 在共享 action provider 中按行进入 FIFO，没有 flatten 后错位。
- episode 内前 10 个 chunk 与后 10 个 chunk 的首动作相比：
  - root XY 范数从 `0.00712` 降到 `0.00320`；
  - 关节绝对值均值从 `0.1825` 增到 `0.2295`；
  - 完整动作绝对值均值从 `0.2182` 增到 `0.2478`；
  - 相邻 chunk 首动作 L2 从 `0.3447` 降到 `0.1555`。
- 因此不是完整动作被服务器压成零，而是输出逐渐趋向变化较小的静态参考姿态。
- 旧测试使用的根目录 `pretrained_model` 与 `100000/pretrained_model` 配置及归一化器相同，
  但模型权重不同；旧评测发生在 100000 权重生成之前，不能代表最终 checkpoint。

## 3. 根因：状态历史时间尺度没有对齐

checkpoint 配置为：

```text
fps = 50
n_obs_steps = 10
obs_delta_sequence = [9,8,7,6,5,4,3,2,1,0]
chunk_size = 25
n_action_steps = 25
```

训练时 10 个 state 是相邻的 50 Hz 帧，覆盖 `9 * 0.02 = 0.18 s`。

原评测流程执行完 25 个动作才请求下一次推理。ACT 在线 history 在服务器端按“推理请求”
追加 state，因此相邻 history token 间隔实际是 `25 * 0.02 = 0.5 s`，10 帧覆盖约
`4.5 s`。模型却仍把它们解释为连续的 50 Hz 状态。这是时序语义错配，而不是 40D 字段错配。

另外，完整执行 25 步意味着每次有 0.5 秒没有视觉闭环更新；接近物体后容易因分布偏移或
误差积累而进入静态均值姿态。

## 4. 已实现修改

### 4.1 连续 50 Hz state history

`act_refpose_alignment.py` 在每个控制步调用共享 provider 原始逻辑之前采集一次 64D state，
维护长度为 10 的 ring buffer。请求推理时发送 `[10, 64]`：

```text
[t-9, t-8, ..., t]
```

episode 开头不足 10 帧时，使用最早可用 state 向左填充，与 trajectory boundary clipping
语义一致。reset 时本地 ring buffer 和 chunk 边界状态同时清空。

本目录的 `serve_act_refpose_vla_http.py` 接受 `[64]` 或 checkpoint 声明的
`[n_obs_steps, 64]`。经过 batch 后，ACT 收到 `[1, 10, 64]`，不会再次按 HTTP 请求频率
构造稀疏 history。

### 4.2 缩短开环执行窗口

模型仍预测完整 25 步，但默认只执行前 5 步，然后使用最新图像和连续 10 帧 state 重新规划。

```text
预测窗口：25 / 50 = 0.50 s
默认执行窗口：5 / 50 = 0.10 s
```

这样没有改变 checkpoint 的输出结构和动作顺序，只改变闭环重规划频率。

### 4.3 完整动作块诊断

默认在 `recordings/vla_outputs/*.jsonl` 中写入 `infer_chunk_aligned`：

- `observation_state_shape`
- `predicted_chunk_size`
- `executed_chunk_size`
- `first_action`
- `executed_last_action`
- `predicted_last_action`
- `boundary_l2`
- `root_xy_norm_mean`
- `action_abs_mean`
- `action_chunk`（默认完整 25x40）

`boundary_l2` 比较上一个实际执行前缀的最后动作和新 chunk 第一动作，可直接判断闭环
重规划边界是否存在大跳变。

## 5. 默认运行方法

建议先测试 100000 checkpoint 的 3 个 seeds：

```bash
MODEL_PATH=/home/user/Documents/fym/vla/outputs/act_sonic_refpose_v3_1_0805_20260805_fym/100000 \
EVAL_SEEDS="0 1 2" \
bash isaaclab_twist2_g1/script/eval_scripts/act_refpose/run_vla_eval.sh
```

`MODEL_PATH` 可以传 `100000`，脚本会自动进入其 `pretrained_model`。
当前机器若存在 `/home/user/anaconda3/envs/lerobot/bin/python`，脚本会优先用它启动
模型服务器，避免基础 Python 缺少 `safetensors`/LeRobot 依赖；也可显式设置
`SERVER_PYTHON=/path/to/python` 覆盖。

默认参数：

```bash
ACT_REFPOSE_HISTORY_STEPS=10
ACT_REFPOSE_EXECUTE_STEPS=5
ACT_REFPOSE_RECORD_FULL_CHUNKS=1
```

可进行以下消融：

```bash
# 只修正 state history，仍执行完整 25 步
ACT_REFPOSE_EXECUTE_STEPS=25

# 更强闭环，每步重新规划；推理成本最高
ACT_REFPOSE_EXECUTE_STEPS=1

# 不在 JSONL 保存完整 chunk，只保存统计和首尾动作
ACT_REFPOSE_RECORD_FULL_CHUNKS=0
```

推荐比较 `EXECUTE_STEPS=5/10/25`，其余条件和 seeds 保持一致。

## 6. 结果判断

- 若 100000 且 history 对齐后 root XY 不再持续收缩，主要问题是旧权重和时序错配。
- 若 `EXECUTE_STEPS=5` 显著好于 25，主要问题是 0.5 秒开环执行。
- 若 `boundary_l2` 持续很大，需要增加 chunk overlap/temporal ensemble 或边界平滑。
- 若预测的累计 root reference 持续移动但机器人实际位置不动，应继续检查 SONIC 跟踪、碰撞和控制增益。
- 若模型输出本身仍收敛到静态姿态，应处理数据中静止帧占比、阶段采样和恢复轨迹，而不是再次调整 40D 字段顺序。

## 7. 未修改部分

- 40D action layout 和关节顺序。
- rot6d row layout。
- root XY 在 reference base frame 中的语义。
- action 的 MEAN_STD 归一化和反归一化。
- SONIC encoder/decoder 模型及其控制增益。
- 原 `script/eval_scripts/sonic` 目录。

除 DoubleDesk 和通用入口外，其余任务专用 shell 文件只是为了保留 Git 原始目录结构，
没有作为本次 ACT DoubleDesk 对齐方案的验收入口。

## 8. 首次实机启动修正

Git 原始版本的 `sim_eval_vla.py` 在把 IsaacLab 项目根目录加入 `sys.path` 之前就导入
`task_runtime_profiles`。复制到独立目录并直接启动时会触发 `ModuleNotFoundError`，导致
`episodes/*.json` 记录 `failure_reason: process_error`。本目录已把该导入移到路径初始化
之后；这只修正启动路径，不改变任务、状态或动作语义。
