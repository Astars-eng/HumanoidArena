# ACT raw107 DoubleDesk 评测：接口诊断与修改记录

## 1. 目录来源与隔离边界

本目录从 Git 提交 `eca00c5ff9bb5bdc76a4d12b18af71a96d4f5891` 的
`script/eval_scripts/sonic` 原始文件复制得到，再进行 raw107 专用修改。原 `sonic` 的已跟踪
文件恢复到该 Git 版本，raw107 入口只放在本目录。

目标 checkpoint：

```text
/home/user/Documents/fym/vla/outputs/act_sonic_raw_20260804_fym/200000/pretrained_model
```

本目录只负责评测入口与接口选择。raw107 的共享机器人适配仍位于：

```text
action_provider/action_provider_sonic.py
action_provider/sonic_raw_policy_adapter.py
action_provider/lerobot_vla_http_client.py
../lerobot/scripts/serve_lerobot_vla_http.py
```

这些共享文件是 raw107 运行依赖，不能随 `sonic` 目录一起恢复，否则新入口无法传递分项状态、
接收逐控制步动作或拆分 107D 输出。

## 2. checkpoint 实际接口

从 `config.json`、processor 配置和训练数据元信息核对得到：

```text
policy             ACT
control/data fps   50 Hz
n_obs_steps        10（0.18 s 历史）
chunk_size         25（0.50 s）
n_action_steps     25

输入：
  observation.joint_pos       29
  observation.joint_vel       29
  observation.ang_vel_b        3
  observation.gravity          3
  observation.state           64 = 29+29+3+3
  observation.images.front     3x224x224

输出：
  action.applied_action        29
  action.encoder_token         64
  action.left_hand              7
  action.right_hand             7
  action                      107 = 29+64+7+7
```

这与 ref-pose checkpoint 的 `[64] -> [40]` 接口不同，不能复用 `act_refpose` 的 40D
root/ref-pose 解码。

## 3. 已执行的接口适配

### 3.1 输入本体状态

每个 50 Hz 控制步从 IsaacLab 读取 SONIC 数值顺序的 29D `q`、29D `qd`、base-frame 3D
角速度和 3D projected gravity。HTTP 同时发送四个具名字段和严格相同的 64D 拼接字段，
让 checkpoint 自带的 `concat_policy_features` processor 按训练顺序工作。

训练数据的部分旧 metadata joint labels 是 DFS 分组顺序，但原始数值由 SONIC `_sonic_idx`
记录；运行端保留实际训练数值顺序，不根据旧标签二次重排。

### 3.2 图像几何

训练数据是直接拉伸到 `224x224` 的 RGB。server 默认启用
`--stretch-image-to-policy-shape`，按 converter 的 OpenCV 插值重现该几何，不使用保持宽高比
的 padding。

### 3.3 task 与 robot metadata

数据集 task 字符串是：

```text
Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody
```

server 启用 `--verbatim-task`，不把它改写成英文 instruction。物理仿真仍使用
`unitree_g1_refpose_v3_1` 选择 G1+Dex3 资产；发给 raw checkpoint 的 LeRobot `robot_type`
单独设为训练元数据中的 `g1`。

### 3.4 107D 输出拆分

输出严格按 `[0:29] / [29:93] / [93:100] / [100:107]` 拆为 body raw、encoder token、
左手和右手。两只手的连续 7D 目标直接写入，不做二值阈值、抓取规则或任务修正。

此前 `enable_dex3_dds=True` 同时承担“启用 Dex3 关节”和“订阅 Dex3 DDS 命令”两种职责；
当 DDS 中存在有效手部命令时，`_apply_hand_targets()` 会优先采用 DDS 值，覆盖上述模型输出。
现在 `act_raw` 将两项职责解耦：`enable_dex3_dds=False`，同时设置
`enable_dex3_model_control=True`。因此不会创建或订阅 Dex3 DDS 命令对象，但仍建立 14 个
Dex3 关节索引，并在每个物理步把模型预测的左右手连续 7D 关节目标直接写入 IsaacLab。
这里的 IsaacLab 关节位置执行器仍是仿真所必需的底层执行机制，不是 DDS 辅助控制。

### 3.5 body 执行接口

默认 `SONIC_RAW107_BODY_SOURCE=native_decoder`：把预测的 64D encoder token 送入原生 SONIC
decoder，并使用实时机器人本体反馈形成 29D body target。这是数据采集时的任务无关闭环 GMT
接口。

保留 `direct_raw` 消融模式：直接把预测的 29D raw head 按
`raw * G1_ACTION_SCALE_ISAACLAB + default_joint_pos` 转成关节目标。该模式不含原生 decoder
反馈，只用于映射审计：

```bash
SONIC_RAW107_BODY_SOURCE=direct_raw \
  bash isaaclab_twist2_g1/script/eval_scripts/act_raw/run_raw107_doubledesk_eval.sh
```

### 3.6 时序接口

客户端每个 50 Hz 控制步调用一次 policy `select_action`。ACT 在线 history 因而每步追加一个
状态，覆盖训练所需的连续 10 帧；policy 使用自身的 25 步 action queue，每 25 步重新预测。
本入口不复用 ref-pose 的“预测 25、执行 5”策略，因为 raw107 当前走 checkpoint 原生
`select_action` 队列语义。

### 3.7 启动与失败诊断

- 默认使用 `/home/user/anaconda3/envs/lerobot/bin/python` 和训练 fork
  `/home/user/Documents/fym/vla/src`。
- server readiness timeout 为 1800 秒，并在 server 提前退出时回显日志尾部。
- `raw107_contract.py` 在启动 server 前校验 policy type、所有输入/输出 key、shape、history
  和 chunk 参数，避免把 40D/64D checkpoint 误接到 raw107 路径。
- episode JSON 额外保存 `sonic_vla_action_format` 和 `sonic_raw107_body_source`。
- 可选输出 delay 被固定为 0，避免环境变量改变时序。

## 4. 默认场景与运行方法

默认使用非 semantic 的 DoubleDesk `base_test`：

```text
tasks/common_test_config/base_test/doubledesk_sonic_test.yaml
```

运行 seed 0/1/2、每个一次：

```bash
bash isaaclab_twist2_g1/script/eval_scripts/act_raw/run_raw107_doubledesk_eval.sh
```

或使用通用入口：

```bash
MODEL_PATH=/home/user/Documents/fym/vla/outputs/act_sonic_raw_20260804_fym/200000/pretrained_model \
ENV_CONFIG_YAML=tasks/common_test_config/base_test/doubledesk_sonic_test.yaml \
EVAL_SEEDS="0 1 2" \
REPEATS_PER_SEED=1 \
bash isaaclab_twist2_g1/script/eval_scripts/act_raw/run_vla_eval.sh
```

## 5. 公平性边界

适配器不读取物体位置、奖励、接触、目标状态或其他任务状态，也不包含 IK、轨迹规划、抓取
latch、动作平滑、动作限幅、PD 调参或任务规则。默认 native decoder 属于需要披露的机器人
接口稳定层；正式对比时应同时记录 body source，并可用 `direct_raw` 做消融。
