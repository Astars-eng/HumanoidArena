# SONIC raw107 policy adapter

## Scope and fairness boundary

This adapter performs robot/policy interface alignment only. It never reads
object poses, reward, contacts, target state, or any other task/environment
state. It contains no grasping, trajectory planning, rule control, or
task-specific action correction.

## Audited checkpoint contract

Both evaluated checkpoints use the same action and proprioception contract:

```text
observation.state (64)
  [0:29]   robot joint position (rad)
  [29:58]  robot joint velocity (rad/s)
  [58:61]  body angular velocity in base frame (rad/s)
  [61:64]  projected gravity in base frame (unit vector)

action (107)
  [0:29]    action.applied_action = SONIC decoder_raw_action
  [29:93]   action.encoder_token  = SONIC encoder_latent
  [93:100]  action.left_hand      = seven hand joint targets (rad)
  [100:107] action.right_hand     = seven hand joint targets (rad)
```

The source converter copied `robot_qpos_before_decimation` and
`robot_qvel_before_decimation` without reordering. Those arrays are recorded by
`SonicActionProvider` through `_sonic_idx`, so their actual numeric order is
SONIC IsaacLab order. Legacy dataset feature names describe DFS order, but
reordering from those labels would corrupt the input actually seen in
training. The adapter therefore reproduces the recorded numeric order.

## HumanoidArena SONIC control contract

The SONIC backend executes 29 absolute body joint-position targets in SONIC
IsaacLab order, plus two seven-joint hand targets. DoubleDesk SONIC control
runs at 50 Hz (`physics_dt=0.005 s`, `decimation=4`).

The native SONIC decoder conversion is:

```text
body_joint_target_rad = decoder_raw_action * G1_ACTION_SCALE_ISAACLAB
                        + SONIC_DEFAULT_POS
```

The deploy path intentionally does not clip `decoder_raw_action` before this
conversion. The robot/actuator model remains responsible for its normal
physical limits.

## Minimal adapter

| Change | Category | Behavior/fairness note |
| --- | --- | --- |
| Strictly split the 107-D output into 29/64/7/7 fields | interface conversion | No values are changed. |
| Feed the predicted 64-D encoder token through the native SONIC decoder using live robot proprioception | control stabilization | Restores the task-agnostic closed-loop GMT interface used to produce the demonstrations. This changes body motion relative to direct 29-D execution and must be disclosed for benchmark fairness. |
| Send continuous 7-D left/right hand outputs directly as hand joint targets | interface conversion | No binary threshold, grasp rule, or hand-pose substitution. |
| Retain `direct_raw` as an explicit audit mode | interface conversion | Applies `raw * scale + default` without clipping/smoothing; useful for mapping checks, but it omits the native decoder's robot-state feedback. |
| Build split robot proprioception and the identical concatenated 64-D state | interface conversion | Only robot state is exposed; HTTP rejects arbitrary extra state fields. |
| Submit one state sample per 50 Hz control tick and use the policy's native action queue | interface conversion | Restores the checkpoint's 10-step history timing and 25-step chunk behavior. |
| Remap missing checkpoint-internal model/tokenizer paths in a temporary server directory | interface conversion | Does not edit checkpoint files or tensors. |
| Pass the literal HumanoidArena task ID through unchanged | interface conversion | The training dataset contains that exact string; translating it to an English alias changes policy conditioning. |
| Stretch camera RGB to the checkpoint's 224x224 input with the converter's OpenCV interpolation | interface conversion | Reproduces training geometry; avoids the model's otherwise different aspect-preserving padded resize. |
| Pin the existing optional output delay to zero in the evaluation wrapper | parameter tuning | Neutral/default timing; prevents an ambient environment variable from silently changing policy behavior. |

The one enabled `control stabilization` component is the fixed native SONIC
decoder. It consumes only the checkpoint's predicted encoder token and live
robot proprioception (`q`, `qd`, base angular velocity, projected gravity, and
raw-action history). It does not consume language, images, object poses,
contacts, rewards, or task state. No IK, smoothing, rate limiting, output
delay, body-action clipping, joint-limit override, grasp latch, or
task-conditioned correction is added. The launcher explicitly pins output
delay to its neutral value of zero and sends the policy's continuous hand
outputs directly.

Because both the 29-D raw head and the 64-D token head are supervised, the
choice of body execution hierarchy can affect benchmark results. The default
`native_decoder` setting follows the SONIC/GMT closed-loop robot interface;
for a fully transparent ablation, run the same evaluation with
`SONIC_RAW107_BODY_SOURCE=direct_raw` and report it as open-loop raw execution.

The native HumanoidArena SONIC PD/actuator configuration still affects
tracking, as it does for every policy evaluated through that backend. Changing
PD gains, enabling effort control, enabling output delay, or adding smoothing
would be a benchmark-affecting parameter/control change and should be reported
as a separate experiment.

## Evaluation

The wrapper defaults to the benchmark DoubleDesk base-test configuration,
seeds `0 1 2`, and 20 repeats per seed for both supplied checkpoints:

```bash
bash isaaclab_twist2_g1/script/eval_scripts/sonic/run_raw107_doubledesk_eval.sh
```

For a one-episode integration run:

```bash
EVAL_SEEDS=0 REPEATS_PER_SEED=1 PERSISTENT_SIM=0 \
  bash isaaclab_twist2_g1/script/eval_scripts/sonic/run_raw107_doubledesk_eval.sh \
  /path/to/checkpoint/pretrained_model
```

The wrapper uses the policy's training fork at
`/home/user/Documents/fym/vla/src` and remaps the unavailable training-machine
PaliGemma path to `/home/user/Downloads/paligemma_model`. Both may be overridden
with `SERVER_LEROBOT_SRC` and `SERVER_CHECKPOINT_REF_REMAP`.
