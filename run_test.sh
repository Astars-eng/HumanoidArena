#!/usr/bin/env bash

set -e


# Isaac仿真环境
# conda activate unitree_sim_env


# LeRobot推理环境
export SERVER_PYTHON=/home/user/anaconda3/envs/lerobot/bin/python

# StreamPolicy 在训练仓库的 LeRobot fork 中，不在 HumanoidArena 自带的 LeRobot 中。
# eval_vla_suite 会清理 PYTHONPATH，因此通过 LEROBOT_SRC 显式告诉推理服务使用哪份源码。
export LEROBOT_SRC=/home/user/Documents/fym/vla/src

# 16 GB Stream checkpoint 的首次构建和权重加载可能超过默认 60 秒。
export SERVER_READY_TIMEOUT=300


# checkpoint
export MODEL_PATH=/home/user/Documents/fym/vla/outputs/doubledesk_stream_sonic_refpose_bs64_20260803_fym/checkpoints/020000
    

# task环境
export ENV_CONFIG_YAML=tasks/common_env_config/doubledesk_sonic.yaml


# sonic encoder decoder
export SONIC_POLICY_ROOT=/home/user/Documents/HumanoidArena/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release


# 测试seed
export EVAL_SEEDS="0"


# 启动evaluation
bash isaaclab_twist2_g1/script/eval_scripts/sonic/run_vla_eval.sh
