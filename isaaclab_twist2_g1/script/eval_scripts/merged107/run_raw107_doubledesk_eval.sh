#!/usr/bin/env bash

# Compatibility wrapper for running all three merged107 checkpoints on DoubleDesk.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_CHECKPOINTS=(
  "/DATA/disk0/fym/vla/outputs/act_humanoidarena_sonic_merged_107_0820_fym/checkpoints/100000/pretrained_model"
  "/DATA/disk0/fym/vla/outputs/dp_humanoidarena_sonic_merged_act107_0820_fym/checkpoints/100000/pretrained_model"
  "/DATA/disk0/fym/vla/outputs/fm_humanoidarena_sonic_merged_act107_0820_fym/checkpoints/100000/pretrained_model"
)

if [[ "$#" -gt 0 ]]; then
  CHECKPOINTS=("$@")
else
  CHECKPOINTS=("${DEFAULT_CHECKPOINTS[@]}")
fi

export ENV_CONFIG_YAML="${ENV_CONFIG_YAML:-tasks/common_test_config/base_test/doubledesk_sonic_test.yaml}"
export SONIC_VLA_ACTION_FORMAT="raw107"
export SONIC_VLA_STATE_FORMAT="raw93"
export SONIC_RAW_STATE_JOINT_ORDER="mujoco"
# [control stabilization] Route the predicted motion token through the native
# task-agnostic SONIC GMT decoder. Set direct_raw only for an open-loop mapping
# audit; it removes the robot-state feedback used during data collection.
export SONIC_RAW107_BODY_SOURCE="${SONIC_RAW107_BODY_SOURCE:-native_decoder}"
# [parameter tuning] Pin the pre-existing optional delay to the neutral value so
# the evaluation wrapper cannot silently change checkpoint timing via ambient env.
export SONIC_OUTPUT_DELAY_STEPS="0"
export SERVER_PYTHON="${SERVER_PYTHON:-/root/miniconda3/envs/lerobot_new/bin/python}"
export SERVER_LEROBOT_SRC="${SERVER_LEROBOT_SRC:-/DATA/disk0/fym/vla/src}"
export SERVER_CHECKPOINT_REF_REMAP="${SERVER_CHECKPOINT_REF_REMAP:-}"
# Preserve the exact dataset prompt resolved by the merged107 task map.
export SERVER_VERBATIM_TASK="1"
# [interface conversion] The dataset converter stretched source frames to
# 224x224 with OpenCV; reproduce that geometry before checkpoint preprocessing.
export SERVER_STRETCH_IMAGE_TO_POLICY_SHAPE="1"
export EVAL_SEEDS="${EVAL_SEEDS:-0 1 2}"
export REPEATS_PER_SEED="${REPEATS_PER_SEED:-1}"
export PERSISTENT_SIM="${PERSISTENT_SIM:-0}"

RESULTS_ROOT="${RESULTS_ROOT:-${SCRIPT_DIR}/eval_results/merged107_doubledesk_$(date +%Y%m%d_%H%M%S)}"

for checkpoint in "${CHECKPOINTS[@]}"; do
  if [[ ! -d "${checkpoint}" ]]; then
    echo "[merged107] checkpoint not found: ${checkpoint}" >&2
    exit 2
  fi
  model_label="$(basename "$(dirname "$(dirname "$(dirname "${checkpoint}")")")")_$(basename "$(dirname "${checkpoint}")")"
  MODEL_PATH="${checkpoint}" \
  RESULTS_DIR="${RESULTS_ROOT}/${model_label}" \
    bash "${SCRIPT_DIR}/run_vla_eval.sh"
done
