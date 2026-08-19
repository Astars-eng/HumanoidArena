#!/usr/bin/env bash

# Fair DoubleDesk evaluation for SONIC DFS/MuJoCo raw29 checkpoints.  This wrapper only
# selects the matching robot/policy interface; it adds no task controller.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_CHECKPOINTS=(
  "/DATA/disk0/fym/vla/outputs/finetune_stream_humanoidarena_sonic_new_multisource_29d_20260815_153724/checkpoints/040000/pretrained_model"
)

if [[ "$#" -gt 0 ]]; then
  CHECKPOINTS=("$@")
else
  CHECKPOINTS=("${DEFAULT_CHECKPOINTS[@]}")
fi

export ENV_CONFIG_YAML="${ENV_CONFIG_YAML:-tasks/common_test_config/base_test/doubledesk_sonic_test.yaml}"
export SONIC_VLA_ACTION_FORMAT="raw29"
# raw29 is an open-loop decoder raw action; it has no motion-token branch.
export SONIC_RAW107_BODY_SOURCE="${SONIC_RAW107_BODY_SOURCE:-direct_raw}"
# [parameter tuning] Pin the pre-existing optional delay to the neutral value so
# the evaluation wrapper cannot silently change checkpoint timing via ambient env.
export SONIC_OUTPUT_DELAY_STEPS="0"
export SERVER_PYTHON="${SERVER_PYTHON:-/root/miniconda3/envs/lerobot_new/bin/python}"
export SERVER_LEROBOT_SRC="${SERVER_LEROBOT_SRC:-/DATA/disk0/fym/vla/src}"
export SERVER_CHECKPOINT_REF_REMAP="${SERVER_CHECKPOINT_REF_REMAP:-}"
# [interface conversion] The dataset contains the literal HumanoidArena task ID;
# do not replace it with a generic English alias at inference time.
export SERVER_VERBATIM_TASK="1"
# [interface conversion] The dataset converter stretched source frames to
# 224x224 with OpenCV; reproduce that geometry before checkpoint preprocessing.
export SERVER_STRETCH_IMAGE_TO_POLICY_SHAPE="1"
export EVAL_SEEDS="${EVAL_SEEDS:-0 1 2}"
export REPEATS_PER_SEED="${REPEATS_PER_SEED:-1}"
export PERSISTENT_SIM="${PERSISTENT_SIM:-0}"

RESULTS_ROOT="${RESULTS_ROOT:-${SCRIPT_DIR}/eval_results/raw29_doubledesk_$(date +%Y%m%d_%H%M%S)}"

for checkpoint in "${CHECKPOINTS[@]}"; do
  if [[ ! -d "${checkpoint}" ]]; then
    echo "[stream_29d] checkpoint not found: ${checkpoint}" >&2
    exit 2
  fi
  model_label="$(basename "$(dirname "$(dirname "${checkpoint}")")")_$(basename "$(dirname "${checkpoint}")")"
  MODEL_PATH="${checkpoint}" \
  RESULTS_DIR="${RESULTS_ROOT}/${model_label}" \
    bash "${SCRIPT_DIR}/run_vla_eval.sh"
done
