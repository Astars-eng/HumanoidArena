#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/lerobot_new/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/DATA/disk0/fym/vla/data/HumanoidArena_lerobot_sonic_new_merged_act107}"
PI05_BASE_PATH="${PI05_BASE_PATH:-/DATA/disk0/fym/models/pi05_base}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/DATA/disk0/fym/models/paligemma-3b-pt-224}"
OUTPUT_DIR="${OUTPUT_DIR:-/DATA/disk0/fym/vla/outputs/pi05_humanoidarena_sonic_merged_raw107}"
JOB_NAME="${JOB_NAME:-pi05_humanoidarena_sonic_merged_raw107}"
GPU_IDS="${GPU_IDS:-0,1}"
STEPS="${STEPS:-200000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SAVE_FREQ="${SAVE_FREQ:-20000}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
NUM_WORKERS="${NUM_WORKERS:-4}"
COMPILE_MODEL="${COMPILE_MODEL:-true}"
FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-false}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-false}"
FSDP_ENABLED="${FSDP_ENABLED:-false}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
LOG_FREQ="${LOG_FREQ:-200}"
EVAL_FREQ="${EVAL_FREQ:-20000}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-false}"
RESUME_CONFIG="${RESUME_CONFIG:-}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[pi05-raw107] Python is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_humanoidarena_pi05_raw107.py" \
  --dataset-root "${DATASET_ROOT}" \
  --checkpoint-root "${PI05_BASE_PATH}" \
  --tokenizer-root "${TOKENIZER_PATH}" \
  --tokenizer-max-length 512

if [[ "${PREFLIGHT_ONLY}" == "true" ]]; then
  echo "[pi05-raw107] preflight-only validation completed"
  exit 0
fi

IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
NPROC_PER_NODE="${#GPU_ID_ARRAY[@]}"
if [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
  echo "[pi05-raw107] GPU_IDS must contain at least one GPU id" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

VISIBLE_GPU_COUNT="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${VISIBLE_GPU_COUNT}" -lt "${NPROC_PER_NODE}" ]]; then
  echo "[pi05-raw107] requested ${NPROC_PER_NODE} process(es), but PyTorch sees ${VISIBLE_GPU_COUNT} GPU(s)" >&2
  exit 2
fi

cd "${LEROBOT_ROOT}"
TRAIN_ARGS=(
  --dataset.repo_id=local/HumanoidArena_lerobot_sonic_new_merged_act107
  --dataset.root="${DATASET_ROOT}"
  --dataset.image_transforms.enable=true
  --dataset.image_transforms.max_num_transforms=3
  --dataset.image_transforms.random_order=false
  --dataset.use_imagenet_stats=true
  --dataset.video_backend=torchcodec
  --policy.type=pi05
  --policy.pretrained_path="${PI05_BASE_PATH}"
  --policy.adapt_action_head_from_pretrained=true
  --policy.device=cuda
  --policy.max_state_dim=93
  --policy.max_action_dim=107
  --policy.tokenizer_name="${TOKENIZER_PATH}"
  --policy.tokenizer_max_length=512
  --policy.n_obs_steps=1
  --policy.chunk_size=20
  --policy.n_action_steps=20
  --policy.optimizer_lr=2.5e-5
  --policy.push_to_hub=false
  --policy.compile_model="${COMPILE_MODEL}"
  --policy.gradient_checkpointing=true
  --policy.dtype=bfloat16
  --policy.freeze_vision_encoder="${FREEZE_VISION_ENCODER}"
  --policy.train_expert_only="${TRAIN_EXPERT_ONLY}"
  --fsdp.enabled="${FSDP_ENABLED}"
  --wandb.enable="${WANDB_ENABLE}"
  --seed=42
  --batch_size="${BATCH_SIZE}"
  --num_workers="${NUM_WORKERS}"
  --steps="${STEPS}"
  --eval_freq="${EVAL_FREQ}"
  --log_freq="${LOG_FREQ}"
  --save_freq="${SAVE_FREQ}"
  --save_checkpoint="${SAVE_CHECKPOINT}"
  --use_policy_training_preset=true
  --output_dir="${OUTPUT_DIR}"
  --job_name="${JOB_NAME}"
)

if [[ -n "${RESUME_CONFIG}" ]]; then
  if [[ ! -f "${RESUME_CONFIG}" ]]; then
    echo "[pi05-raw107] resume config does not exist: ${RESUME_CONFIG}" >&2
    exit 2
  fi
  TRAIN_ARGS+=(--resume=true --config_path="${RESUME_CONFIG}")
fi

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  src/lerobot/scripts/lerobot_train.py \
  "${TRAIN_ARGS[@]}"
