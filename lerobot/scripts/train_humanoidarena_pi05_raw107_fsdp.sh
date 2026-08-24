#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Keep the original launcher and configuration as the single source of truth;
# this wrapper only selects the memory-sharded multi-GPU execution preset.
export FSDP_ENABLED="${FSDP_ENABLED:-true}"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export COMPILE_MODEL="${COMPILE_MODEL:-false}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export OUTPUT_DIR="${OUTPUT_DIR:-/DATA/disk0/fym/vla/outputs/pi05_humanoidarena_sonic_merged_raw107_fsdp}"
export JOB_NAME="${JOB_NAME:-pi05_humanoidarena_sonic_merged_raw107_fsdp}"

exec bash "${SCRIPT_DIR}/train_humanoidarena_pi05_raw107.sh"
