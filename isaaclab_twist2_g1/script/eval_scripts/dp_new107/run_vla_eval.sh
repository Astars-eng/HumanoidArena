#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry point. All dp_new107 evaluations use the single-GPU
# scheduler so server ports are selected safely instead of being fixed at 8443.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${1:-}" && -z "${MODEL_PATH:-}" ]]; then
  export MODEL_PATH="$1"
fi
if [[ -n "${EVAL_SEEDS:-}" && -z "${SEEDS_OVERRIDE:-}" ]]; then
  export SEEDS_OVERRIDE="${EVAL_SEEDS}"
fi

exec "${SCRIPT_DIR}/run_vla_eval_parallel.sh"
