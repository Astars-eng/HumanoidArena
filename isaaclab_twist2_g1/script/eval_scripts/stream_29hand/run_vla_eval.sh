#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry point. All stream_29hand evaluations use the split-capable
# scheduler so server ports and model/Isaac GPU placement are handled safely.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${1:-}" && -z "${MODEL_PATH:-}" ]]; then
  export MODEL_PATH="$1"
fi
if [[ -n "${EVAL_SEEDS:-}" && -z "${SEEDS_OVERRIDE:-}" ]]; then
  export SEEDS_OVERRIDE="${EVAL_SEEDS}"
fi

exec "${SCRIPT_DIR}/run_vla_eval_parallel.sh"
