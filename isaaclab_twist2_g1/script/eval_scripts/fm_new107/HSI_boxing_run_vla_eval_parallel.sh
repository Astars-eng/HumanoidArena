#!/usr/bin/env bash
set -euo pipefail

RUN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="$(cd "${RUN_SCRIPT_DIR}/../../.." && pwd)"

# Optional leading GPU index. Any remaining arguments are checkpoint paths.
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  export GPU_ID="${1}"
  shift
fi

export ENV_CONFIG_YAML="${ENV_CONFIG_YAML:-tasks/common_test_config/base_test/boxing_sonic_test.yaml}"
RESULTS_TAG_BASE="${RESULTS_TAG:-1test_HSI_boxing_sonic_batch_2000_mix}"
if [[ -n "${RESULTS_TAG_PREFIX:-}" ]]; then
  export RESULTS_TAG="${RESULTS_TAG_PREFIX}_${RESULTS_TAG_BASE}"
else
  export RESULTS_TAG="${RESULTS_TAG_BASE}"
fi
export MAX_STEPS="${MAX_STEPS:-1500}"
exec "${RUN_SCRIPT_DIR}/run_vla_eval_parallel.sh" "$@"
