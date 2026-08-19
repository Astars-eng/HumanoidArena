#!/usr/bin/env bash
set -euo pipefail

RUN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Optional leading GPU index. Any remaining arguments are checkpoint paths.
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  export GPU_ID="${1}"
  shift
fi

export ENV_CONFIG_YAML="${ENV_CONFIG_YAML:-tasks/common_test_config/base_test/doubledesk_sonic_test.yaml}"
export RESULTS_TAG="${RESULTS_TAG:-HOI_double_desk_fm_new107}"
export MAX_STEPS="${MAX_STEPS:-2000}"

exec "${RUN_SCRIPT_DIR}/run_vla_eval_parallel.sh" "$@"
