#!/usr/bin/env bash
set -euo pipefail

RUN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Optional positional GPU index: `...parallel.sh 3` is equivalent to GPU_ID=3.
if [[ -n "${1:-}" ]]; then
  if [[ ! "${1}" =~ ^[0-9]+$ ]]; then
    echo "Error: optional first argument must be one physical GPU index, got: ${1}" >&2
    exit 2
  fi
  export GPU_ID="${1}"
  shift
fi

export ENV_CONFIG_YAML="${ENV_CONFIG_YAML:-tasks/common_test_config/base_test/football_single_sonic_test.yaml}"
RESULTS_TAG_BASE="${RESULTS_TAG:-HOI_football_dp_new107}"
if [[ -n "${RESULTS_TAG_PREFIX:-}" ]]; then
  export RESULTS_TAG="${RESULTS_TAG_PREFIX}_${RESULTS_TAG_BASE}"
else
  export RESULTS_TAG="${RESULTS_TAG_BASE}"
fi
export MAX_STEPS="${MAX_STEPS:-1300}"

exec "${RUN_SCRIPT_DIR}/run_vla_eval_parallel.sh" "$@"
