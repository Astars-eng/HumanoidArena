#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry point. All fm_new107 evaluations use the single-GPU
# scheduler so server ports are selected safely instead of being fixed at 8443.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${EVAL_SEEDS:-}" && -z "${SEEDS_OVERRIDE:-}" ]]; then
  export SEEDS_OVERRIDE="${EVAL_SEEDS}"
fi

# Forward checkpoint arguments unchanged.  The parallel entry point gives these
# explicit paths precedence over inherited MODEL_PATHS_* selectors.
exec "${SCRIPT_DIR}/run_vla_eval_parallel.sh" "$@"
