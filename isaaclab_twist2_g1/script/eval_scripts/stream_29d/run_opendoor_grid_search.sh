#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${GRID_CONFIG:-${SCRIPT_DIR}/grid_configs/open_door_stream29d.json}"
if [[ "${REFINER_MODE:-off}" == "on" ]]; then
  export DISABLE_REFINER="${DISABLE_REFINER:-0}"
elif [[ "${REFINER_MODE:-off}" == "off" ]]; then
  export DISABLE_REFINER="${DISABLE_REFINER:-1}"
else
  echo "REFINER_MODE must be 'on' or 'off', got: ${REFINER_MODE}" >&2
  exit 2
fi

case "${1:---launch}" in
  --launch) exec python3 "${SCRIPT_DIR}/eval_grid.py" launch --config "${CONFIG}" ;;
  --manifest) exec python3 "${SCRIPT_DIR}/eval_grid.py" manifest --config "${CONFIG}" ;;
  --summarize) exec python3 "${SCRIPT_DIR}/eval_grid.py" summarize --config "${CONFIG}" ;;
  --dry-run) exec python3 "${SCRIPT_DIR}/eval_grid.py" launch --config "${CONFIG}" --dry-run ;;
  --worker)
    [[ "${2:-}" =~ ^[0-9]+$ ]] || { echo "worker id must be a non-negative integer" >&2; exit 2; }
    exec python3 "${SCRIPT_DIR}/eval_grid.py" worker --config "${CONFIG}" --worker-id "$2"
    ;;
  *) echo "usage: $0 [--launch | --manifest | --summarize | --dry-run | --worker N]" >&2; exit 2 ;;
esac
