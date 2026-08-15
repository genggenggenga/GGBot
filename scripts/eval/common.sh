#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing ${PYTHON_BIN}. Create the project virtual environment first." >&2
  exit 1
fi

run_suite() {
  local suite="$1"
  shift
  cd "${ROOT_DIR}"
  exec "${PYTHON_BIN}" -m evaluation.run \
    --suite "${suite}" \
    --mode deterministic \
    "$@"
}

run_chunking() {
  cd "${ROOT_DIR}"
  exec "${PYTHON_BIN}" -m evaluation.chunking_eval "$@"
}
