#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "$0")/common.sh"

if [[ "${GGBOT_EVAL_JUDGE:-false}" == "true" ]]; then
  run_suite golden --judge "$@"
else
  run_suite golden "$@"
fi
