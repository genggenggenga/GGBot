#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/run-smoke.sh" "$@"
"${SCRIPT_DIR}/run-bad-cases.sh" "$@"
"${SCRIPT_DIR}/run-golden.sh" "$@"
"${SCRIPT_DIR}/run-chunking.sh" "$@"
"${SCRIPT_DIR}/run-enterprise-rag.sh" "$@"
