#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1090
source "${ROOT_DIR}/scripts/_common.sh"

ensure_venv
cd "${ROOT_DIR}"

run_module --config config/pipeline.json run
