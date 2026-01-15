#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

ensure_venv() {
  if [[ ! -d "${VENV_DIR}" ]]; then
    python -m venv "${VENV_DIR}"
  fi
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
}

run_module() {
  PYTHONPATH="${ROOT_DIR}/src" python -m ai_qkd.cli "$@"
}
