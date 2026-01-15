#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1090
source "${ROOT_DIR}/scripts/_common.sh"

ensure_venv

python - <<'PY'
import sys
print("Python", sys.version)
PY
