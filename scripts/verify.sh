#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1090
source "${ROOT_DIR}/scripts/_common.sh"

ensure_venv
cd "${ROOT_DIR}"

run_module --config config/pipeline.json run

python - <<'PY'
import json
from pathlib import Path

metrics_path = Path("results/metrics.json")
metrics = json.loads(metrics_path.read_text())

anomaly_accuracy = metrics["anomaly_detection"]["accuracy"]
error_rate = metrics["error_correction"]["bit_error_rate"]
mean_reward = metrics["key_distribution"]["mean_reward"]

assert anomaly_accuracy >= 0.7, f"Anomaly accuracy too low: {anomaly_accuracy}"
assert error_rate <= 0.35, f"Bit error rate too high: {error_rate}"
assert mean_reward >= 0.55, f"Mean reward too low: {mean_reward}"
print("Smoke checks passed:", anomaly_accuracy, error_rate, mean_reward)
PY

PYTHONPATH=src python -m unittest discover -s tests
