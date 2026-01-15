"""Evaluation helpers for the QKD pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from .anomaly import evaluate as evaluate_anomaly
from .error_correction import evaluate as evaluate_error_correction
from .rl import evaluate_policy


def save_metrics(metrics: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))


def summarize_metrics(metrics: Dict[str, Any]) -> str:
    lines = ["Verification metrics:"]
    for section, values in metrics.items():
        lines.append(f"- {section}:")
        for key, value in values.items():
            lines.append(f"  - {key}: {value:.4f}" if isinstance(value, float) else f"  - {key}: {value}")
    return "\n".join(lines)


def evaluate_pipeline(
    anomaly_model,
    anomaly_features,
    anomaly_labels,
    error_model,
    noisy_sequences,
    target_sequences,
    rl_policy,
) -> Dict[str, Any]:
    metrics = {
        "anomaly_detection": evaluate_anomaly(anomaly_model, anomaly_features, anomaly_labels),
        "error_correction": evaluate_error_correction(error_model, noisy_sequences, target_sequences),
        "key_distribution": evaluate_policy(rl_policy),
    }
    return metrics
