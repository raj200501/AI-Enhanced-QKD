"""Configuration loading and validation utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any


@dataclass(frozen=True)
class DataConfig:
    """Configuration for data generation and preprocessing."""

    seed: int = 1337
    sessions: int = 200
    sequence_length: int = 32
    noise_mean: float = 0.08
    noise_std: float = 0.03
    loss_mean: float = 0.12
    loss_std: float = 0.04


@dataclass(frozen=True)
class AnomalyConfig:
    """Configuration for anomaly detection training."""

    learning_rate: float = 0.4
    epochs: int = 120
    anomaly_threshold: float = 0.15


@dataclass(frozen=True)
class ErrorCorrectionConfig:
    """Configuration for error-correction tuning."""

    window_sizes: tuple[int, ...] = (1, 3, 5, 7)


@dataclass(frozen=True)
class RLConfig:
    """Configuration for the bandit-style RL policy."""

    actions: int = 4
    episodes: int = 200
    epsilon: float = 0.12
    alpha: float = 0.4


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration for the QKD pipeline."""

    data: DataConfig = DataConfig()
    anomaly: AnomalyConfig = AnomalyConfig()
    error_correction: ErrorCorrectionConfig = ErrorCorrectionConfig()
    rl: RLConfig = RLConfig()


DEFAULT_CONFIG = PipelineConfig()


def _merge_defaults(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for key, value in defaults.items():
        if key not in overrides:
            merged[key] = value
            continue
        if isinstance(value, dict):
            merged[key] = _merge_defaults(value, overrides.get(key, {}))
        else:
            merged[key] = overrides[key]
    return merged


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate configuration from a JSON file."""

    path = Path(path)
    raw = json.loads(path.read_text())
    defaults = {
        "data": DataConfig().__dict__,
        "anomaly": AnomalyConfig().__dict__,
        "error_correction": {"window_sizes": list(ErrorCorrectionConfig().window_sizes)},
        "rl": RLConfig().__dict__,
    }
    merged = _merge_defaults(defaults, raw)
    return PipelineConfig(
        data=DataConfig(**merged["data"]),
        anomaly=AnomalyConfig(**merged["anomaly"]),
        error_correction=ErrorCorrectionConfig(
            window_sizes=tuple(merged["error_correction"]["window_sizes"])
        ),
        rl=RLConfig(**merged["rl"]),
    )


def save_default_config(path: str | Path) -> None:
    """Persist the default configuration to disk."""

    path = Path(path)
    payload = {
        "data": DEFAULT_CONFIG.data.__dict__,
        "anomaly": DEFAULT_CONFIG.anomaly.__dict__,
        "error_correction": {"window_sizes": list(DEFAULT_CONFIG.error_correction.window_sizes)},
        "rl": DEFAULT_CONFIG.rl.__dict__,
    }
    path.write_text(json.dumps(payload, indent=2))
