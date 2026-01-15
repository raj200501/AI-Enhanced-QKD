"""Synthetic QKD data generation and preprocessing utilities."""

from __future__ import annotations

import csv
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .config import DataConfig
from .math_utils import mean, stddev


RAW_COLUMNS = (
    "session_id",
    "time_index",
    "basis",
    "bit",
    "noise_level",
    "channel_loss",
    "detector_click",
)


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def generate_raw_sessions(config: DataConfig) -> List[Dict[str, float]]:
    """Generate deterministic synthetic QKD data.

    Each row represents a measurement event in a QKD session.
    """

    rng = _rng(config.seed)
    rows: List[Dict[str, float]] = []
    for session_id in range(config.sessions):
        for idx in range(config.sequence_length):
            noise = max(rng.gauss(config.noise_mean, config.noise_std), 0.0)
            loss = max(rng.gauss(config.loss_mean, config.loss_std), 0.0)
            basis = float(rng.randint(0, 1))
            bit = float(rng.randint(0, 1))
            detector_click = float(1.0 if rng.random() > (noise + loss) else 0.0)
            rows.append(
                {
                    "session_id": float(session_id),
                    "time_index": float(idx),
                    "basis": basis,
                    "bit": bit,
                    "noise_level": noise,
                    "channel_loss": loss,
                    "detector_click": detector_click,
                }
            )
    return rows


def write_raw_csv(rows: Iterable[Dict[str, float]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_raw_csv(path: str | Path) -> List[Dict[str, float]]:
    path = Path(path)
    with path.open() as handle:
        reader = csv.DictReader(handle)
        return [
            {key: float(value) for key, value in row.items()} for row in reader
        ]


def _normalize(values: List[float]) -> Tuple[List[float], float, float]:
    avg = mean(values)
    deviation = stddev(values, mean_value=avg)
    deviation = deviation if deviation > 0 else 1.0
    normalized = [(value - avg) / deviation for value in values]
    return normalized, avg, deviation


def preprocess_rows(
    rows: List[Dict[str, float]],
    anomaly_threshold: float = 0.3,
) -> Tuple[List[List[float]], List[float], Dict[str, float]]:
    """Turn raw rows into feature matrix and labels.

    Returns:
        features: list of rows shape (n_samples, 4)
        labels: anomaly labels (1 if noisy/lossy)
        stats: normalization statistics
    """

    noise = [row["noise_level"] for row in rows]
    loss = [row["channel_loss"] for row in rows]
    basis = [row["basis"] for row in rows]
    detector = [row["detector_click"] for row in rows]

    noise_norm, noise_mean, noise_std = _normalize(noise)
    loss_norm, loss_mean, loss_std = _normalize(loss)

    features = [
        [noise_norm[i], loss_norm[i], basis[i], detector[i]]
        for i in range(len(rows))
    ]
    labels = [1.0 if (noise[i] + loss[i] > anomaly_threshold) else 0.0 for i in range(len(rows))]

    stats = {
        "noise_mean": noise_mean,
        "noise_std": noise_std,
        "loss_mean": loss_mean,
        "loss_std": loss_std,
    }
    return features, labels, stats


def split_data(
    features: List[List[float]],
    labels: List[float],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 1337,
) -> Dict[str, List[List[float]]]:
    """Split the dataset into train/val/test partitions."""

    rng = _rng(seed)
    indices = list(range(len(features)))
    rng.shuffle(indices)

    train_end = int(len(indices) * train_ratio)
    val_end = train_end + int(len(indices) * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    def _select(idxs: List[int]) -> List[List[float]]:
        return [features[i] for i in idxs]

    def _select_labels(idxs: List[int]) -> List[float]:
        return [labels[i] for i in idxs]

    return {
        "train_features": _select(train_idx),
        "train_labels": _select_labels(train_idx),
        "val_features": _select(val_idx),
        "val_labels": _select_labels(val_idx),
        "test_features": _select(test_idx),
        "test_labels": _select_labels(test_idx),
    }


def _save_matrix(path: Path, matrix: List[List[float]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(matrix)


def _save_vector(path: Path, vector: List[float]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows([[value] for value in vector])


def save_dataset(dataset: Dict[str, List[List[float]]], path: str | Path) -> None:
    """Save dataset as CSV files in a directory."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    _save_matrix(path / "train_features.csv", dataset["train_features"])
    _save_vector(path / "train_labels.csv", dataset["train_labels"])
    _save_matrix(path / "val_features.csv", dataset["val_features"])
    _save_vector(path / "val_labels.csv", dataset["val_labels"])
    _save_matrix(path / "test_features.csv", dataset["test_features"])
    _save_vector(path / "test_labels.csv", dataset["test_labels"])


def _load_matrix(path: Path) -> List[List[float]]:
    with path.open() as handle:
        reader = csv.reader(handle)
        return [[float(value) for value in row] for row in reader]


def _load_vector(path: Path) -> List[float]:
    with path.open() as handle:
        reader = csv.reader(handle)
        return [float(row[0]) for row in reader]


def load_dataset(path: str | Path) -> Dict[str, List[List[float]]]:
    path = Path(path)
    return {
        "train_features": _load_matrix(path / "train_features.csv"),
        "train_labels": _load_vector(path / "train_labels.csv"),
        "val_features": _load_matrix(path / "val_features.csv"),
        "val_labels": _load_vector(path / "val_labels.csv"),
        "test_features": _load_matrix(path / "test_features.csv"),
        "test_labels": _load_vector(path / "test_labels.csv"),
    }


def describe_config(config: DataConfig) -> Dict[str, float]:
    """Return a JSON-serializable description of the data configuration."""

    return asdict(config)
