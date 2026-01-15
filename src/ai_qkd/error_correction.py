"""Error-correction utilities for QKD bit sequences."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass
class ErrorCorrectionModel:
    window_size: int
    threshold: float = 0.5

    def correct(self, sequences: List[List[float]]) -> List[List[float]]:
        """Apply a rolling-window majority filter to binary sequences."""

        window = self.window_size
        corrected: List[List[float]] = []
        for sequence in sequences:
            padded = [sequence[0]] * (window - 1) + sequence
            row: List[float] = []
            for idx in range(len(sequence)):
                segment = padded[idx : idx + window]
                value = 1.0 if (sum(segment) / window) >= self.threshold else 0.0
                row.append(value)
            corrected.append(row)
        return corrected


@dataclass
class ErrorCorrectionReport:
    window_size: int
    bit_error_rate: float


def _bit_error_rate(predicted: List[List[float]], target: List[List[float]]) -> float:
    total = 0
    errors = 0
    for pred_row, target_row in zip(predicted, target):
        for pred, actual in zip(pred_row, target_row):
            total += 1
            if pred != actual:
                errors += 1
    return errors / total if total else 0.0


def tune_error_correction(
    noisy_sequences: List[List[float]],
    target_sequences: List[List[float]],
    window_sizes: Iterable[int],
) -> Tuple[ErrorCorrectionModel, List[ErrorCorrectionReport]]:
    """Select the best window size for error correction."""

    reports: List[ErrorCorrectionReport] = []
    best_model = None
    best_rate = float("inf")
    for window in window_sizes:
        model = ErrorCorrectionModel(window_size=window)
        corrected = model.correct(noisy_sequences)
        rate = _bit_error_rate(corrected, target_sequences)
        reports.append(ErrorCorrectionReport(window_size=window, bit_error_rate=rate))
        if rate < best_rate:
            best_rate = rate
            best_model = model
    if best_model is None:
        raise ValueError("No window sizes provided")
    return best_model, reports


def save_model(model: ErrorCorrectionModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "window_size": model.window_size,
        "threshold": model.threshold,
    }
    path.write_text(json.dumps(payload, indent=2))


def load_model(path: str | Path) -> ErrorCorrectionModel:
    payload = json.loads(Path(path).read_text())
    return ErrorCorrectionModel(window_size=int(payload["window_size"]), threshold=float(payload["threshold"]))


def evaluate(model: ErrorCorrectionModel, noisy_sequences: List[List[float]], target_sequences: List[List[float]]) -> Dict[str, float]:
    corrected = model.correct(noisy_sequences)
    return {
        "bit_error_rate": _bit_error_rate(corrected, target_sequences),
        "sequence_length": float(len(noisy_sequences[0])) if noisy_sequences else 0.0,
    }
