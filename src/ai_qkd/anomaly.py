"""Lightweight anomaly detection using logistic regression."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

from .math_utils import sigmoid


@dataclass
class LogisticModel:
    weights: List[float]
    bias: float

    def predict_proba(self, features: List[List[float]]) -> List[float]:
        return [sigmoid(sum(w * x for w, x in zip(self.weights, row)) + self.bias) for row in features]

    def predict(self, features: List[List[float]], threshold: float = 0.5) -> List[float]:
        return [1.0 if value >= threshold else 0.0 for value in self.predict_proba(features)]


@dataclass
class TrainingReport:
    losses: list
    accuracy: float


def train_logistic_regression(
    features: List[List[float]],
    labels: List[float],
    learning_rate: float,
    epochs: int,
) -> Tuple[LogisticModel, TrainingReport]:
    """Train a logistic regression classifier with gradient descent."""

    rng = __import__("random").Random(42)
    weights = [rng.uniform(-0.1, 0.1) for _ in range(len(features[0]))]
    bias = 0.0
    losses = []

    for _ in range(epochs):
        grad_w = [0.0 for _ in weights]
        grad_b = 0.0
        loss_sum = 0.0
        for row, label in zip(features, labels):
            logit = sum(w * x for w, x in zip(weights, row)) + bias
            prob = sigmoid(logit)
            error = prob - label
            for i in range(len(weights)):
                grad_w[i] += error * row[i]
            grad_b += error
            loss_sum += -(
                label * (0.0 if prob == 0 else math.log(prob))
                + (1 - label) * (0.0 if prob == 1 else math.log(1 - prob))
            )

        count = float(len(features))
        grad_w = [value / count for value in grad_w]
        grad_b /= count
        weights = [w - learning_rate * dw for w, dw in zip(weights, grad_w)]
        bias -= learning_rate * grad_b
        losses.append(loss_sum / count)

    model = LogisticModel(weights=weights, bias=bias)
    predictions = model.predict(features)
    accuracy = sum(int(p == y) for p, y in zip(predictions, labels)) / len(labels)
    return model, TrainingReport(losses=losses, accuracy=accuracy)


def save_model(model: LogisticModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "weights": model.weights,
        "bias": model.bias,
    }
    path.write_text(json.dumps(payload, indent=2))


def load_model(path: str | Path) -> LogisticModel:
    payload = json.loads(Path(path).read_text())
    return LogisticModel(weights=list(payload["weights"]), bias=float(payload["bias"]))


def evaluate(model: LogisticModel, features: List[List[float]], labels: List[float]) -> Dict[str, float]:
    predictions = model.predict(features)
    accuracy = sum(int(p == y) for p, y in zip(predictions, labels)) / len(labels)
    true_positive = sum(1 for p, y in zip(predictions, labels) if p == 1 and y == 1)
    predicted_positive = sum(1 for p in predictions if p == 1)
    actual_positive = sum(1 for y in labels if y == 1)
    precision = true_positive / (predicted_positive + 1e-8)
    recall = true_positive / (actual_positive + 1e-8)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
    }
