"""Small math helpers to avoid external dependencies."""

from __future__ import annotations

import math
from typing import Iterable, List


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def dot(a: Iterable[float], b: Iterable[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def vector_add(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def vector_scale(a: List[float], scale: float) -> List[float]:
    return [x * scale for x in a]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def variance(values: Iterable[float], mean_value: float | None = None) -> float:
    values = list(values)
    if not values:
        return 0.0
    mean_value = mean_value if mean_value is not None else mean(values)
    return sum((x - mean_value) ** 2 for x in values) / len(values)


def stddev(values: Iterable[float], mean_value: float | None = None) -> float:
    return math.sqrt(variance(values, mean_value=mean_value))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
