"""Simple bandit-style policy for key distribution."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .math_utils import clamp


@dataclass
class BanditPolicy:
    action_values: List[float]

    def act(self, epsilon: float, rng: random.Random) -> int:
        if rng.random() < epsilon:
            return rng.randint(0, len(self.action_values) - 1)
        return int(max(range(len(self.action_values)), key=lambda idx: self.action_values[idx]))


@dataclass
class BanditReport:
    rewards: List[float]
    action_counts: List[int]


def simulate_environment(action: int, rng: random.Random) -> float:
    """Return reward for selecting an action.

    Action 0/1 are more reliable, 2/3 are riskier.
    """

    base = 0.7 if action < 2 else 0.45
    noise = rng.gauss(0, 0.1)
    reward = clamp(base + noise, 0.0, 1.0)
    return float(reward)


def train_bandit_policy(
    actions: int,
    episodes: int,
    epsilon: float,
    alpha: float,
    seed: int = 1337,
) -> Tuple[BanditPolicy, BanditReport]:
    rng = random.Random(seed)
    values = [0.0 for _ in range(actions)]
    counts = [0 for _ in range(actions)]
    rewards: List[float] = []

    for _ in range(episodes):
        action = rng.randint(0, actions - 1) if rng.random() < epsilon else int(max(range(actions), key=lambda idx: values[idx]))
        reward = simulate_environment(action, rng)
        counts[action] += 1
        values[action] = values[action] + alpha * (reward - values[action])
        rewards.append(reward)

    policy = BanditPolicy(action_values=values)
    return policy, BanditReport(rewards=rewards, action_counts=counts)


def save_policy(policy: BanditPolicy, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "action_values": policy.action_values,
    }
    path.write_text(json.dumps(payload, indent=2))


def load_policy(path: str | Path) -> BanditPolicy:
    payload = json.loads(Path(path).read_text())
    return BanditPolicy(action_values=list(payload["action_values"]))


def evaluate_policy(policy: BanditPolicy, episodes: int = 100, seed: int = 2024) -> Dict[str, float]:
    rng = random.Random(seed)
    rewards = [simulate_environment(int(max(range(len(policy.action_values)), key=lambda idx: policy.action_values[idx])), rng) for _ in range(episodes)]
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    return {
        "mean_reward": float(mean_reward),
        "min_reward": float(min(rewards)) if rewards else 0.0,
        "max_reward": float(max(rewards)) if rewards else 0.0,
    }
