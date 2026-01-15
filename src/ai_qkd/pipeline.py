"""Pipeline orchestration for the AI-Enhanced QKD workflow."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

from . import __version__
from .anomaly import LogisticModel, TrainingReport, train_logistic_regression, save_model
from .config import PipelineConfig, load_config
from .data import (
    DataConfig,
    describe_config,
    generate_raw_sessions,
    preprocess_rows,
    read_raw_csv,
    split_data,
    write_raw_csv,
    save_dataset,
)
from .error_correction import (
    ErrorCorrectionModel,
    ErrorCorrectionReport,
    tune_error_correction,
    save_model as save_error_model,
)
from .evaluation import evaluate_pipeline, save_metrics
from .rl import BanditPolicy, BanditReport, train_bandit_policy, save_policy


@dataclass
class PipelineArtifacts:
    raw_path: Path
    processed_dir: Path
    anomaly_model_path: Path
    error_model_path: Path
    rl_policy_path: Path
    metrics_path: Path


@dataclass
class PipelineResults:
    anomaly_report: TrainingReport
    error_reports: list[ErrorCorrectionReport]
    rl_report: BanditReport
    metrics: Dict[str, Any]


def _default_artifacts(base_dir: Path) -> PipelineArtifacts:
    return PipelineArtifacts(
        raw_path=base_dir / "data" / "raw" / "quantum_data.csv",
        processed_dir=base_dir / "data" / "processed",
        anomaly_model_path=base_dir / "models" / "anomaly" / "logistic_model.json",
        error_model_path=base_dir / "models" / "error_correction" / "window_model.json",
        rl_policy_path=base_dir / "models" / "key_distribution" / "bandit_policy.json",
        metrics_path=base_dir / "results" / "metrics.json",
    )


def generate_data(config: DataConfig, artifacts: PipelineArtifacts) -> None:
    rows = generate_raw_sessions(config)
    write_raw_csv(rows, artifacts.raw_path)


def preprocess_data(
    artifacts: PipelineArtifacts,
    seed: int = 1337,
    anomaly_threshold: float = 0.3,
) -> Dict[str, list]:
    rows = read_raw_csv(artifacts.raw_path)
    features, labels, _ = preprocess_rows(rows, anomaly_threshold=anomaly_threshold)
    dataset = split_data(features, labels, seed=seed)
    save_dataset(dataset, artifacts.processed_dir)
    return dataset


def train_anomaly_detector(
    dataset: Dict[str, list],
    config: PipelineConfig,
    artifacts: PipelineArtifacts,
) -> tuple[LogisticModel, TrainingReport]:
    model, report = train_logistic_regression(
        dataset["train_features"],
        dataset["train_labels"],
        learning_rate=config.anomaly.learning_rate,
        epochs=config.anomaly.epochs,
    )
    save_model(model, artifacts.anomaly_model_path)
    return model, report


def _sequence_from_rows(config: DataConfig) -> tuple[list[list[float]], list[list[float]]]:
    rng = random.Random(config.seed)
    sequences = []
    clean_sequences = []
    for _ in range(config.sessions):
        bits = [float(rng.randint(0, 1)) for _ in range(config.sequence_length)]
        noise = [rng.random() < (config.noise_mean * 1.5) for _ in range(config.sequence_length)]
        noisy_bits = [float(bit != is_noisy) for bit, is_noisy in zip(bits, noise)]
        sequences.append(noisy_bits)
        clean_sequences.append(bits)
    return sequences, clean_sequences


def train_error_correction(
    config: PipelineConfig,
    artifacts: PipelineArtifacts,
) -> tuple[ErrorCorrectionModel, list[ErrorCorrectionReport]]:
    noisy_sequences, clean_sequences = _sequence_from_rows(config.data)
    model, reports = tune_error_correction(
        noisy_sequences,
        clean_sequences,
        window_sizes=config.error_correction.window_sizes,
    )
    save_error_model(model, artifacts.error_model_path)
    return model, reports


def train_rl_policy(config: PipelineConfig, artifacts: PipelineArtifacts) -> tuple[BanditPolicy, BanditReport]:
    policy, report = train_bandit_policy(
        actions=config.rl.actions,
        episodes=config.rl.episodes,
        epsilon=config.rl.epsilon,
        alpha=config.rl.alpha,
        seed=config.data.seed,
    )
    save_policy(policy, artifacts.rl_policy_path)
    return policy, report


def evaluate(
    dataset: Dict[str, list],
    anomaly_model: LogisticModel,
    error_model: ErrorCorrectionModel,
    rl_policy: BanditPolicy,
    config: PipelineConfig,
    artifacts: PipelineArtifacts,
) -> Dict[str, Any]:
    noisy_sequences, clean_sequences = _sequence_from_rows(config.data)
    metrics = evaluate_pipeline(
        anomaly_model,
        dataset["test_features"],
        dataset["test_labels"],
        error_model,
        noisy_sequences,
        clean_sequences,
        rl_policy,
    )
    save_metrics(metrics, artifacts.metrics_path)
    return metrics


def run_pipeline(config_path: str | Path | None = None, base_dir: str | Path | None = None) -> PipelineResults:
    base_dir = Path(base_dir or Path.cwd())
    artifacts = _default_artifacts(base_dir)
    config = load_config(config_path) if config_path else PipelineConfig()

    generate_data(config.data, artifacts)
    dataset = preprocess_data(
        artifacts,
        seed=config.data.seed,
        anomaly_threshold=config.anomaly.anomaly_threshold,
    )
    anomaly_model, anomaly_report = train_anomaly_detector(dataset, config, artifacts)
    error_model, error_reports = train_error_correction(config, artifacts)
    rl_policy, rl_report = train_rl_policy(config, artifacts)
    metrics = evaluate(dataset, anomaly_model, error_model, rl_policy, config, artifacts)

    return PipelineResults(
        anomaly_report=anomaly_report,
        error_reports=error_reports,
        rl_report=rl_report,
        metrics=metrics,
    )


def pipeline_metadata(config: PipelineConfig) -> Dict[str, Any]:
    return {
        "version": __version__,
        "data": describe_config(config.data),
        "anomaly": config.anomaly.__dict__,
        "error_correction": {"window_sizes": list(config.error_correction.window_sizes)},
        "rl": config.rl.__dict__,
    }
