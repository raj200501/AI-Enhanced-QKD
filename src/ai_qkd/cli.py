"""Command-line interface for the AI-Enhanced QKD pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config, save_default_config
from .pipeline import (
    PipelineConfig,
    _default_artifacts,
    generate_data,
    preprocess_data,
    train_anomaly_detector,
    train_error_correction,
    train_rl_policy,
    evaluate,
    run_pipeline,
    pipeline_metadata,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI-Enhanced QKD pipeline CLI")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-config", help="Write a default configuration file")
    subparsers.add_parser("generate-data", help="Generate synthetic raw QKD data")
    subparsers.add_parser("preprocess", help="Preprocess data into train/val/test splits")
    subparsers.add_parser("train-anomaly", help="Train anomaly detector")
    subparsers.add_parser("train-error", help="Train error correction model")
    subparsers.add_parser("train-rl", help="Train key distribution policy")
    subparsers.add_parser("evaluate", help="Evaluate trained models")
    subparsers.add_parser("run", help="Run the full pipeline")
    subparsers.add_parser("metadata", help="Print pipeline metadata")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    base_dir = Path.cwd()
    artifacts = _default_artifacts(base_dir)

    if args.command == "init-config":
        target = Path(args.config or "config/pipeline.json")
        save_default_config(target)
        print(f"Wrote default config to {target}")
        return

    config = load_config(args.config) if args.config else PipelineConfig()

    if args.command == "generate-data":
        generate_data(config.data, artifacts)
        print(f"Generated raw data at {artifacts.raw_path}")
        return

    if args.command == "preprocess":
        preprocess_data(
            artifacts,
            seed=config.data.seed,
            anomaly_threshold=config.anomaly.anomaly_threshold,
        )
        print(f"Saved processed data to {artifacts.processed_dir}")
        return

    if args.command == "train-anomaly":
        dataset = preprocess_data(
            artifacts,
            seed=config.data.seed,
            anomaly_threshold=config.anomaly.anomaly_threshold,
        )
        train_anomaly_detector(dataset, config, artifacts)
        print(f"Saved anomaly model to {artifacts.anomaly_model_path}")
        return

    if args.command == "train-error":
        train_error_correction(config, artifacts)
        print(f"Saved error correction model to {artifacts.error_model_path}")
        return

    if args.command == "train-rl":
        train_rl_policy(config, artifacts)
        print(f"Saved RL policy to {artifacts.rl_policy_path}")
        return

    if args.command == "evaluate":
        dataset = preprocess_data(
            artifacts,
            seed=config.data.seed,
            anomaly_threshold=config.anomaly.anomaly_threshold,
        )
        anomaly_model, _ = train_anomaly_detector(dataset, config, artifacts)
        error_model, _ = train_error_correction(config, artifacts)
        rl_policy, _ = train_rl_policy(config, artifacts)
        metrics = evaluate(dataset, anomaly_model, error_model, rl_policy, config, artifacts)
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "run":
        results = run_pipeline(args.config, base_dir)
        print(json.dumps(results.metrics, indent=2))
        return

    if args.command == "metadata":
        print(json.dumps(pipeline_metadata(config), indent=2))
        return


if __name__ == "__main__":
    main()
