"""Legacy wrapper for anomaly detection training."""

from pathlib import Path

from ai_qkd.pipeline import PipelineConfig, _default_artifacts, preprocess_data, train_anomaly_detector


def main() -> None:
    config = PipelineConfig()
    artifacts = _default_artifacts(Path.cwd())
    dataset = preprocess_data(
        artifacts,
        seed=config.data.seed,
        anomaly_threshold=config.anomaly.anomaly_threshold,
    )
    model, report = train_anomaly_detector(dataset, config, artifacts)
    print(f"Saved anomaly model to {artifacts.anomaly_model_path}")
    print(f"Training accuracy: {report.accuracy:.3f}")


if __name__ == "__main__":
    main()
