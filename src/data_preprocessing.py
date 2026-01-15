"""Legacy wrapper for data preprocessing."""

from pathlib import Path

from ai_qkd.pipeline import PipelineConfig, _default_artifacts, generate_data, preprocess_data


def main() -> None:
    config = PipelineConfig()
    artifacts = _default_artifacts(Path.cwd())
    generate_data(config.data, artifacts)
    preprocess_data(
        artifacts,
        seed=config.data.seed,
        anomaly_threshold=config.anomaly.anomaly_threshold,
    )
    print(f"Raw data stored at {artifacts.raw_path}")
    print(f"Processed data stored at {artifacts.processed_dir}")


if __name__ == "__main__":
    main()
