"""Legacy wrapper for error correction training."""

from pathlib import Path

from ai_qkd.pipeline import PipelineConfig, _default_artifacts, train_error_correction


def main() -> None:
    config = PipelineConfig()
    artifacts = _default_artifacts(Path.cwd())
    model, reports = train_error_correction(config, artifacts)
    best = reports[0]
    for report in reports:
        if report.bit_error_rate < best.bit_error_rate:
            best = report
    print(f"Saved error correction model to {artifacts.error_model_path}")
    print(f"Best window size: {best.window_size}, BER: {best.bit_error_rate:.3f}")


if __name__ == "__main__":
    main()
