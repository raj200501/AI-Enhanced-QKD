"""Legacy wrapper for key distribution policy training."""

from pathlib import Path

from ai_qkd.pipeline import PipelineConfig, _default_artifacts, train_rl_policy


def main() -> None:
    config = PipelineConfig()
    artifacts = _default_artifacts(Path.cwd())
    policy, report = train_rl_policy(config, artifacts)
    print(f"Saved RL policy to {artifacts.rl_policy_path}")
    print(f"Episode reward sample: {report.rewards[:5]}")


if __name__ == "__main__":
    main()
