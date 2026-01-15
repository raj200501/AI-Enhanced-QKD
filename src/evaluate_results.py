"""Legacy wrapper for pipeline evaluation."""

import json
from pathlib import Path

from ai_qkd.pipeline import run_pipeline


def main() -> None:
    results = run_pipeline(base_dir=Path.cwd())
    print(json.dumps(results.metrics, indent=2))


if __name__ == "__main__":
    main()
