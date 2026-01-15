import tempfile
import unittest
from pathlib import Path

from ai_qkd.pipeline import run_pipeline


class TestPipeline(unittest.TestCase):
    def test_pipeline_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = run_pipeline(base_dir=Path(tmp_dir))
            self.assertGreaterEqual(results.metrics["anomaly_detection"]["accuracy"], 0.7)
            self.assertLessEqual(results.metrics["error_correction"]["bit_error_rate"], 0.35)
            self.assertGreaterEqual(results.metrics["key_distribution"]["mean_reward"], 0.55)
            self.assertTrue((Path(tmp_dir) / "results" / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
