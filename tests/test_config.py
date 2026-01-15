import tempfile
import unittest
from pathlib import Path

from ai_qkd.config import load_config, save_default_config, PipelineConfig


class TestConfig(unittest.TestCase):
    def test_save_and_load_config(self) -> None:
        with self.subTest("save and load"):
            with tempfile.TemporaryDirectory() as tmp_dir:
                config_path = Path(tmp_dir) / "config.json"
                save_default_config(config_path)

                config = load_config(config_path)
                self.assertIsInstance(config, PipelineConfig)
                self.assertGreater(config.data.sessions, 0)
                self.assertGreater(config.anomaly.epochs, 0)


if __name__ == "__main__":
    unittest.main()
