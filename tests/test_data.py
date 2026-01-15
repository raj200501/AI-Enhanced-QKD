import tempfile
import unittest
from pathlib import Path

from ai_qkd.config import DataConfig
from ai_qkd.data import (
    generate_raw_sessions,
    preprocess_rows,
    split_data,
    write_raw_csv,
    read_raw_csv,
)


class TestData(unittest.TestCase):
    def test_generate_and_preprocess(self) -> None:
        config = DataConfig(seed=123, sessions=5, sequence_length=4)
        rows = generate_raw_sessions(config)
        self.assertEqual(len(rows), config.sessions * config.sequence_length)

        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path = Path(tmp_dir) / "raw.csv"
            write_raw_csv(rows, raw_path)
            loaded = read_raw_csv(raw_path)
            self.assertEqual(set(loaded[0].keys()), set(rows[0].keys()))

            features, labels, stats = preprocess_rows(loaded)
            self.assertEqual(len(features), len(labels))
            self.assertEqual(len(features[0]), 4)
            self.assertIn("noise_mean", stats)

            dataset = split_data(features, labels, seed=123)
            self.assertGreater(len(dataset["train_features"]), 0)
            self.assertEqual(len(dataset["test_features"][0]), 4)


if __name__ == "__main__":
    unittest.main()
