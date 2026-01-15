import random
import unittest

from ai_qkd.error_correction import tune_error_correction, evaluate


class TestErrorCorrection(unittest.TestCase):
    def test_error_correction_window_selection(self) -> None:
        rng = random.Random(7)
        clean = [[float(rng.randint(0, 1)) for _ in range(16)] for _ in range(20)]
        noisy = []
        for row in clean:
            noisy_row = []
            for bit in row:
                flip = rng.random() < 0.1
                noisy_row.append(float(bit != flip))
            noisy.append(noisy_row)

        model, reports = tune_error_correction(noisy, clean, window_sizes=[1, 3, 5])
        metrics = evaluate(model, noisy, clean)

        self.assertEqual(len(reports), 3)
        self.assertLessEqual(metrics["bit_error_rate"], 0.3)


if __name__ == "__main__":
    unittest.main()
