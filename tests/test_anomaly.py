import random
import unittest

from ai_qkd.anomaly import train_logistic_regression, evaluate


class TestAnomaly(unittest.TestCase):
    def test_logistic_regression_training(self) -> None:
        rng = random.Random(1)
        features = [[rng.uniform(-1, 1) for _ in range(4)] for _ in range(200)]
        weights = [0.5, -0.3, 0.2, 0.1]
        labels = [1.0 if sum(w * x for w, x in zip(weights, row)) > 0 else 0.0 for row in features]

        model, report = train_logistic_regression(features, labels, learning_rate=0.5, epochs=80)
        metrics = evaluate(model, features, labels)

        self.assertGreaterEqual(report.accuracy, 0.8)
        self.assertGreaterEqual(metrics["accuracy"], 0.8)
        self.assertGreaterEqual(metrics["precision"], 0.75)


if __name__ == "__main__":
    unittest.main()
