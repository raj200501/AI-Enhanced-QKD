import unittest

from ai_qkd.rl import train_bandit_policy, evaluate_policy


class TestRL(unittest.TestCase):
    def test_bandit_training(self) -> None:
        policy, report = train_bandit_policy(actions=4, episodes=120, epsilon=0.1, alpha=0.5, seed=21)
        self.assertEqual(len(report.rewards), 120)
        metrics = evaluate_policy(policy, episodes=50)
        self.assertGreater(metrics["mean_reward"], 0.5)


if __name__ == "__main__":
    unittest.main()
