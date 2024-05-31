# Train_RL_Key_Distribution.ipynb

import gym
import numpy as np
from stable_baselines3 import PPO

class QKDEnv(gym.Env):
    def __init__(self, config):
        super(QKDEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(config['observation_size'],), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(config['action_size'])
        self.state = None

    def reset(self):
        self.state = self._get_initial_state()
        return self.state

    def step(self, action):
        next_state, reward, done, info = self._take_action(action)
        return next_state, reward, done, info

    def _get_initial_state(self):
        return np.random.rand(self.observation_space.shape[0])

    def _take_action(self, action):
        next_state = np.random.rand(self.observation_space.shape[0])
        reward = np.random.rand()
        done = np.random.rand() > 0.95
        info = {}
        return next_state, reward, done, info

# Load configuration
config = {
    "observation_size": 10,
    "action_size": 4,
    "total_timesteps": 10000,
    "ppo_params": {
        "learning_rate": 0.0003,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10
    }
}

# Create environment
env = QKDEnv(config)

# Train the RL model
model = PPO("MlpPolicy", env, verbose=1, **config['ppo_params'])
model.learn(total_timesteps=config['total_timesteps'])
model.save('../models/rl_key_distribution/model')
