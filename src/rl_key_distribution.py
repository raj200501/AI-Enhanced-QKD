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

def train_rl_model(env, config):
    model = PPO("MlpPolicy", env, verbose=1, **config['ppo_params'])
    model.learn(total_timesteps=config['total_timesteps'])
    model.save('models/rl_key_distribution/model')
    return model

if __name__ == "__main__":
    # Load configuration
    import json
    with open('config/rl_config.json', 'r') as f:
        config = json.load(f)
    # Create environment
    env = QKDEnv(config)
    # Train the RL model
    model = train_rl_model(env, config)
