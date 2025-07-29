from uqslam.p3at_fast_control_env import P3atFastControlEnv
from gymnasium import Env

class P3atSlowControlEnv(Env):
    """
    Wrapper Gymnasium environment that repeats each action for N steps
    to simulate a slower control frequency.
    """
    def __init__(self, prefix="robot1", robot_namespace="robot1_ns", repeat_steps=4):
        self.env = P3atFastControlEnv(prefix=prefix, robot_namespace=robot_namespace)
        self.repeat_steps = repeat_steps
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        total_reward = 0.0
        done = False
        truncated = False
        info = None
        for i in range(self.repeat_steps):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done or truncated:
                break
        return obs, total_reward, done, truncated, info

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()