"""
Vectorized Gymnasium environment for a 4x4 grid of HospitalBots in Gazebo.
Each robot is assumed to be spawned with namespace 'hospitalbot_ns_{idx}' (idx: 0-15).
"""

import rclpy
import numpy as np
from gymnasium.vector import VectorEnv
from uqslam.p3at_fast_control_env import P3atFastControlEnv

class HospitalBotVecEnv(VectorEnv):
    def __init__(self, num_envs=16):
        # Each env is a HospitalBotEnv with a unique namespace
        self.num_envs = num_envs
        self.envs = []
        self._namespaces = [f"hospitalbot_ns_{i}" for i in range(num_envs)]
        for ns in self._namespaces:
            env = HospitalBotEnv(namespace=ns)
            self.envs.append(env)
        # Assume all envs have the same spaces
        observation_space = self.envs[0].observation_space
        action_space = self.envs[0].action_space
        super().__init__(num_envs, observation_space, action_space)

    def reset(self, seed=None, options=None):
        obs = []
        infos = []
        for env in self.envs:
            ob, info = env.reset(seed=seed, options=options)
            obs.append(ob)
            infos.append(info)
        return np.stack(obs), infos

    def step(self, actions):
        obs, rewards, dones, truncs, infos = [], [], [], [], []
        for i, env in enumerate(self.envs):
            ob, reward, done, trunc, info = env.step(actions[i])
            obs.append(ob)
            rewards.append(reward)
            dones.append(done)
            truncs.append(trunc)
            infos.append(info)
        return (
            np.stack(obs),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array(truncs, dtype=bool),
            infos,
        )

    def close(self):
        for env in self.envs:
            env.close()

# Example usage:
# vec_env = HospitalBotVecEnv(num_envs=16)
# obs, infos = vec_env.reset()
# actions = np.zeros(16, dtype=int)
# obs, rewards, dones, truncs, infos = vec_env.step(actions)