"""
Gymnasium wrapper to transpose observation channels for PyTorch compatibility.
Converts from (H, W, C) format to (C, H, W) format expected by PyTorch/SB3.
"""
import gymnasium as gym
import numpy as np


class TransposeObsWrapper(gym.ObservationWrapper):
    """
    Transpose observation from (H, W, C) to (C, H, W) for PyTorch
    """
    def __init__(self, env):
        super().__init__(env)
        # Update observation space to reflect the transposed shape
        old_shape = env.observation_space.shape
        new_shape = (old_shape[2], old_shape[0], old_shape[1])  # (H, W, C) -> (C, H, W)
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.min(),
            high=env.observation_space.high.max(),
            shape=new_shape,
            dtype=env.observation_space.dtype
        )
    
    def observation(self, obs):
        # Transpose from (H, W, C) to (C, H, W)
        return np.transpose(obs, (2, 0, 1))
