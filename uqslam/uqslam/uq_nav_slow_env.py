"""
Slow control frequency wrapper for UQNavBotEnv.
Repeats each action multiple times to achieve slower control frequency.
"""
import gymnasium as gym
import numpy as np
from uqslam.uq_nav_env import UQNavBotEnv


class UQNavSlowEnv(gym.Wrapper):
    """
    Wrapper that repeats each action multiple times for slower control frequency.
    """
    
    def __init__(self, instance_num=0, repeat_steps=4):
        # Create the base environment
        base_env = UQNavBotEnv(instance_num=instance_num)
        super().__init__(base_env)
        
        self.repeat_steps = repeat_steps
        self.env.get_logger().info(f"Slow control wrapper: repeating actions {repeat_steps} times")
    
    def step(self, action):
        """
        Repeat the action multiple times and return the final observation and cumulative reward.
        """
        total_reward = 0
        done = False
        truncated = False
        
        for i in range(self.repeat_steps):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            
            # If episode ends during any repeat, break early
            if done or truncated:
                break
        
        return obs, total_reward, done, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def close(self):
        return self.env.close()
