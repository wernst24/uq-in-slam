'''Sanity Test Script'''

# TODO: Complete this draft and test Baseline CartPole 

from agents.a2c_baseline_cartpole import A2CBaselineCartPole
import gymnasium as gym
import numpy as np

# Step 1: Get environment info
env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

# Step 2: Build model
model = A2CBaselineCartPole(obs_dim, n_actions)

# Step 3: Pass dummy input to test forward pass
obs = env.reset()[0]
obs_tensor = tf.convert_to_tensor([obs], dtype=tf.float32)
policy_logits, value = model(obs_tensor)

# Print to verify
print("Policy logits:", policy_logits)
print("Value estimate:", value)