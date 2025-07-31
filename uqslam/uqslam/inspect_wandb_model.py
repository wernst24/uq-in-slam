#!/usr/bin/env python3

import wandb
import os
import numpy as np
import tensorflow as tf
from uqslam.uqslam.Environments.uq_nav_slow_env import UQNavSlowEnv
# Import your network model
from .Deterministic import Network_Model_Det_TRAIN as net

def download_and_load_model():
    """Download and load the trained model from W&B"""
    
    # Initialize W&B API
    api = wandb.Api()
    
    # Get the specific run
    run = api.run("wernst24-rochester-institute-of-technology/Train_Traditional_A2C_10000_lr_0.001_repeat_4/c0yvxwmi")
    
    # Download all files from the run
    files = run.files()
    
    # Find the latest model weights file
    weight_files = [f for f in files if f.name.endswith('.weights.h5')]
    if not weight_files:
        print("No weight files found!")
        return None
    
    # Get the latest weights (highest episode number)
    latest_weights = max(weight_files, key=lambda x: int(x.name.split('_ep')[1].split('_')[0]))
    print(f"Downloading: {latest_weights.name}")
    
    # Download the file
    latest_weights.download(replace=True)
    
    # Also download parameters if available
    param_files = [f for f in files if f.name.endswith('.json')]
    if param_files:
        latest_params = max(param_files, key=lambda x: int(x.name.split('_ep')[1].split('.')[0]))
        print(f"Downloading: {latest_params.name}")
        latest_params.download(replace=True)
    
    return latest_weights.name, latest_params.name if param_files else None

def load_and_test_model(weights_path, params_path=None):
    """Load the model and test it in the environment"""
    
    # Create the same model architecture
    model_A2C = net.Model_A2C(num_actions=3)
    agent = net.A2CAgent(model_A2C, learning_rate=1e-3)  # Learning rate doesn't matter for inference
    
    # Load the weights
    try:
        model_A2C.load_weights(weights_path)
        print(f"Successfully loaded weights from {weights_path}")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return
    
    # Create environment (same as training)
    env = UQNavSlowEnv(repeat_steps=4)
    
    # Test the model for several episodes
    num_test_episodes = 5
    test_rewards = []
    
    for episode in range(num_test_episodes):
        observation, info = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        print(f"\n--- Episode {episode + 1} ---")
        
        while not done and steps < 1000:  # Max 1000 steps
            # Get action from trained model (deterministic)
            action, value = agent.action_value(observation)
            print(f"Step {steps}: Action={action}, Value={value:.2f}")
            
            # Take step
            observation, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Print action names for clarity
            action_names = ["Forward", "Left", "Right"]
            if steps % 10 == 0:  # Print every 10 steps
                print(f"  Step {steps}: {action_names[int(action)]}, Reward: {reward:.2f}")
            
            if done or truncated:
                break
        
        test_rewards.append(total_reward)
        print(f"Episode {episode + 1} completed: {steps} steps, Total reward: {total_reward:.2f}")
    
    print(f"\n=== Test Results ===")
    print(f"Average reward: {np.mean(test_rewards):.2f}")
    print(f"Reward std: {np.std(test_rewards):.2f}")
    print(f"All rewards: {test_rewards}")
    
    env.close()
    return test_rewards

def analyze_action_distribution(weights_path, params_path=None):
    """Analyze what actions the model tends to take"""
    
    # Load model
    model_A2C = net.Model_A2C(num_actions=3)
    agent = net.A2CAgent(model_A2C, learning_rate=1e-3)
    model_A2C.load_weights(weights_path)
    
    # Create environment
    env = UQNavSlowEnv(repeat_steps=4)
    
    # Collect actions over multiple random starts
    action_counts = [0, 0, 0]  # Forward, Left, Right
    total_actions = 0
    
    for _ in range(10):  # 10 different starting positions
        observation, info = env.reset()
        
        for _ in range(50):  # 50 actions per start
            action, _ = agent.action_value(observation)
            action_counts[int(action)] += 1
            total_actions += 1
            
            observation, _, done, truncated, _ = env.step(action)
            if done or truncated:
                break
    
    # Print action distribution
    action_names = ["Forward", "Left", "Right"]
    print(f"\n=== Action Distribution Analysis ===")
    for i, (name, count) in enumerate(zip(action_names, action_counts)):
        percentage = (count / total_actions) * 100
        print(f"{name}: {count}/{total_actions} ({percentage:.1f}%)")
    
    env.close()
    return action_counts

def main():
    # Download model from W&B
    weights_file, params_file = download_and_load_model()
    
    if weights_file:
        # Test the model
        test_rewards = load_and_test_model(weights_file, params_file)
        
        # Analyze action distribution
        action_dist = analyze_action_distribution(weights_file, params_file)
        
        print(f"\nModel inspection complete!")
        print(f"As expected from training logs, model likely takes mostly Forward actions")

if __name__ == "__main__":
    main()