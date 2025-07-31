#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import os
import time
import numpy as np
import wandb
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
from uqslam.uqslam.Environments.uq_nav_slow_env import UQNavSlowEnv


class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for 32x32 images
    """
    def __init__(self, observation_space, features_dim=512):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        
        # CNN layers designed for 32x32 images
        # SB3 expects channel-first format: (batch, channels, height, width)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # (3, 32, 32) -> (32, 16, 16)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (32, 16, 16) -> (64, 8, 8)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # (64, 8, 8) -> (64, 4, 4)
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate the output dimension with proper channel-first format
        with torch.no_grad():
            sample_input = torch.randn(1, 3, 32, 32)  # Channel-first format
            output_dim = self.cnn(sample_input).shape[1]
        
        # Linear layer to get desired features_dim
        self.linear = nn.Sequential(
            nn.Linear(output_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # SB3 automatically handles the channel conversion, so observations should already be in correct format
        # Input should be (batch, channels, height, width) format
        return self.linear(self.cnn(observations))


class WandbCallback(BaseCallback):
    """
    Custom callback for logging to Weights & Biases
    """
    def __init__(self, check_freq=100, verbose=1):
        super(WandbCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Log training metrics
            if len(self.model.ep_info_buffer) > 0:
                ep_info = self.model.ep_info_buffer[-1]
                self.episode_count += 1
                
                wandb.log({
                    'sb3_episode_reward': ep_info['r'],
                    'sb3_episode_length': ep_info['l'],
                    'sb3_episode_count': self.episode_count,
                    'sb3_total_timesteps': self.num_timesteps,
                    'sb3_fps': int(self.num_timesteps / (time.time() - self.training_start_time)) if hasattr(self, 'training_start_time') else 0
                })
        
        return True
    
    def _on_training_start(self) -> None:
        self.training_start_time = time.time()


class TrainingNode(Node):
    def __init__(self):
        super().__init__("sb3_a2c_training", allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)
        self._training_mode = "training"


def make_env(repeat_steps=2):
    """
    Create environment for SB3
    """
    def _init():
        env = UQNavSlowEnv(repeat_steps=repeat_steps)
        return env
    return _init


def main():
    rclpy.init()
    node = TrainingNode()
    node.get_logger().info("SB3 A2C Training node has been created")
    
    # Set API key
    os.environ["WANDB_API_KEY"] = "084538caaa5ded119d5b9e04a7b699ac8de907fe"
    
    # Training parameters (matching TF implementation but using SB3 defaults)
    repeat_steps = 2
    learning_rate = 1e-3
    total_timesteps = 320000  # ~10k episodes * 32 steps per episode
    n_steps = 32  # Batch size equivalent to TF implementation
    gamma = 0.99
    ent_coef = 0.01  # SB3 default entropy coefficient (higher than our custom)
    vf_coef = 0.5
    
    # Create vectorized environment
    env = DummyVecEnv([make_env(repeat_steps)])
    env = VecMonitor(env)
    
    # Initialize wandb
    wandb.init(
        project=f"SB3_Default_A2C_Comparison_{total_timesteps}_lr_{learning_rate}_repeat_{repeat_steps}",
        config={
            "algorithm": "SB3_A2C_Default",
            "total_timesteps": total_timesteps,
            "learning_rate": learning_rate,
            "repeat_steps": repeat_steps,
            "n_steps": n_steps,
            "gamma": gamma,
            "entropy_coeff": ent_coef,
            "value_coeff": vf_coef,
            "policy_type": "CnnPolicy_Default"
        },
        tags=["sb3", "a2c", "default", "comparison"]
    )
    
    # Create A2C model with custom CNN policy
    model = A2C(
        policy="CnnPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=0.5,
        use_rms_prop=False,
        verbose=1,
        device="auto",
        policy_kwargs={
            "features_extractor_class": CustomCNN,
            "features_extractor_kwargs": {"features_dim": 512},
            "normalize_images": False
        }
    )
    
    # Create callback for logging
    callback = WandbCallback(check_freq=50)  # Log more frequently
    
    node.get_logger().info("Starting SB3 Default A2C training...")
    start_time = time.time()
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Save the model
    model_path = f"/tmp/sb3_default_a2c_model_{total_timesteps}_steps.zip"
    model.save(model_path)
    wandb.save(model_path)
    
    # Log final training metrics
    wandb.log({
        'sb3_total_training_time': training_time,
        'sb3_final_timesteps': total_timesteps
    })
    
    node.get_logger().info(f"Training completed in {training_time:.2f} seconds")
    node.get_logger().info(f"Model saved to {model_path}")
    
    # Evaluation phase
    node.get_logger().info("Starting evaluation...")
    eval_episodes = 10
    eval_rewards = []
    eval_lengths = []
    
    for episode in range(eval_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done and episode_length < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            episode_length += 1
            
            if done[0]:
                eval_rewards.append(episode_reward)
                eval_lengths.append(episode_length)
                break
    
    # Log evaluation results
    avg_eval_reward = np.mean(eval_rewards)
    std_eval_reward = np.std(eval_rewards)
    avg_eval_length = np.mean(eval_lengths)
    
    wandb.log({
        'sb3_eval_mean_reward': avg_eval_reward,
        'sb3_eval_std_reward': std_eval_reward,
        'sb3_eval_mean_length': avg_eval_length,
        'sb3_eval_episodes': eval_episodes
    })
    
    node.get_logger().info(f"Evaluation Results:")
    node.get_logger().info(f"  Average reward: {avg_eval_reward:.2f} ± {std_eval_reward:.2f}")
    node.get_logger().info(f"  Average length: {avg_eval_length:.2f}")
    node.get_logger().info(f"  Training time: {training_time:.2f}s")
    
    # Log summary to wandb
    wandb.log({
        'final_summary_reward': avg_eval_reward,
        'final_summary_training_time': training_time,
        'final_summary_timesteps': total_timesteps
    })
    
    wandb.finish()
    env.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
