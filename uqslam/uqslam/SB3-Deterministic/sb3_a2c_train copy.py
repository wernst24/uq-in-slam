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
from stable_baselines3.common.policies import ActorCriticCnnPolicy
import torch
import torch.nn as nn
from uqslam.uqslam.Environments.uq_nav_slow_env import UQNavSlowEnv


class CustomCNNPolicy(ActorCriticCnnPolicy):
    """
    Custom CNN policy that matches Dr. Dera's architecture as closely as possible
    """
    def __init__(self, *args, **kwargs):
        super(CustomCNNPolicy, self).__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Override to match TensorFlow implementation architecture.
        """
        # CNN feature extractor (similar to TF implementation)
        self.cnn = nn.Sequential(
            # Convert RGB to grayscale simulation
            nn.Conv2d(3, 1, kernel_size=1, bias=False),  # RGB to grayscale
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 16x16x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 8x8x64
            nn.ReLU(), 
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # 4x4x64
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            sample_input = torch.randn(1, 3, 32, 32)
            cnn_output_dim = self.cnn(sample_input).shape[1]
        
        # Shared dense layer (512 neurons like TF implementation)
        self.shared_net = nn.Sequential(
            nn.Linear(cnn_output_dim, 512),
            nn.ReLU()
        )
        
        # Policy network (actor)
        self.policy_net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_space.n)
        )
        
        # Value network (critic)
        self.value_net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs, deterministic=False):
        """
        Forward pass through the network
        """
        features = self.cnn(obs)
        shared_features = self.shared_net(features)
        
        # Get action logits and value
        action_logits = self.policy_net(shared_features)
        value = self.value_net(shared_features)
        
        # Sample action
        if deterministic:
            action = torch.argmax(action_logits, dim=1)
        else:
            # Apply temperature scaling for more deterministic behavior
            temperature = 0.1
            scaled_logits = action_logits / temperature
            dist = torch.distributions.Categorical(logits=scaled_logits)
            action = dist.sample()
        
        return action, value, action_logits


class WandbCallback(BaseCallback):
    """
    Custom callback for logging to Weights & Biases
    """
    def __init__(self, check_freq=1, verbose=1):
        super(WandbCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Log training metrics
            if len(self.model.ep_info_buffer) > 0:
                ep_info = self.model.ep_info_buffer[-1]
                
                wandb.log({
                    'sb3_episode_reward': ep_info['r'],
                    'sb3_episode_length': ep_info['l'],
                    'sb3_total_timesteps': self.num_timesteps,
                    'sb3_fps': int(self.num_timesteps / (time.time() - self.training_start_time))
                })
                
                # Log policy loss and value loss if available
                if hasattr(self.model, '_last_episode_starts'):
                    wandb.log({
                        'sb3_step': self.num_timesteps
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
    
    # Training parameters (matching TF implementation)
    repeat_steps = 2
    learning_rate = 1e-3
    total_timesteps = 1000000  # Equivalent to 10k episodes * 100 steps avg
    n_steps = 32  # Batch size equivalent
    gamma = 0.99
    ent_coef = 1e-5  # Low entropy for deterministic behavior
    vf_coef = 0.5
    
    # Create vectorized environment
    env = DummyVecEnv([make_env(repeat_steps)])
    env = VecMonitor(env)
    
    # Initialize wandb
    wandb.init(
        project=f"SB3_A2C_Comparison_{total_timesteps}_lr_{learning_rate}_repeat_{repeat_steps}",
        config={
            "algorithm": "SB3_A2C",
            "total_timesteps": total_timesteps,
            "learning_rate": learning_rate,
            "repeat_steps": repeat_steps,
            "n_steps": n_steps,
            "gamma": gamma,
            "entropy_coeff": ent_coef,
            "value_coeff": vf_coef,
            "policy_type": "CustomCNN"
        },
        tags=["sb3", "a2c", "deterministic", "comparison"]
    )
    
    # Create A2C model with custom policy
    model = A2C(
        policy="CnnPolicy",  # Use built-in CNN policy for simplicity
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=0.5,  # Gradient clipping like TF implementation
        use_rms_prop=False,  # Use Adam optimizer
        policy_kwargs={
            "features_extractor_kwargs": {
                "features_dim": 512  # Match TF implementation
            },
            "net_arch": [256, 128]  # Match TF dense layers
        },
        verbose=1,
        device="auto"
    )
    
    # Create callback for logging
    callback = WandbCallback(check_freq=100)
    
    node.get_logger().info("Starting SB3 A2C training...")
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
    model_path = f"/tmp/sb3_a2c_model_{total_timesteps}_steps.zip"
    model.save(model_path)
    wandb.save(model_path)
    
    # Log final metrics
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
    
    obs = env.reset()
    for episode in range(eval_episodes):
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            steps += 1
            
            if done[0]:
                eval_rewards.append(episode_reward)
                obs = env.reset()
                break
    
    # Log evaluation results
    avg_eval_reward = np.mean(eval_rewards)
    std_eval_reward = np.std(eval_rewards)
    
    wandb.log({
        'sb3_eval_mean_reward': avg_eval_reward,
        'sb3_eval_std_reward': std_eval_reward,
        'sb3_eval_episodes': eval_episodes
    })
    
    node.get_logger().info(f"Evaluation complete: {avg_eval_reward:.2f} ± {std_eval_reward:.2f}")
    
    # Compare with custom TF implementation
    node.get_logger().info("=" * 50)
    node.get_logger().info("COMPARISON SUMMARY:")
    node.get_logger().info(f"SB3 A2C - Average reward: {avg_eval_reward:.2f}")
    node.get_logger().info(f"SB3 A2C - Training time: {training_time:.2f}s")
    node.get_logger().info("Check Wandb for detailed comparison with TF implementation")
    node.get_logger().info("=" * 50)
    
    wandb.finish()
    env.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
