#!usr/bin/env python3

import rclpy
from rclpy.node import Node
from gymnasium.envs.registration import register
from test_new_pkg_name.hospitalbot_env import HospitalBotEnv
import gymnasium as gym
from stable_baselines3.dqn import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os
from stable_baselines3.common.monitor import Monitor


class TrainingNode(Node):

    def __init__(self):
        super().__init__("hospitalbot_training", allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

        # Defines which action the script will perform "random_agent", "training", "retraining" or "hyperparam_tuning"
        self._training_mode = "training"


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        self.extractor = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self._features_dim = 32

    def forward(self, observations):
        return self.extractor(observations["laser"])


policy_kwargs = dict(
    features_extractor_class=CustomFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=32),
    net_arch=[64, 64]  # Two-layer Q-network
)

def main(args=None):

    # Initialize the training node to get the desired parameters
    rclpy.init()
    node = TrainingNode()
    node.get_logger().info("Training node has been created")

    # Create the dir where the trained RL models will be saved
    home_dir = os.path.expanduser('~')
    pkg_dir = 'ros2_ws/src/uq-in-slam/test_new_pkg_name'
    trained_models_dir = os.path.join(home_dir, pkg_dir, 'rl_models')
    log_dir = os.path.join(home_dir, pkg_dir, 'logs')
    
    # If the directories do not exist we create them
    if not os.path.exists(trained_models_dir):
        os.makedirs(trained_models_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # First we register the gym environment created in hospitalbot_env module
    register(
        id="HospitalBotEnv-v0",
        entry_point="test_new_pkg_name.hospitalbot_env:HospitalBotEnv",
        max_episode_steps=300000,
    )

    node.get_logger().info("The environment has been registered")

    env = gym.make('HospitalBotEnv-v0')
    env = Monitor(env)

    # Here we check if the custom gym environment is fine
    check_env(env)
    node.get_logger().info("Environment check finished")

    # Now we create two callbacks which will be executed during training
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=2000, verbose=1)
    eval_callback = EvalCallback(env, callback_on_new_best=stop_callback, eval_freq=10000, best_model_save_path=trained_models_dir, n_eval_episodes=40)

    model = DQN("MultiInputPolicy",
                env,
                tensorboard_log=f"{log_dir}/sb3_logs/",
                learning_rate=1e-3,
                policy_kwargs=policy_kwargs,
                buffer_size=32,
                learning_starts=5000,
                target_update_interval=500,
                train_freq=4,
                gamma=0.99,
                exploration_fraction=0.3,
                exploration_final_eps=0.1,
                verbose=1)
    
    # Execute training
    try:
        model.learn(total_timesteps=int(400000), reset_num_timesteps=False, callback=eval_callback, tb_log_name="DQN_test_3")
    except KeyboardInterrupt:
        model.save(f"{trained_models_dir}/DQN_test_3")
    # Save the trained model
    model.save(f"{trained_models_dir}/DQN_test_3")

    node.get_logger().info("The training is finished, now the node is destroyed")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
