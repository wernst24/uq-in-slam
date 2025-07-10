#!usr/bin/env python3

import rclpy
from rclpy.node import Node
from gymnasium.envs.registration import register
from uqslam.uqslam.hospitalbot_env import HospitalBotEnv
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os
from stable_baselines3.common.monitor import Monitor


class TrainingNode(Node):

    def __init__(self):
        super().__init__("hospitalbot_training", allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

        # Defines which action the script will perform "random_agent", "training", "retraining" or "hyperparam_tuning"
        self._training_mode = "training"

def main(args=None):

    # Initialize the training node to get the desired parameters
    rclpy.init()
    node = TrainingNode()
    node.get_logger().info("Training node has been created")

    # Create the dir where the trained RL models will be saved
    home_dir = os.path.expanduser('~')
    pkg_dir = 'ros2_ws/src/uq-in-slam/uqslam'
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

    model = A2C(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
    )
    
    # Execute training
    trial_number = 1
    try:
        model.learn(total_timesteps=int(400000), reset_num_timesteps=False, callback=eval_callback, tb_log_name="A2C_test_{}".format(trial_number))
    except KeyboardInterrupt:
        model.save(f"{trained_models_dir}/A2C_test_{trial_number}")
    # Save the trained model
    model.save(f"{trained_models_dir}/A2C_test_{trial_number}")

    node.get_logger().info("The training is finished, now the node is destroyed")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
