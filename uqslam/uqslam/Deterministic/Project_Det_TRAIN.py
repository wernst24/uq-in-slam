#!/usr/bin/env python3

# imports from random_agent.py (bare minimum for Gymnasium env)
import rclpy
import gymnasium as gym
from uqslam.uqslam.Environments.uq_nav_slow_env import UQNavSlowEnv
from rclpy.node import Node

# previous imports
import os
import time
import numpy as np
import random
from .. import liveplot
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import wandb

from distutils.dir_util import copy_tree
import json

from . import Network_Model_Det_TRAIN as net

# TensorFlow performance optimizations - UPDATED
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow verbosity
tf.get_logger().setLevel('ERROR')

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# Disable mixed precision to avoid issues
tf.keras.mixed_precision.set_global_policy('float32')

# Set API key
os.environ["WANDB_API_KEY"] = "084538caaa5ded119d5b9e04a7b699ac8de907fe"


def render():
    pass


class TrainingNode(Node):
    def __init__(self):
        super().__init__("hospitalbot_training", allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)
        self._training_mode = "training"


def main():
    rclpy.init()
    node = TrainingNode()
    node.get_logger().info("Training node has been created")
    
    args = net.parser.parse_args()
    
    # CONFIGURABLE SLOW CONTROL PARAMETERS
    repeat_steps = 2  # Number of times to repeat each action
    
    # Create environment with configurable repeat steps
    env = UQNavSlowEnv(repeat_steps=repeat_steps)
    node.get_logger().info(f"Created UQNavSlowEnv with repeat_steps={repeat_steps}")
    
    outdir = '/tmp/gazebo_gym_experiments'
    plotter = liveplot.LivePlot(outdir)
    last_time_steps = np.ndarray(0)

    # DR. DERA'S TRAINING PARAMETERS
    minibatch_size = 32  # Increased batch size
    learningRate = 1e-3  # Standard Adam learning rate
    discountFactor = 0.99
    network_outputs = 3
    memorySize = 100000
    learnStart = 500    # Start learning earlier
    EXPLORE = 10000     # Longer exploration
    INITIAL_EPSILON = 0.9  # Start lower for more deterministic training
    FINAL_EPSILON = 0.01   # Lower final epsilon
    epsilon_discount = 0.995  # Slower decay
    explorationRate = INITIAL_EPSILON
    current_epoch = 0
    loadsim_seconds = 0
    img_rows, img_cols, img_channels = 32, 32, 3

    # Create model and agent with updated parameters - FIX INITIALIZATION
    model_A2C = net.Model_A2C(num_actions=3)
    agent = net.A2CAgent(model_A2C, lr=learningRate, gamma=discountFactor, entropy_c=1e-5)
    
    # Pre-compile model to avoid retracing warnings
    dummy_input = tf.random.normal((minibatch_size, img_rows, img_cols, img_channels))
    _ = model_A2C(dummy_input, training=False)
    node.get_logger().info("Model pre-compiled successfully")

    # Reward tracking
    last100Rewards = [0] * 100
    last100RewardsIndex = 0
    last100Filled = False
    myRewardList = []
    x_plot_avg = []
    y_plot_avg = []
    steps = 1000  # Max steps per episode

    total_episodes = 10000
    highest_reward = 0

    # Batch processing
    batch_sz = 16
    ep_rewards = [0.0]
    angleRange = np.pi/3

    # CURRICULUM LEARNING PARAMETERS
    curriculum_episode_threshold = 1000  # Switch curriculum every 1000 episodes
    current_curriculum_level = 0

    # Initialize wandb with updated project name
    wandb.init(
        project=f"Train_Traditional_A2C_{total_episodes}_lr_{learningRate}_repeat_{repeat_steps}",
        config={
            "total_episodes": total_episodes,
            "learning_rate": learningRate,
            "repeat_steps": repeat_steps,
            "batch_size": minibatch_size,
            "max_steps_per_episode": steps,
            "discount_factor": discountFactor,
            "network_outputs": network_outputs,
            "memory_size": memorySize,
            "learn_start": learnStart,
            "initial_epsilon": INITIAL_EPSILON,
            "final_epsilon": FINAL_EPSILON,
            "epsilon_discount": epsilon_discount
        },
        tags=["a2c", "deterministic", "slow_control"]
    )

    start_time = time.time()
    observation, info = env.reset()
    node.get_logger().info("Starting training loop")

    # Main training loop
    for x in range(1, total_episodes, 1):
        done = False
        cumulated_reward = 0

        # Initialize batch arrays
        mem_observation = np.zeros((minibatch_size, img_rows, img_cols, img_channels))
        actions = np.zeros(minibatch_size)
        values = np.zeros(minibatch_size)
        rewards = np.zeros(minibatch_size)
        dones = np.zeros(minibatch_size)
        batch_step = 0

        # Episode loop
        for i in range(steps):
            # More deterministic exploration strategy
            if np.random.random() < explorationRate:
                # Epsilon-greedy with bias toward forward movement
                if np.random.random() < 0.6:  # 60% chance to go forward when exploring
                    actions[batch_step] = 0  # Forward
                else:
                    actions[batch_step] = np.random.choice([1, 2])  # Left or Right
                values[batch_step] = 0.0
            else:
                # Policy-based action (more deterministic)
                try:
                    actions[batch_step], values[batch_step] = agent.action_value(observation)
                except Exception as e:
                    node.get_logger().error(f"Error in action_value: {e}")
                    actions[batch_step] = 0  # Default to forward
                    values[batch_step] = 0.0

            # Take step in environment
            newObservation, rewards[batch_step], dones[batch_step], truncated, info = env.step(actions[batch_step])

            # Store experience
            mem_observation[batch_step] = observation
            observation = newObservation
            cumulated_reward += rewards[batch_step]

            # Training logic
            if batch_step == minibatch_size - 1:  # Train when batch is full
                if i > 100:  # Start learning very early
                    try:
                        _, next_value = agent.action_value(observation)
                        returns, advs = agent._returns_advantages(rewards, dones, values, next_value)
                        acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
                        
                        # Clip advantages for stability (Dr. Dera's approach)
                        acts_and_advs[:, 1] = np.clip(acts_and_advs[:, 1], -1.0, 1.0)
                        
                        losses = agent.model.train_on_batch(mem_observation, [acts_and_advs, returns])
                        
                        # Enhanced logging
                        if isinstance(losses, list) and len(losses) >= 3:
                            wandb.log({
                                'total_loss': losses[0],
                                'actor_loss': losses[1],
                                'critic_loss': losses[2],
                                'step': x * steps + i,
                                'epsilon': explorationRate,
                                'batch_step': batch_step
                            })
                            
                    except Exception as e:
                        node.get_logger().error(f"Training error: {e}")

                # Reset batch
                batch_step = -1
                mem_observation = np.zeros((minibatch_size, img_rows, img_cols, img_channels))
                actions = np.zeros(minibatch_size)
                values = np.zeros(minibatch_size)
                rewards = np.zeros(minibatch_size)
                dones = np.zeros(minibatch_size)

            # Episode termination
            if dones[batch_step] or truncated:
                last_time_steps = np.append(last_time_steps, [int(i + 1)])

                # Update reward tracking
                last100Rewards[last100RewardsIndex] = cumulated_reward
                last100RewardsIndex += 1
                if last100RewardsIndex >= 100:
                    last100Filled = True
                    last100RewardsIndex = 0

                # Time tracking
                m, s = divmod(int(time.time() - start_time + loadsim_seconds), 60)
                h, m = divmod(m, 60)
                
                # Logging
                if not last100Filled:
                    print(f"EP {x} - {i+1} steps - CReward: {round(cumulated_reward, 2)} "
                          f"Eps={round(explorationRate, 2)} Time: {h:d}:{m:02d}:{s:02d}")
                else:
                    avg_reward = int(sum(last100Rewards) / len(last100Rewards))
                    print(f"EP {x} - {i+1} steps - last100 C_Rewards: {avg_reward} "
                          f"- CReward: {round(cumulated_reward, 2)} "
                          f"Eps={round(explorationRate, 2)} Time: {h:d}:{m:02d}:{s:02d}")

                # Log episode metrics to wandb
                episode_metrics = {
                    'episode_reward': cumulated_reward,
                    'episode_length': i + 1,
                    'episode': x,
                    'exploration_rate': explorationRate
                }
                if last100Filled:
                    episode_metrics['avg_reward_last_100'] = sum(last100Rewards) / len(last100Rewards)
                
                wandb.log(episode_metrics)

                # CURRICULUM LEARNING - Update environment based on curriculum level
                if current_curriculum_level == 0:
                    # Level 0: Close spawns only
                    env.env._valid_spawn_xy = [[0, 0], [2, 0], [0, -2]]
                elif current_curriculum_level == 1:
                    # Level 1: Medium distance spawns
                    env.env._valid_spawn_xy = [[0, 0], [2, 0], [4.5, 0], [3, -3], [0, -2]]
                else:
                    # Level 2: All spawn locations
                    env.env._valid_spawn_xy = [
                        [0, 0], [2, 0], [4.5, 0], [4.5, -2], [4.5, -4],
                        [3, -4], [3, -3], [2, -3], [0, -4], [0, -2]
                    ]

                observation, info = env.reset()
                break

            # Progress logging
            if i % 2500 == 0:
                print(f"Frames = {i}")

            batch_step += 1

        # Save and plot data
        myRewardList.append(cumulated_reward)

        # Slower epsilon decay for more stable learning
        if x > 200:  # Start decay after more episodes
            explorationRate = max(FINAL_EPSILON, explorationRate * epsilon_discount)

        # Log action distribution for debugging
        if x % 10 == 0:  # Every 10 episodes
            node.get_logger().info(f"Episode {x}: explorationRate = {explorationRate:.4f}")

        # Update wandb with current exploration rate
        wandb.log({
            'current_reward': myRewardList[x-1], 
            'episode': x,
            'exploration_rate_current': explorationRate  # Track actual exploration rate
        })

        # Debug logging
        if 'losses' in locals():
            node.get_logger().debug(f"[{x + 1}/{total_episodes}] Losses: {losses}")

        # Early stopping for timing
        if x == 200:
            total_time = time.time() - start_time
            wandb.log({'total_training_time': total_time})
            np.save('Total_Test_Time_DET.npy', total_time)

        if x % 25 == 0:  # Every 25 episodes
            # Calculate action distribution for this episode
            episode_actions = actions[:batch_step] if batch_step > 0 else actions
            action_counts = np.bincount(episode_actions.astype(int), minlength=3)
            total_actions = len(episode_actions)
            
            if total_actions > 0:
                forward_pct = (action_counts[0] / total_actions) * 100
                left_pct = (action_counts[1] / total_actions) * 100
                right_pct = (action_counts[2] / total_actions) * 100
                
                # Log to wandb
                wandb.log({
                    'action_forward_pct': forward_pct,
                    'action_left_pct': left_pct,
                    'action_right_pct': right_pct,
                    'episode': x,
                    'curriculum_level': current_curriculum_level if 'current_curriculum_level' in locals() else 0
                })
                
                node.get_logger().info(f"Ep {x} Actions - Forward: {forward_pct:.1f}%, "
                                      f"Left: {left_pct:.1f}%, Right: {right_pct:.1f}%")

    # Final statistics
    l = last_time_steps.tolist()
    l.sort()
    final_score = last_time_steps.mean()
    print(f"Overall score: {final_score:.2f}")

    final_model_path = f'/tmp/turtle_c2c_Actor_final_ep{x}_train.weights.h5'
    agent.model.save_weights(final_model_path)
    wandb.save(final_model_path)
    print(f"Final model saved: {final_model_path}")    
        
    wandb.log({'final_overall_score': final_score})
    wandb.finish()
    
    env.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
