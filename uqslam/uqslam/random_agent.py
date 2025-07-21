import rclpy
import gymnasium as gym
from uqslam.p3at_slow_control_env import P3atSlowControlEnv
import argparse


class TrainingNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("hospitalbot_random_agent", allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        # defines which action the script will perform ("random_agent", "training", "retraining", :hyperparam_tuning")
        self._training_mode = "random_agent"


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat_steps", type=int, default=4, help="Number of times to repeat each action (slower control frequency)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    parsed_args = parser.parse_args()

    rclpy.init()
    node = TrainingNode()
    node.get_logger().info("Random agent node has been created")

    # Use the slower control frequency environment
    env = P3atSlowControlEnv(repeat_steps=parsed_args.repeat_steps)
    node.get_logger().info(f"Using P3atSlowControlEnv with repeat_steps={parsed_args.repeat_steps}")

    episodes = parsed_args.episodes
    # execute random agent
    node.get_logger().info("Starting the RANDOM AGENT now")
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        while not done:
            obs, reward, done, truncated, info = env.step(env.action_space.sample())
            # Optionally log agent state and reward here

    env.close()


if __name__ == "__main__":
    main()
