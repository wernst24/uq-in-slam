import rclpy
from uqslam.p3at_slow_control_env import P3atSlowControlEnv
from uqslam.p3at_vec_env import P3atVecEnv
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
    envs = P3atVecEnv(repeat_steps=parsed_args.repeat_steps, num_envs=4)
    node.get_logger().info(f"Using vec env (n = {envs.num_envs} envs) with repeat_steps={parsed_args.repeat_steps}")

    episodes = parsed_args.episodes
    # execute random agent
    node.get_logger().info("Starting the RANDOM AGENT now")
    for ep in range(episodes):
        obs, infos = envs.reset()
        done = [False] * envs.num_envs
        while not any(done):
            obs, rewards, dones, truncated, infos = envs.step([envs.action_space.sample() for _ in range(envs.num_envs)])
            # Optionally log agent state and reward here

    envs.close()


if __name__ == "__main__":
    main()
