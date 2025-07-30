import rclpy
from uqslam.uq_nav_slow_env import UQNavSlowEnv


class TrainingNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("hospitalbot_random_agent", allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)
        self._training_mode = "random_agent"


def main(args=None):
    rclpy.init()
    node = TrainingNode()
    node.get_logger().info("Random agent node has been created")

    # Directly instantiate the slow environment (no gym.make needed)
    env = UQNavSlowEnv(instance_num=0, repeat_steps=4)
    node.get_logger().info("Slow environment created")

    episodes = 10
    node.get_logger().info("Starting the RANDOM AGENT now")
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            obs, reward, done, truncated, info = env.step(env.action_space.sample())
            # node.get_logger().info(f"Episode {ep+1}, Reward: {reward}")
        # node.get_logger().info(f"Episode {ep+1} completed with total reward: {env._total_reward}")

    env.close()
    node.get_logger().info("Random agent completed")


if __name__ == "__main__":
    main()
