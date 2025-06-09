import rclpy
import gymnasium as gym
from test_new_pkg_name.hospitalbot_env import HospitalBotEnv


class TrainingNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("hospitalbot_random_agent", allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        # defines which action the script will perform ("random_agent", "training", "retraining", :hyperparam_tuning")
        self._training_mode = "random_agent"


def main(args=None):
    rclpy.init()
    node = TrainingNode()
    node.get_logger().info("Random agent node has been created")

    # register gymnasium env created in hospitalbot_env module
    gym.envs.registration.register(
            id="HospitalBotEnv-v0",
            entry_point="test_new_pkg_name.hospitalbot_env:HospitalBotEnv",
            max_episode_steps=300,
    )

    node.get_logger().info("The environment has been registered")

    env = gym.make("HospitalBotEnv-v0")
    # old code checks env here
    # number of episodes
    episodes = 10
    # execute random agent
    node.get_logger().info("Starting the RANDOM AGENT now")
    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            obs, reward, done, truncated, info = env.step(env.action_space.sample())
            # should log agent state and reward


if __name__ == "__main__":
    main()
