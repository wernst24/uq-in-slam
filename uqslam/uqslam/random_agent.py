import rclpy
import gymnasium as gym
from uqslam.uqslam.Environments.uq_nav_env import UQNavBotEnv
# import cv2


class TrainingNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("hospitalbot_random_agent", allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)
        # defines which action the script will perform ("random_agent", "training", "retraining", :hyperparam_tuning")
        self._training_mode = "random_agent"


def main(args=None):
    rclpy.init()
    node = TrainingNode()
    node.get_logger().info("Random agent node has been created")

    # register gymnasium env created in uq_nav_env module
    gym.envs.registration.register(
        id="HospitalBotEnv-v0",
        entry_point="uqslam.uq_nav_env:UQNavBotEnv",
        max_episode_steps=3000,
    )

    node.get_logger().info("The environment has been registered")

    env = gym.make("HospitalBotEnv-v0")
    # old code checks env here
    # number of episodes
    episodes = 10
    # execute random agent
    node.get_logger().info("Starting the RANDOM AGENT now")
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        while not done:
            obs, reward, done, truncated, info = env.step(
                env.action_space.sample())
            # node.get_logger().info(f"shape of obs: {obs.shape}")
            # should log agent state and reward

            # while cv2.waitKey(1) != ord('q'):
            #   cv2.imshow("[DEBUG] /demo/my_camera/image_raw", obs)
            # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
