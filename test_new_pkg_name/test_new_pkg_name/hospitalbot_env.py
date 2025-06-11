import rclpy
from gymnasium import Env
from gymnasium.spaces import Dict, Box, Discrete
import numpy as np
from test_new_pkg_name.robot_controller import RobotController
import math
# from rcl_interfaces.srv import GetParameters


class HospitalBotEnv(RobotController, Env):
    """
    This class defines the RL environment. Here are defined:
        - Action space
        - State space
        - Target location

    The methods available are:
        - step: makes a step of the current episode
        - reset: resets the simulation environment when an episode is finished
        - close: terminates the environment
    This class inherits from both RobotController and Env.
    Env is a standard class of Gymnasium library which defines the basic needs of an RL environment.

    RobotController is a ROS2 Node used to control the agent. It includes the following attributes:
        - _agent_location: current position of the robot (in gazebo's coordinate system)
        - _laser_reads: current scan of the LIDAR

    And the following methods (only the ones usefull here):
        - send_velocity_command: imposes a velocity to the agent (the robot)
        - call_reset_simulation_service: resets the simulation
        - call_reset_robot_service: resets the robot position to desired position
    """
    def __init__(self):

        # Initialize the Robot Controller Node
        super().__init__()
        self.get_logger().info("All the publishers/subscribers have been started")

        # ENVIRONMENT PARAMETERS
        self.robot_name = 'HospitalBot'

        # Initializes the starting agent location for each episode (x,y,angle)
        self._initial_agent_location = np.array([0, 0, -90], dtype=np.float32)

        # If True, the observation space is normalized between [0,1] (except distance which is between [0,6], see below)
        self._normalize_obs = True

        # Initializes the min distance from an obstacle for which the episode is concluded without success
        # This accounts for the front dimension of the robot - DO NOT CHANGE THIS
        # I changed it
        self._minimum_dist_from_obstacles = 0.05

        # Initialize step count
        self._num_steps = 0

        # max steps for truncate
        self._max_num_steps = 300

        # episodes
        self._num_episodes = 0

        # total reward for session (for debug)
        self._total_reward = 0

        # Debug prints on console
        self.get_logger().info("INITIAL AGENT LOCATION: " + str(self._initial_agent_location))
        self.get_logger().info("MIN OBSTACLE DIST: " + str(self._minimum_dist_from_obstacles))

        # Action space - 3 discrete options: forward, left, right
        self.action_space = Discrete(3)

        # Linear velocity, Angular velocity
        # [m/s, rad/s]
        self.action_to_direction = {
                0: np.array([0.3, 0.0], dtype=np.float32),  # Forward
                1: np.array([0.05, 0.3], dtype=np.float32),  # Left
                2: np.array([0.05, -0.3], dtype=np.float32),  # Right
                }

        if self._normalize_obs is True:
            # # Normalized State space - dictionary with: "Robot position", "Laser reads"
            self.observation_space = Dict({"laser": Box(low=0, high=1, shape=(61,), dtype=np.float32)})
        else:
            self.observation_space = Dict({"laser": Box(low=0, high=np.inf, shape=(61,), dtype=np.float32)})

    def step(self, action):
        done = False
        truncated = False

        reward = 0

        if action == 0:
            reward += 5  # 5 for forward
        else:
            reward += 1  # 1 for turn

        self._num_steps += 1

        # self.get_logger().info("Action applied: " + str(action))
        self.send_velocity_command(self.action_to_direction[action])

        # Spin the node until laser reads and agent location are updated - VERY IMPORTANT
        self.spin()

        # This is used to overwrite laser reads and see how the agent behaves when all laser samples have max value
        # self._laser_reads = np.full((61,),10, dtype=np.float32)

        # Update robot location and laser reads
        observation = self._get_obs()
        # self.get_logger().info(str(observation["laser"]))
        # Update infos
        info = self._get_info()

        # check for crash
        if any(observation["laser"] < self._minimum_dist_from_obstacles):
            reward -= 200
            self.get_logger().info("CRASHED, reward = " + str(self._total_reward))
            self._total_reward += reward
            done = True
            # debug

        # Check if episode is terminated
        if self._num_steps > self._max_num_steps:
            # truncate session
            done = True
            truncated = True
            # debug

        self._total_reward += reward
        return observation, reward, done, truncated, info

    def render(self):
        # Function to render env steps
        # Our env is always rendered if you start gazebo_world.launch.py
        # If you don't want to see graphical output launch headless_world.launch.py, it is also much faster for training (>20 times faster)
        pass

    def reset(self, seed=None, options=None):
        # Increment episode counter
        self._num_episodes += 1

        # Get the new pose of the robot
        # NOTE: doesn't actually randomize
        pose2d = self.randomize_robot_location()

        # Reset the done reset variable
        self._done_set_rob_state = False
        # Call the set robot position service
        self.call_set_robot_state_service(pose2d)
        # Here we spin the node until the /set_entity_state service responds, otherwise we get random observations
        while self._done_set_rob_state is False:
            rclpy.spin_once(self)

        # Compute the initial observation
        self.spin()

        # This is used to overwrite laser reads and see how the agent behaves when all laser samples have max value
        # self._laser_reads = np.full((61,),10, dtype=np.float32)

        # Updates state and additional infos
        observation = self._get_obs()
        info = self._get_info()

        # Reset the number of steps
        self._num_steps = 0

        # reset total reward
        self._total_reward = 0

        # Debug print
        # self.get_logger().info("Exiting reset function")

        return observation, info

    def _get_obs(self):
        # Returns the current state of the system
        obs = self._laser_reads
        # Normalize observations
        if self._normalize_obs is True:
            obs = self.normalize_observation(obs)
        # self.get_logger().info("Agent Location: " + str(self._agent_location))
        return {"laser": obs}

    def _get_info(self):
        return {"laser": self._laser_reads}

    def spin(self):
        # This function spins the node until it gets new sensor data (executes both laser and odom callbacks)
        self._done_pose = False
        self._done_laser = False
        while (self._done_pose is False) or (self._done_laser is False):
            rclpy.spin_once(self)

    # NOTE: change to randomize position along circuit, IF DESIRED.
    def randomize_robot_location(self):
        position_x = 0
        position_y = 0
        orientation_z = 0
        orientation_w = 1.0
        return [position_x, position_y, orientation_z, orientation_w]

    # NOTE: implement velocity & collision-based reward
    def compute_rewards(self, info, action=None):
        reward = 0
        return reward

    def normalize_observation(self, observation):
        # This method normalizes the observations taken from the robot in the range [0,1]
        # Laser reads range from 0 to 10

        return observation/10

    def close(self):
        # # Shuts down the node to avoid creating multiple nodes on re-creation of the env
        # Print achieved data
        self.get_logger().info("Number of episodes: " + str(self._num_episodes-1))

        # Destroy all clients/publishers/subscribers
        self.destroy_node()
