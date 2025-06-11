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

        # Initializes the starting agent location for each episode (x,y,angle) - effective only for randomization level 0 and 2 (see below)
        self._initial_agent_location = np.array([0, 0, -90], dtype=np.float32)  # Default is [1, 16, -90]

        # If True, the observation space is normalized between [0,1] (except distance which is between [0,6], see below)
        self._normalize_obs = True

        # Because action space is Discrete, normalization is unnecessary. I will not implement normalize_act = True.
        self._normalize_act = False

        # Initializes the maximal linear velocity used in actions
        self._max_linear_velocity = 1

        # Initializes the minimal linear velocity used in actions
        self._min_linear_velocity = 0

        # Initializes the angular velocity used in actions - This must always be symmetric (no need for max and min)
        self._angular_velocity = 1

        # Initializes the min distance from an obstacle for which the episode is concluded without success
        # This accounts for the front dimension of the robot - DO NOT CHANGE THIS
        self._minimum_dist_from_obstacles = 0.26

        # Initialize step count
        self._num_steps = 0

        # Initialize episode count
        self._num_episodes = 0

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
                2: np.array([0.05, 0.3], dtype=np.float32),  # Right
                }

        if self._normalize_obs is True:
            # # Normalized State space - dictionary with: "Robot position", "Laser reads"
            self.observation_space = Dict(
                {
                    # Agent position can be anywhere inside the hospital, its limits are x=[-12;12] and y=[-35,21]
                    # Maximum distance can be 60 but in 95% of cases it never goes over 10
                    "agent": Box(low=np.array([0, 0]), high=np.array([6, 1]), dtype=np.float32),
                    # Laser reads are 61 and can range from 0.08 to 10
                    "laser": Box(low=0, high=1, shape=(61,), dtype=np.float32),
                }
            )

        else:
            # # State space - dictionary with: "Robot position", "Laser reads"
            self.observation_space = Dict(
                {
                    # Agent position can be anywhere inside the hospital, its limits are x=[-12;12] and y=[-35,21]
                    #"agent": Box(low=np.array([-12, -35]), high=np.array([12, 21]), dtype=np.float32),
                    "agent": Box(low=np.array([0, -math.pi]), high=np.array([60, math.pi]), dtype=np.float32),
                    # Laser reads are 61 and can range from 0.08 to 10
                    "laser": Box(low=0, high=np.inf, shape=(61,), dtype=np.float32),
                }
            )

        # This variable defines all the possible locations where the robot can spawn with a self._randomize_env_level of 5
        # [x, y, angle, x_lowerbound, x_upperbound, y_lowerbound, y_upperbound, angle_lowerbound, angle_upperbound]
        # x, y and angle defines the center of the location and the orientation, the bounds define the limits from the center in which the robot will spawn
        self.robot_locations = [[0, 0, 0, 0, 0, 0, 0, 0, 0]]

        # This variable defines all the possible locations where the target can spawn with a self._randomize_env_level of 5
        # Obviously these locations are strictly associated with the locations where the robot spawns
        # [x, y, x_lowerbound, x_upperbound, y_lowerbound, y_upperbound]
        # x and y defines the center of the location, the bounds define the limits from the center in which the target will spawn

        # Variables to montior successes and failures
        self._successes = 0
        self._failures = 0
        self._completed_paths = 0

    def step(self, action):

        # Increase step number
        self._num_steps += 1

        # De-normalize the action to send the command to robot
        if self._normalize_act is True:
            action = self.denormalize_action(action)

        # Apply the action
        #self.get_logger().info("Action applied: " + str(action))
        self.send_velocity_command(action)

        # Spin the node until laser reads and agent location are updated - VERY IMPORTANT
        self.spin()

        # Compute the polar coordinates of the robot with respect to the target
        self.transform_coordinates()

        # This is used to overwrite laser reads and see how the agent behaves when all laser samples have max value
        #self._laser_reads = np.full((61,),10, dtype=np.float32)

        # Update robot location and laser reads
        observation = self._get_obs()
        #self.get_logger().info(str(observation["laser"]))
        # Update infos
        info = self._get_info()

        # Compute Reward
        # NOTE: change compute_rewards to depend on action, not info.
        reward = self.compute_rewards(info)

        # Check if episode is terminated

        # NOTE: only need to check if collision occurred, or if enough steps
        done = False  # NOTE NOTE NOTE NOTE BAD!!!!!
        # have passed.

        return observation, reward, done, False, info

    def render(self):
        # Function to render env steps
        # Our env is always rendered if you start gazebo_world.launch.py
        # If you don't want to see graphical output launch headless_world.launch.py, it is also much faster for training (>20 times faster)
        pass

    def reset(self, seed=None, options=None):
        # Increment episode counter
        self._num_episodes += 1

        # Get the new pose of the robot
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
        self.transform_coordinates()

        # This is used to overwrite laser reads and see how the agent behaves when all laser samples have max value
        # self._laser_reads = np.full((61,),10, dtype=np.float32)

        # Updates state and additional infos
        observation = self._get_obs()
        info = self._get_info()

        # Reset the number of steps
        self._num_steps = 0

        # Debug print
        # self.get_logger().info("Exiting reset function")

        return observation, info

    def _get_obs(self):
        # Returns the current state of the system
        obs = {"agent": self._polar_coordinates, "laser": self._laser_reads}
        # Normalize observations
        if self._normalize_obs is True:
            obs = self.normalize_observation(obs)
        # self.get_logger().info("Agent Location: " + str(self._agent_location))
        return obs

    def _get_info(self):
        # returns the distance from agent to target and laser reads
        return {
            "distance": math.dist(self._agent_location, self._target_location),
            "laser": self._laser_reads,
            "angle": self._theta
        }

    def spin(self):
        # This function spins the node until it gets new sensor data (executes both laser and odom callbacks)
        self._done_pose = False
        self._done_laser = False
        while (self._done_pose is False) or (self._done_laser is False):
            rclpy.spin_once(self)

    def transform_coordinates(self):
        # This function takes as input: robot position, robot orientation and target position
        # To compute the polar coordinates of the robot with respect to target

        # Radius
        self._radius = math.dist(self._agent_location, self._target_location)

        # Target X coordinate expressed in the robot cartesian system
        self._robot_target_x = math.cos(-self._agent_orientation) * \
            (self._target_location[0]-self._agent_location[0]) - \
                math.sin(-self._agent_orientation)*(self._target_location[1]-self._agent_location[1])

        # Here we initialize the final variable which will be passed to the Gym environment
        self._polar_coordinates = np.array([self._radius, self._theta], dtype=np.float32)

        # Debug prints for new cartesian systems transformation (gazebo -> robot)
        # self.get_logger().info("Theta: " + str(math.degrees(self._theta)) + "\n" + "Radius: " + str(self._radius))
        # self.get_logger().info("Distance from target: " + str(math.sqrt(self._robot_target_x**2 + self._robot_target_y**2)))
        # self.get_logger().info("Xt: " + str(self._robot_target_x) + " - Yt: " + str(self._robot_target_y))
        # self.get_logger().info("Polar coordinates: " + str(self._polar_coordinates))

    # NOTE: change to randomize position along circuit, IF DESIRED.
    def randomize_robot_location(self):
        position_x = 0
        position_y = 0
        orientation_z = 0
        orientation_w = 0
        return [position_x, position_y, orientation_z, orientation_w]

    # NOTE: implement velocity & collision-based reward
    def compute_rewards(self, info, action=None):
        reward = 0
        return reward

    def normalize_observation(self, observation):
        # This method normalizes the observations taken from the robot in the range [0,1]
        # Distance from target can range from 0 to 60 but we divide by ten since most times the agent never goes further
        observation["agent"][0] = observation["agent"][0]/10
        # Angle from target can range from -pi to pi
        observation["agent"][1] = (observation["agent"][1] + math.pi)/(2*math.pi)
        # Laser reads range from 0 to 10
        observation["laser"] = observation["laser"]/10

        # Debug
        # self.get_logger().info("Agent: " + str(observation["agent"]))
        # self.get_logger().info("Laser: " + str(observation["laser"]))

        return observation

    def denormalize_action(self, norm_act):
        # # This method de-normalizes the action before sending it to the robot - The action is normalized between [-1,1]
        # Linear velocity can also be asymmetric
        action_linear = ((self._max_linear_velocity*(norm_act[0]+1)) + (self._min_linear_velocity*(1-norm_act[0])))/2
        # Angular velicity is symmetric
        action_angular = ((self._angular_velocity*(norm_act[1]+1)) + (-self._angular_velocity*(1-norm_act[1])))/2

        # Debug
        # self.get_logger().info("Linear velocity: " + str(action_linear))
        # self.get_logger().info("Angular velocity: " + str(action_angular))

        return np.array([action_linear, action_angular], dtype=np.float32)

    def close(self):
        # # Shuts down the node to avoid creating multiple nodes on re-creation of the env
        # Print achieved data
        self.get_logger().info("Number of episodes: " + str(self._num_episodes-1))
        self.get_logger().info("Successes: " + str(self._successes))
        self.get_logger().info("Failures: " + str(self._failures))
        self.get_logger().info("Truncated episodes: " + str(self._num_episodes-1 - self._successes - self._failures))

        # Destroy all clients/publishers/subscribers
        """self.destroy_client(self.client_sim)
        self.destroy_client(self.client_state)
        self.destroy_publisher(self.action_pub)
        self.destroy_subscription(self.pose_sub)
        self.destroy_subscription(self.laser_sub)"""
        self.destroy_node()
