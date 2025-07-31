"""
Gymnasium-compatible reinforcement learning environment for training a mobile robot
to navigate within a Gazebo-based ROS 2 simulation using LIDAR and camera data.

This module defines the `UQNavBotEnv` class, which inherits from both `RobotController`
(a ROS 2 node handling the robot's interaction with Gazebo) and `gymnasium.Env` (providing
a standard interface for reinforcement learning). The environment is intended for research
in uncertainty-aware navigation and decision making in complex environments and has been refactored from HospitalBot.

Key Features:
- Discrete action space (forward, left, right)
- Visual observation input (32x32 RGB image)
- Collision detection via LIDAR
- Spawn point randomization
- ROS 2 integration for velocity commands, reset services, and sensor feedback
- Compatible with Stable-Baselines3 and Gymnasium training pipelines

"""
import rclpy  # ROS 2 client library in Python
from gymnasium import Env  # Base class for Gym-compatible RL environments
from gymnasium.spaces import Box, Discrete  # Spaces for observation and action
import numpy as np  # Numerical computing
# Custom ROS2 node for robot control
from uqslam.uqslam.Environments.robot_controller import RobotController


class UQNavBotEnv(RobotController, Env):
    """Can 
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

        # Initializes the min distance from an obstacle for which the episode is concluded without success
        # This accounts for the front dimension of the robot - DO NOT CHANGE THIS
        # I changed it
        self._minimum_dist_from_obstacles = 0.35

        # valid spawn locations
        self._valid_spawn_xy = [
            [0, 0],
            [2, 0],
            [4.5, 0],
            [4.5, -2],
            [4.5, -4],
            [3, -4],
            [3, -3],
            [2, -3],
            [0, -4],
            [0, -2]

        ]

        # Initialize step count
        self._num_steps = 0

        # max steps for truncate
        self._max_num_steps = 300

        # episodes
        self._num_episodes = 0

        # total reward for session (for debug)
        self._total_reward = 0

        # Debug prints on console
        self.get_logger().info("INITIAL AGENT LOCATION: " +
                               str(self._initial_agent_location))
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

        # only image
        self.observation_space = Box(
            low=0, high=1, shape=(32, 32, 3), dtype=np.float32)

        # DR. DERA'S REWARD PARAMETERS
        self._goal_position = np.array([4.5, -4.0], dtype=np.float32)  # Target destination
        self._previous_distance_to_goal = None
        self._collision_reward = -100.0
        self._goal_reward = 200.0
        self._step_penalty = -0.1
        self._distance_reward_factor = 10.0
        self._min_laser_range = 3.5  # From gym-gazebo

    def step(self, action):
        done = False
        truncated = False
        
        # Store previous position for distance calculation
        prev_position = self._agent_location.copy() if hasattr(self, '_agent_location') else np.array([0, 0])
        
        self._num_steps += 1
        self.send_velocity_command(self.action_to_direction[action])
        self.spin()
        
        observation = self._get_obs()
        info = self._get_info()
        
        # DR. DERA'S REWARD IMPLEMENTATION
        reward = self._step_penalty  # Small penalty for each step
        
        # Calculate distance to goal
        current_distance = np.linalg.norm(self._agent_location - self._goal_position)
        
        # Distance-based reward (similar to gym-gazebo)
        if self._previous_distance_to_goal is not None:
            distance_improvement = self._previous_distance_to_goal - current_distance
            reward += distance_improvement * self._distance_reward_factor
        
        self._previous_distance_to_goal = current_distance
        
        # Goal reached check
        if current_distance < 0.5:  # Within 0.5m of goal
            reward += self._goal_reward
            done = True
            self.get_logger().info(f"GOAL REACHED! Distance: {current_distance:.2f}")
        
        # Collision detection (Dr. Dera's approach)
        min_laser_reading = np.min(info["laser"])
        if min_laser_reading < self._minimum_dist_from_obstacles:
            reward += self._collision_reward
            done = True
            self.get_logger().info(f"COLLISION! Min laser: {min_laser_reading:.2f}")
        
        # Laser-based reward shaping (encourage safe navigation)
        if min_laser_reading < 1.0:
            reward -= (1.0 - min_laser_reading) * 5.0  # Penalty for getting too close
        
        # Check if episode is terminated
        if self._num_steps > self._max_num_steps:
            done = True
            truncated = True

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

        # Updates state and additional infos
        observation = self._get_obs()
        info = self._get_info()

        # Reset the number of steps
        self._num_steps = 0

        # reset total reward
        self._total_reward = 0

        # Reset distance tracking
        self._previous_distance_to_goal = None

        return observation, info

    def _get_obs(self):
        return (self._image_raw/255.0).astype(np.float32)

    def _get_info(self):
        return {
            "laser": self._laser_reads,
            "image_raw": self._image_raw
        }

    def spin(self):
        # spin node until new lidar data
        self._done_laser = False
        while self._done_laser is False:
            rclpy.spin_once(self)

    # Random angle & position selected from valid spawn points
    def randomize_robot_location(self):
        xy = self._valid_spawn_xy[np.random.randint(len(self._valid_spawn_xy))]
        position_x = xy[0]
        position_y = xy[1]

        theta = np.random.uniform(0, 2 * np.pi)
        orientation_z = np.sin(theta/2)  # Corrected quaternion math
        orientation_w = np.cos(theta/2)

        # ENABLE PROPER RANDOMIZATION
        return [position_x, position_y, orientation_z, orientation_w]

    def close(self):
        # # Shuts down the node to avoid creating multiple nodes on re-creation of the env
        # Print achieved data
        self.get_logger().info("Number of episodes: " + str(self._num_episodes-1))

        # Destroy all clients/publishers/subscribers
        self.destroy_node()
