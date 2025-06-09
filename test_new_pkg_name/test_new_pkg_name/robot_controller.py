from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from functools import partial
import numpy as np
import math
from gazebo_msgs.srv import DeleteEntity, SpawnEntity, SetModelState, SetEntityState
import os
from ament_index_python.packages import get_package_share_directory
#from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

class RobotController(Node):
    """
    This class defines all the methods to:
        - Publish actions to the agent (move the robot)
        - Subscribe to sensors of the agent (get laser scans and robot position)
        - Reset the simulation

    Topics list:
        - /demo/cmd_vel : linear and angular velocity of the robot
        - /demo/odom : odometry readings of the chassis of the robot
        - /demo/laser/out : laser readings
    
    Services used:
        - /demo/set_entity_state : sets the new state of the robot and target when an episode ends

    Services not used:
        - /reset_simulation : resets the gazebo simulation
        - /delete_entity : unspawns the robot from the simulation
        - /spawn_entity : spawns the robot in the simulation in a semi-random position
    """
    def __init__(self):
        super().__init__('robot_controller')
        self.get_logger().info("The robot controller node has just been created")

        # Action publisher
        self.action_pub = self.create_publisher(Twist, '/demo/cmd_vel', 10)
        # Position subscriber
        self.pose_sub = self.create_subscription(Odometry, '/demo/odom', self.pose_callback, 1)
        # Laser subscriber
        self.laser_sub = self.create_subscription(LaserScan, '/demo/laser/out', self.laser_callback, 1)
        # Reset model state client - this resets the pose and velocity of a given model within the world
        self.client_state = self.create_client(SetEntityState, "/demo/set_entity_state")

        # Reset simulation client - UNUSED
        self.client_sim = self.create_client(Empty, "/reset_simulation")


        # Get the directory of the sdf of the robot
        self._pkg_dir = os.path.join(
            get_package_share_directory("test_new_pkg_name"), "models",
        "pioneer3at", "model.sdf")

        # Initialize attributes - This will be immediately re-written when the simulation starts
        self._agent_location = np.array([np.float32(1),np.float32(16)])
        self._laser_reads = np.array([np.float32(10)] * 61)

    # Method to send the velocity command to the robot
    def send_velocity_command(self, velocity):
        msg = Twist()
        msg.linear.x = float(velocity[0])
        msg.angular.z = float(velocity[1])
        self.action_pub.publish(msg)

    # Method that saves the position of the robot each time the topic /demo/odom receives a new message
    def pose_callback(self, msg: Odometry):
        self._agent_location = np.array([np.float32(np.clip(msg.pose.pose.position.x,-12,12)), np.float32(np.clip(msg.pose.pose.position.y,-35,21))])
        self._agent_orientation = 2* math.atan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        #self.get_logger().info("Agent position: " + str(self._agent_location))
        #self.get_logger().info("Agent orientation: " + str(math.degrees(self._agent_orientation)))
        self._done_pose = True

    # Method that saves the laser reads each time the topic /demo/laser/out receives a new message
    def laser_callback(self, msg: LaserScan):
        self._laser_reads = np.array(msg.ranges)
        # Converts inf values to 10
        self._laser_reads[self._laser_reads == np.inf] = np.float32(10)
        #self.get_logger().info("Min Laser Read: " + str(min(self._laser_reads)))
        self._done_laser = True

    # Method to set the state of the robot when an episode ends - /demo/set_entity_state service
    def call_set_robot_state_service(self, robot_pose=[1, 16, -0.707, 0.707]):
        while not self.client_state.wait_for_service(1.0):
            self.get_logger().warn("Waiting for service...")

        request = SetEntityState.Request()
        request.state.name = self.robot_name
        # Pose (position and orientation)
        request.state.pose.position.x = float(robot_pose[0])
        request.state.pose.position.y = float(robot_pose[1])
        request.state.pose.orientation.z = float(robot_pose[2])
        request.state.pose.orientation.w = float(robot_pose[3])
        # Velocity
        request.state.twist.linear.x = float(0)
        request.state.twist.linear.y = float(0)
        request.state.twist.linear.z = float(0)
        request.state.twist.angular.x = float(0)
        request.state.twist.angular.y = float(0)
        request.state.twist.angular.z = float(0)

        future = self.client_state.call_async(request)
        future.add_done_callback(partial(self.callback_set_robot_state))

    # Method that elaborates the future obtained by callig the call_set_robot_state_service method
    def callback_set_robot_state(self, future):
        try:
            response= future.result()
            #self.get_logger().info("The Environment has been successfully reset")
            self._done_set_rob_state = True
        except Exception as e:
            self.get_logger().error("Service call failed: %r" % (e,))

    # Method to set the state of the target when an episode ends - /demo/set_entity_state service
    def call_set_target_state_service(self, position=[1, 10]):
        while not self.client_state.wait_for_service(1.0):
            self.get_logger().warn("Waiting for service...")

        request = SetEntityState.Request()
        request.state.name = "Target"
        # Pose (position and orientation)
        request.state.pose.position.x = float(position[0])
        request.state.pose.position.y = float(position[1])

        future = self.client_state.call_async(request)
        future.add_done_callback(partial(self.callback_set_target_state))

    # Method that elaborates the future obtained by callig the call_set_target_state_service method
    def callback_set_target_state(self, future):
        try:
            response= future.result()
            #self.get_logger().info("The Environment has been successfully reset")
        except Exception as e:
            self.get_logger().error("Service call failed: %r" % (e,))
