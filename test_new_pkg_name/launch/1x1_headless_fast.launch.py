"""
Demo for spawn_entity.
Launches Gazebo and spawns a model
"""

# A bunch of software packages that are needed to launch ROS2
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = LaunchDescription()
    world_file_name = "1x1_fast_circuit.world"
    pkg_dir = get_package_share_directory("test_new_pkg_name")

    os.environ["GAZEBO_MODEL_PATH"] = os.path.join(pkg_dir, "models")

    world = os.path.join(pkg_dir, "worlds", world_file_name)

    ld.add_action(
        ExecuteProcess(
            cmd=[
                "gzserver",
                "--verbose",
                world,
                "-s",
                "libgazebo_ros_init.so",
                "-s",
                "libgazebo_ros_factory.so",
            ]
        )
    )

    ld.add_action(
        Node(
            package="test_new_pkg_name",
            executable="spawn_demo",
            arguments=[
                "p3at",
                "demo",
                "1",
                "16.0",  # x position
                "0.0",  # y position
            ],
        )
    )

    return ld
