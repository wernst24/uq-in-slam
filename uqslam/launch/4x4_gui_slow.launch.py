"""
Launches a world with a 4x4 grid of circuits & robots in Gazebo, with GUI enabled and real-time simulation.
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = LaunchDescription()
    world_file_name = "4x4_slow_circuit.world"
    pkg_dir = get_package_share_directory("uqslam")

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
            ],
            output="screen",
        )
    )

    ld.add_action(
        Node(package="uqslam", executable="spawn_4x4", output="screen")
    )

    return ld
