from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    ld = LaunchDescription()

    random_agent = Node(
        package='hospital_robot_spawner',
        executable='random_agent',
    )

    ld.add_action(random_agent)

    return ld
