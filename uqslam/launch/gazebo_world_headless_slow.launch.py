"""
Demo for spawn_entity.
Launches Gazebo and spawns a model
"""
import os
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    ld = LaunchDescription()
    use_sim_time = LaunchConfiguration('use_sim_time', default='True')
    world_file_name = 'circuit_realtime.world'
    pkg_dir = get_package_share_directory('uqslam')

    os.environ["GAZEBO_MODEL_PATH"] = os.path.join(pkg_dir, 'models')

    world = os.path.join(pkg_dir, 'worlds', world_file_name)

    # Launch Gazebo with GUI
    ld.add_action(ExecuteProcess(
        cmd=['gzserver', '--verbose', world, '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so'],
        output='screen'))

    ld.add_action(Node(
        package='uqslam',
        executable='spawn_demo',
        arguments=[
            'HospitalBot',
            'demo',
            '1',
            '0.0',
            '0.0'
        ],
        output='screen'
    ))

    return ld