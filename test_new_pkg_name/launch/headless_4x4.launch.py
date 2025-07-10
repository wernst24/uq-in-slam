"""
Launches a world with a 4x4 grid of circuits & robots in Gazebo, with no GUI and accelerated simulation.
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
    world_file_name = 'circuit_4x4.world'
    pkg_dir = get_package_share_directory('test_new_pkg_name')

    os.environ["GAZEBO_MODEL_PATH"] = os.path.join(pkg_dir, 'models')

    world = os.path.join(pkg_dir, 'worlds', world_file_name)

    ld.add_action(ExecuteProcess(
        cmd=['gzserver', '--verbose', world, '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so']
    ))

    ld.add_action(Node(
        package='test_new_pkg_name', 
        executable='spawn_4x4' 
    ))

    return ld