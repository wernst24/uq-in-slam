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
    ld = []
    use_sim_time = LaunchConfiguration('use_sim_time', default='True')
    world_file_name = 'circuit.world'
    pkg_dir = get_package_share_directory('uqslam')

    os.environ["GAZEBO_MODEL_PATH"] = os.path.join(pkg_dir, 'models')

    world = os.path.join(pkg_dir, 'worlds', world_file_name)

    # Launch Gazebo with GUI
    ld.append(ExecuteProcess(
        cmd=['gazebo', '--verbose', world, '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so'],
        output='screen'))

    # Spawn 16 robots in a 4x4 grid, each with a unique name and position
    spawn_entity = Node(package='uqslam', executable='spawn_demo',
                        arguments=['HospitalBot', 'demo', '1', '16.0', '0.0'],
                        output='screen')
    ld.append(spawn_entity)
    return LaunchDescription(ld)
