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
    for r in range(4):
        for c in range(4):
            ld.append(Node(
                package='uqslam',
                executable='spawn_demo',
                arguments=[
                    'HospitalBot',
                    f'simulation_{r*4+c}',
                    '1',
                    str(100.0 + c*2.0),  # X position
                    str(-100.0 + r*2.0)  # Y position
                ],
                output='screen'
            ))

    return LaunchDescription(ld)
