import os # Operating system library
from glob import glob # Handles file path names
from setuptools import setup # Facilitates the building of packages

package_name = 'uqslam'

# Path of the current directory
cur_directory_path = os.path.abspath(os.path.dirname(__file__))

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # Path to the launch file      
        (os.path.join('share', package_name,'launch'), glob('launch/*.launch.py')),

        # Path to the config file
        (os.path.join('share', package_name,'config'), glob('config/*.yaml')),

        # Path to the world file
        (os.path.join('share', package_name,'worlds/'), glob('./worlds/*')),

        # Path to the mobile robot sdf and config file
        (os.path.join('share', package_name,'models/mobile_warehouse_robot/'), glob('./models/mobile_warehouse_robot/*')),
        
        # Path to the pioneer sdf and xacro files
        (os.path.join('share', package_name,'models/pioneer3at/'), glob('./models/pioneer3at/model.sdf')),
        (os.path.join('share', package_name,'models/pioneer3at/'), glob('./models/pioneer3at/model.sdf.xacro')),
        (os.path.join('share', package_name,'models/pioneer3at/'), glob('./models/pioneer3at/pioneer3at.xacro')),
        (os.path.join('share', package_name,'models/pioneer3at/'), glob('./models/pioneer3at/model.config')),

        # Path to the world file (i.e. warehouse + global environment)
        (os.path.join('share', package_name,'models/'), glob('./worlds/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Liam Ernst',
    maintainer_email='lae5777@rit.edu',
    description='This package creates a simulation in Gazebo which includes a differential drive robot with Lidar and a Maze world',
    license='I am a student, I dont know what to put here :)',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
          f'spawn_demo = {package_name}.spawn_demo:main',
          f'spawn_robot_param = {package_name}.spawn_robot_param:main',
          f'start_training = {package_name}.start_training:main',
          f'trained_agent = {package_name}.trained_agent:main',
          f'random_agent = {package_name}.random_agent:main',
          f'always_forward = {package_name}.always_forward:main',
          f'det_a2c_train = {package_name}.Project_Det_TRAIN:main',
        ],
    },
)
