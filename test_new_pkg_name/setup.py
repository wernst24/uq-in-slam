import os # Operating system library
from glob import glob # Handles file path names
from setuptools import setup # Facilitates the building of packages

package_name = 'test_new_pkg_name'

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
        
        # Path to the pioneer sdf file
        (os.path.join('share', package_name,'models/pioneer3at/'), glob('./models/pioneer3at/model.sdf')),

        # Path to the pioneer config file
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
          'spawn_demo = test_new_pkg_name.spawn_demo:main',
          'start_training = test_new_pkg_name.start_training:main',
          'trained_agent = test_new_pkg_name.trained_agent:main',
          'random_agent = test_new_pkg_name.random_agent:main',
          'always_forward = test_new_pkg_name.always_forward:main',
          'det_a2c_train = test_new_pkg_name.Project_Det_TRAIN:main',
        ],
    },
)
