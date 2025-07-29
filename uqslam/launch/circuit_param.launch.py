"""
Parameterized launch file for spawning N non-colliding robots in a configurable circuit world.
Allows toggling GUI, speedup, robot visibility, and world file.
Automatically assigns unique prefixes, namespaces, and collision bitmasks.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory

def launch_setup(context, *args, **kwargs):
    pkg_dir = get_package_share_directory("uqslam")
    num_robots = LaunchConfiguration("num_robots").perform(context)
    gui = LaunchConfiguration("gui").perform(context).lower() == "true"
    speedup = LaunchConfiguration("speedup").perform(context).lower() == "true"
    visible = LaunchConfiguration("visible").perform(context).lower() == "true"
    world_file = LaunchConfiguration("world_file").perform(context)

    # Generate world from xacro with speedup
    world_xacro_path = os.path.join(pkg_dir, "worlds", world_file)
    generated_world_path = f"/tmp/generated_{world_file.replace('.xacro','')}"
    xacro_cmd = [
        "xacro",
        world_xacro_path,
        f"speedup:={'true' if speedup else 'false'}"
    ]
    os.makedirs("/tmp", exist_ok=True)
    with open(generated_world_path, "w") as f:
        import subprocess
        subprocess.run(xacro_cmd, stdout=f, check=True)

    # Set GAZEBO_MODEL_PATH
    os.environ["GAZEBO_MODEL_PATH"] = os.path.join(pkg_dir, "models")

    # Launch Gazebo server/client
    gz_cmd = [
        "gazebo" if gui else "gzserver",
        "--verbose",
        generated_world_path,
        "-s", "libgazebo_ros_init.so",
        "-s", "libgazebo_ros_factory.so"
    ]
    actions = [
        ExecuteProcess(
            cmd=gz_cmd,
            output="screen"
        )
    ]

    # Spawn all robots in one process (sequentially) using spawn_robot_param.py
    actions.append(
        ExecuteProcess(
            cmd=[
                "ros2", "run", "uqslam", "spawn_robot_param",
                str(num_robots), "0", "0", "0", str(visible).lower()
            ],
            output="screen"
        )
    )

    return actions

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "num_robots",
            default_value=TextSubstitution(text="1"),
            description="Number of robots to spawn"
        ),
        DeclareLaunchArgument(
            "gui",
            default_value=TextSubstitution(text="true"),
            description="Launch Gazebo with GUI (true/false)"
        ),
        DeclareLaunchArgument(
            "speedup",
            default_value=TextSubstitution(text="false"),
            description="Run simulation as fast as possible (true/false)"
        ),
        DeclareLaunchArgument(
            "visible",
            default_value=TextSubstitution(text="true"),
            description="Robots visible in Gazebo (and to eachother) (true/false)"
        ),
        DeclareLaunchArgument(
            "world_file",
            default_value=TextSubstitution(text="circuit.world.xacro"),
            description="World xacro file to use (in worlds/)"
        ),
        OpaqueFunction(function=launch_setup)
    ])