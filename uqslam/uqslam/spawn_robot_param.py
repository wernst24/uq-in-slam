"""
Spawns N p3at robots in Gazebo simulation, using xacro parameters for prefix, namespace, collide_bitmask, and visible.
Spawns robots sequentially in a single node to avoid /spawn_entity contention.
"""
import os
import sys
import rclpy
import math
import subprocess

from ament_index_python.packages import get_package_share_directory
from gazebo_msgs.srv import SpawnEntity

def main():
    # Usage: spawn_robot_param.py <num_robots> <x> <y> <z> <visible>
    argv = sys.argv[1:]
    if len(argv) < 5:
        print("Usage: spawn_robot_param.py <num_robots> <x> <y> <z> <visible>")
        sys.exit(1)

    num_robots = int(argv[0])
    x, y, z = float(argv[1]), float(argv[2]), float(argv[3])
    visible = argv[4]

    # Prepare SDF generation
    pkg_share = get_package_share_directory("uqslam")
    xacro_path = os.path.join(pkg_share, "models", "pioneer3at", "pioneer3at.xacro")

    # Start node
    rclpy.init()
    node = rclpy.create_node("entity_spawner")
    node.get_logger().info('Creating Service client to connect to `/spawn_entity`')
    client = node.create_client(SpawnEntity, "/spawn_entity")

    node.get_logger().info("Connecting to `/spawn_entity` service...")
    if not client.service_is_ready():
        client.wait_for_service()
        node.get_logger().info("...connected!")

    for i in range(num_robots):
        robot_name = f"robot{i+1}"
        namespace = f"{robot_name}_ns"
        collide_bitmask = hex(1 << i)
        prefix = f"{robot_name}_"

        sdf_out_path = f"/tmp/{robot_name}.sdf"
        xacro_cmd = [
            "xacro",
            xacro_path,
            f"prefix:={prefix}",
            f"namespace:={namespace}",
            f"collide_bitmask:={collide_bitmask}",
            f"visible:={visible}"
        ]
        with open(sdf_out_path, "w") as sdf_out:
            subprocess.run(xacro_cmd, stdout=sdf_out, check=True)

        # Set data for request
        request = SpawnEntity.Request()
        request.name = robot_name
        with open(sdf_out_path, 'r') as f:
            request.xml = f.read()
        request.robot_namespace = namespace
        request.initial_pose.position.x = x
        request.initial_pose.position.y = y
        request.initial_pose.position.z = z

        desired_angle = float(math.radians(-90))
        request.initial_pose.orientation.z = float(math.sin(desired_angle/2))
        request.initial_pose.orientation.w = float(math.cos(desired_angle/2))

        node.get_logger().info(f"Spawning {robot_name} in namespace {namespace} with bitmask {collide_bitmask}")
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future)
        if future.result() is not None:
            print(f'{robot_name} response: {future.result()}')
        else:
            raise RuntimeError(
                f'Exception while calling service for {robot_name}: {future.exception()}')

    node.get_logger().info("All robots spawned. Shutting down node.")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()