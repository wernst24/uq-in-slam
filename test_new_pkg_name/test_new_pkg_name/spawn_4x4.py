"""
ROS 2 node to spawn 16 mobile robots (Hospitalbot) in a 4x4 grid, 100 units from (0, 0).
"""

import os
import math
import rclpy
from ament_index_python.packages import get_package_share_directory
from gazebo_msgs.srv import SpawnEntity


def main():
    rclpy.init()
    node = rclpy.create_node("hospitalbot_4x4_spawner")
    client = node.create_client(SpawnEntity, "/spawn_entity")
    node.get_logger().info("Connecting to `/spawn_entity` service...")
    if not client.service_is_ready():
        client.wait_for_service()
        node.get_logger().info("...connected!")

    # Path to robot SDF
    sdf_file_path = os.path.join(
        get_package_share_directory("test_new_pkg_name"),
        "models",
        "pioneer3at",
        "model.sdf",
    )

    # 4x4 grid, 100 units from (0, 0), spaced by 5 units
    grid_size = 4
    spacing = 5.0
    radius = 100.0

    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            name = f"Hospitalbot_{idx}"
            namespace = f"hospitalbot_ns_{idx}"

            # Place robots in a grid, centered at (radius, radius)
            x = radius + (i - (grid_size - 1) / 2) * spacing
            y = radius + (j - (grid_size - 1) / 2) * spacing
            z = 0.0

            request = SpawnEntity.Request()
            request.name = name
            with open(sdf_file_path, "r") as f:
                request.xml = f.read()
            request.robot_namespace = namespace
            request.initial_pose.position.x = x
            request.initial_pose.position.y = y
            request.initial_pose.position.z = z

            desired_angle = math.radians(-90)
            request.initial_pose.orientation.z = math.sin(desired_angle / 2)
            request.initial_pose.orientation.w = math.cos(desired_angle / 2)

            node.get_logger().info(f"Spawning {name} at ({x:.2f}, {y:.2f}, {z:.2f})")
            future = client.call_async(request)
            rclpy.spin_until_future_complete(node, future)
            if future.result() is not None:
                node.get_logger().info(f"Spawned {name}: {future.result()}")
            else:
                node.get_logger().error(
                    f"Exception while spawning {name}: {future.exception()}"
                )

    node.get_logger().info("All robots spawned. Shutting down node.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
