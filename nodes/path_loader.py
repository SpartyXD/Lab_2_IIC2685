#!/usr/bin/env python3
import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from std_msgs.msg import Header


class PathLoader(Node):

    def __init__(self):
        super().__init__("path_loader")
        self.declare_parameter("path_file", "path1.txt")
        self.declare_parameter("topic", "nav_plan")
        self.declare_parameter("frame_id", "odom")

        frame_id = self.get_parameter("frame_id").value
        self.path_pub = self.create_publisher(
            Path, self.get_parameter("topic").value, 10
        )
        self.path = Path(header=Header(frame_id=frame_id))
        self.published = False

        # Resolver ruta del archivo
        f_path = self.get_parameter("path_file").value
        if not os.path.isabs(f_path):
            f_path = os.path.join(
                get_package_share_directory("lab2"), "config", f_path
            )

        # Carga de puntos (soporta "x, y", "x y" y "(x, y)")
        with open(f_path, encoding="utf-8") as f:
            for number, line in enumerate(f, start=1):
                try:
                    parts = [
                        float(x)
                        for x in line.strip().strip("()").replace(",", " ").split()
                        if x
                    ]
                except ValueError:
                    self.get_logger().warn(
                        f'Linea {number} ignorada: "{line.strip()}"'
                    )
                    continue

                if len(parts) >= 2:
                    pose = PoseStamped()
                    pose.header.frame_id = frame_id
                    pose.pose.position.x, pose.pose.position.y = parts[0], parts[1]
                    pose.pose.orientation.w = 1.0
                    self.path.poses.append(pose)

        self.get_logger().info(
            f"{len(self.path.poses)} puntos cargados desde {f_path}"
        )
        self.timer = self.create_timer(0.5, self.try_publish)

    def try_publish(self):
        if self.path_pub.get_subscription_count() > 0:
            self.path.header.stamp = self.get_clock().now().to_msg()
            self.path_pub.publish(self.path)
            self.get_logger().info("Trayectoria publicada")
            self.published = True
            self.timer.cancel()
        else:
            self.get_logger().info(
                "Esperando a que follow_the_carrot pida la trayectoria...",
                throttle_duration_sec=2.0,
            )


def main():
    rclpy.init()
    node = PathLoader()
    while rclpy.ok() and not node.published:
        rclpy.spin_once(node)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()