#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from tf_transformations import quaternion_from_euler
from ament_index_python.packages import get_package_share_directory as pkg_dir
import os.path

class PoseLoader(Node):
    def __init__(self):
        super().__init__(f'Pose_loader_Node')
        self.goal_list_pub = self.create_publisher(PoseArray, '/goal_list', 10)
        self.published = False
        self.poses_list = PoseArray()
        self.load()
        self.timer = self.create_timer(0.5, self.try_publish)

    def load(self):
        #Cargar lista de poses
        path = os.path.join(pkg_dir('lab1'), 'config', 'pose_list.txt')
        with open(path) as poses_file:
            for line in poses_file.readlines():
                pose = Pose()
                pose.position.x, pose.position.y, yaw = map(float, line.strip()[1:-1].split(','))
                (pose.orientation.x, 
                 pose.orientation.y, 
                 pose.orientation.z, 
                 pose.orientation.w) = quaternion_from_euler(0,0,yaw)
                self.poses_list.poses.append(pose)
                self.get_logger().info('Pose agregada (%.1f, %.1f, %.1f)' % (pose.position.x, pose.position.y, yaw))
        self.get_logger().info('Lista de poses lista para publicar') 

    def try_publish(self):
        #Intentar publicar si es que algun nodo me la recibe
        if self.goal_list_pub.get_subscription_count() > 0:
            self.goal_list_pub.publish(self.poses_list)
            self.get_logger().info('Lista de poses publicada')
            self.published = True
        else:
            self.get_logger().info('Esperando a que nodo pida poses...', throttle_duration_sec=2.0)
            

if __name__ == '__main__':
    rclpy.init()
    node = PoseLoader()

    #Esperar a que otro nodo reciba poses
    while rclpy.ok() and not node.published:
        rclpy.spin_once(node)
    node.destroy_node()
    rclpy.shutdown()