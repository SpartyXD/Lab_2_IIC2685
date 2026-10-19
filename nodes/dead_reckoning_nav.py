#!/usr/bin/env python3
import math
from time import time, sleep
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseArray, Vector3
from std_msgs.msg import String
from math import atan2, sqrt
from threading import Thread
#Ocuparemos el molde que nos dio el Nacho, crack

FACTOR_CORRECCION = 1.1

class DeadReckoningNav(Node):
    def __init__(self):
        super().__init__("dead_reckoning_nav")
        self.pos_actual = [0.0, 0.0, 0.0]
        self.lineal_vel = 0.2 
        self.rot_vel = 1.0
        self.wait_time = 0.1

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.goal_list_sub = self.create_subscription(PoseArray, "/goal_list", self.accion_mover, 10)
        self.occupancy_state_sub = self.create_subscription(Vector3, "/occupancy_state", self.callback_occupancy_state, 10)
        self.evento_pub = self.create_publisher(String, "/trajectory_events", 10)
        self.hay_obstaculo = False

    def aplicar_velocidad(self, speed_command_list):
        speed = Twist()
        for x, w, t in speed_command_list:
            self.get_logger().info('Publicando velocidad (%f, %f) por %fs' % (x, w, t))
            t0 = time()
            while time()-t0 <= t:         
                if self.hay_obstaculo:
                    speed.linear.x, speed.angular.z = (0.0, 0.0)
                    self.cmd_vel_pub.publish(speed)
                    t+= self.wait_time
                    sleep(self.wait_time)
                else:
                    speed.linear.x, speed.angular.z = x, w
                    self.cmd_vel_pub.publish(speed)
            self.get_logger().info('Listo')

            speed.linear.x, speed.angular.z = (0.0, 0.0)
            self.cmd_vel_pub.publish(speed)


    def giro(self, ang_actual, ang_deseado):
        delta = math.atan2(math.sin(ang_deseado - ang_actual), math.cos(ang_deseado - ang_actual))
        return (0.0, math.copysign(self.rot_vel, delta), abs(delta)/self.rot_vel*FACTOR_CORRECCION)


    def reapuntar(self, ang_actual, diff_coord, vertical=False):
        #Funcion auxiliar para obtener angulo para reapuntar rapido
        ang_deseado = 0
        if(vertical):
            ang_deseado = math.pi/2 if diff_coord > 0 else -math.pi/2
        else:
            ang_deseado = 0 if diff_coord > 0 else math.pi

        return self.giro(ang_actual, ang_deseado)


    def avanzar(self, diff_coord):
        #Funcion auxiliar que devuelve comando listo para ir a un lugar
        return (self.lineal_vel, 0.0, abs(diff_coord)/self.lineal_vel)
        
            
    def mover_robot_a_destino(self, goal_pose):
        self.get_logger().info('Moviendose de (%f, %f, %f) a (%f, %f, %f)' % (*self.pos_actual, *goal_pose))
        x_goal = goal_pose[0]
        y_goal = goal_pose[1]
        angle_goal = goal_pose[2]

        x_diff = x_goal-self.pos_actual[0]
        y_diff = y_goal-self.pos_actual[1]
        angulo_act = self.pos_actual[2]

        #Obtener pasos y aplicar
        comand_list = []

        #Eje X
        if(x_diff):
            comand_list.append(self.reapuntar(angulo_act, x_diff))
            angulo_act = 0 if x_diff >0 else math.pi
            comand_list.append(self.avanzar(x_diff))

        #Eje Y
        if(y_diff):
            comand_list.append(self.reapuntar(angulo_act, y_diff, vertical=True))
            angulo_act = math.pi/2 if y_diff >0 else -math.pi/2
            comand_list.append(self.avanzar(y_diff))

        #Angulo final
        comand_list.append(self.giro(angulo_act, angle_goal))

        self.aplicar_velocidad(comand_list)
        self.get_logger().info('Llegue a (%f, %f, %f)!' % (x_goal, y_goal, angle_goal))
        self.pos_actual = goal_pose
        sleep(2)
        

    def accion_mover(self, msg):
        self.get_logger().info(f'Poses recibidas: {msg}')

        def moverse():
            for pose in msg.poses:
                self.evento_pub.publish(String(data="Landmark"))
                pos_x = pose.position.x
                pos_y = pose.position.y
                yaw = 2*math.atan2(pose.orientation.z, pose.orientation.w)
                goal_pose = (pos_x, pos_y, yaw)
                self.mover_robot_a_destino(goal_pose)
            self.evento_pub.publish(String(data="Listo"))

        thread_movimiento = Thread(target=moverse, daemon=True)
        thread_movimiento.start()

    def callback_occupancy_state(self, msg):
        self.occupancy = (msg.x, msg.y, msg.z)
        self.hay_obstaculo = any(self.occupancy)

        if msg.x:
            self.get_logger().info("obstacle_left")
        if msg.y:
            self.get_logger().info("obstacle_center")
        if msg.z:
            self.get_logger().info("obstacle_right")


if __name__ == "__main__":
    sleep(3)
    rclpy.init()
    node = DeadReckoningNav()
    rclpy.spin(node)
    rclpy.shutdown()