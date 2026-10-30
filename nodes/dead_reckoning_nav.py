#!/usr/bin/env python3
import math
from time import time, sleep
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseArray, Vector3, Pose
from std_msgs.msg import String, Float64
from math import atan2, sqrt
from threading import Thread
#Ocuparemos el molde que nos dio el Nacho, crack

FACTOR_CORRECCION = 1.1
INFTY = 1e9 #Le mandaremos esto al controlador pa que se quede quietito si no quiero pescar el setpoint
POSE_EPSILON = 0.1 #epsilon de pose
ANGLE_EPSILON = 0.1 #epsilon de angulo
UMBRAL_TRAMO = 0.15 

def normalizar(a):
    #Lleva cualquier angulo a (-pi, pi)
    return math.atan2(math.sin(a), math.cos(a))


class DeadReckoningNav(Node):
    def __init__(self):
        super().__init__("dead_reckoning_nav")
        self.pos_actual = [0.0, 0.0, 0.0]
        self.wait_time = 0.1

        self.real_pose = self.create_subscription(Pose, "/real_pose", self.real_cb, 10)
        self.vel_effort_queue = []
        self.angle_effort_queue = []

        self.dist_set_point = self.create_publisher( Float64, "/velocity_setpoint", 1 )
        self.dist_state = self.create_publisher( Float64, "/velocity_measure", 1 )
        self.vel_actuation = self.create_subscription( Float64, "/velocity_effort", self.vel_cb, 1)

        self.angle_set_point = self.create_publisher( Float64, "/angle_setpoint", 1 )
        self.angle_state = self.create_publisher( Float64, "/angle_measure", 1 )
        self.angle_actuation = self.create_subscription( Float64, "/angle_effort", self.angle_cb, 1 )

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.goal_list_sub = self.create_subscription(PoseArray, "/goal_list", self.accion_mover, 10)
        self.occupancy_state_sub = self.create_subscription(Vector3, "/occupancy_state", self.callback_occupancy_state, 10)
        self.evento_pub = self.create_publisher(String, "/trajectory_events", 10)
        self.hay_obstaculo = False


    def real_cb(self, msg):
        self.pos_actual[0] = msg.position.x
        self.pos_actual[1] = msg.position.y
        self.pos_actual[2] = normalizar(2*math.atan2(msg.orientation.z, msg.orientation.w))
        self.get_logger().info(f"Estoy en {self.pos_actual}", throttle_duration_sec=1.0)

    def vel_cb(self, msg):
        self.vel_effort_queue.append(msg.data)

    def angle_cb(self, msg):
        self.angle_effort_queue.append(msg.data)

    def detener(self):
        speed = Twist()
        speed.linear.x, speed.angular.z = (0.0, 0.0)
        self.cmd_vel_pub.publish(speed)


    def aplicar_controlador_vel(self, distancia):
        self.vel_effort_queue.clear()
        msg = Float64()
        speed = Twist()
 
        x0, y0, th0 = self.pos_actual
        cos_angle, sin_angle = math.cos(th0), math.sin(th0)
 
        #Silenciar al Controlador de angulo
        msg.data = INFTY
        self.angle_set_point.publish(msg)
 
        msg.data = float(distancia)
        self.dist_set_point.publish(msg)
 
        recorrido = 0.0
        while(abs(distancia - recorrido) > POSE_EPSILON):
            #Publicar estado
            msg.data = recorrido
            self.dist_state.publish(msg)
 
            #Aplicar lo q el controlador me diga
            try:
                vel = self.vel_effort_queue.pop(0)
                speed.linear.x, speed.angular.z = (vel, 0.0)
                self.cmd_vel_pub.publish(speed)
            except IndexError:
                self.get_logger().warning("Cola de vel vacia :c")
            sleep(0.1)
 
            recorrido = (self.pos_actual[0] - x0)*cos_angle + (self.pos_actual[1] - y0)*sin_angle


    def aplicar_controlador_angulo(self, setpoint):
        self.angle_effort_queue.clear()
        msg = Float64()
        speed = Twist()

        #Silenciar al Controlador de vel
        msg.data = INFTY
        self.dist_set_point.publish(msg)

        msg.data = setpoint
        self.angle_set_point.publish(msg)

        diff = normalizar(setpoint - self.pos_actual[2])
        while(abs(diff) > ANGLE_EPSILON):
            #Publicar estado
            msg.data = self.pos_actual[2]
            self.angle_state.publish(msg)

            #Aplicar lo q el controlador me diga
            try:
                vel = self.angle_effort_queue.pop(0)
                speed.linear.x, speed.angular.z = (0.0, vel)
                self.cmd_vel_pub.publish(speed)
            except IndexError:
                self.get_logger().warning("Cola de angulos vacia :c")
                pass
            sleep(0.1)

            diff = normalizar(setpoint - self.pos_actual[2])
            

    def aplicar_velocidad(self, displacement_list):
        theta_obj = self.pos_actual[2]
 
        for lin, rot in displacement_list:
            self.get_logger().info(f"Desplazamiento: ({lin:.3f}, {rot:.3f})")
 
            while self.hay_obstaculo:
                self.detener()
                sleep(self.wait_time)
 
            if rot != 0.0:
                theta_obj = normalizar(theta_obj + rot)
                self.get_logger().info(f"Girando a {theta_obj:.3f}...")
                self.aplicar_controlador_angulo(theta_obj)
            elif lin != 0.0:
                self.get_logger().info(f"Avanzando {lin:.3f} m...")
                self.aplicar_controlador_vel(lin)
 
            self.get_logger().info('Listo')
            self.detener()


            
    def reapuntar(self, diff_coord, vertical=False):
        #Funcion auxiliar para obtener angulo para reapuntar rapido
        ang_deseado = 0
        if(vertical):
            ang_deseado = math.pi/2 if diff_coord > 0 else -math.pi/2
        else:
            ang_deseado = 0 if diff_coord > 0 else math.pi

        return float(ang_deseado)
        
            
    def mover_robot_a_destino(self, goal_pose):
        self.get_logger().info('Moviendose de (%f, %f, %f) a (%f, %f, %f)' % (*self.pos_actual, *goal_pose))
        x_goal = float(goal_pose[0])
        y_goal = float(goal_pose[1])
        angle_goal = float(goal_pose[2])

        x, y, th = self.pos_actual
        displacement_list = []
 
        #Eje X
        x_diff = x_goal - x
        if abs(x_diff) > UMBRAL_TRAMO:
            rot = normalizar(self.reapuntar(x_diff) - th)
            if abs(rot) > ANGLE_EPSILON:
                displacement_list.append((0.0, rot))
                th = normalizar(th + rot)
            displacement_list.append((abs(x_diff), 0.0))
            x = x_goal
 
        #Eje Y
        y_diff = y_goal - y
        if abs(y_diff) > UMBRAL_TRAMO:
            rot = normalizar(self.reapuntar(y_diff, vertical=True) - th)
            if abs(rot) > ANGLE_EPSILON:
                displacement_list.append((0.0, rot))
                th = normalizar(th + rot)
            displacement_list.append((abs(y_diff), 0.0))
            y = y_goal
 
        #Angulo final
        rot = normalizar(angle_goal - th)
        if abs(rot) > ANGLE_EPSILON:
            displacement_list.append((0.0, rot))


        self.aplicar_velocidad(displacement_list)
        self.get_logger().info('Creo que llegue a (%f, %f, %f)...' % (x_goal, y_goal, angle_goal))
        self.get_logger().info('Posicion real (%f, %f, %f)!' % (self.pos_actual[0], self.pos_actual[1], self.pos_actual[2]))
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