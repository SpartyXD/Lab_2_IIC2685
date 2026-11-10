#!/usr/bin/env python3
import math
import os
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Vector3, Twist
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory as pkg_dir
import matplotlib.pyplot as plt

from lab2.nodes.robot_controller import VEL_KP, VEL_KI, VEL_KD, ANGLE_KP, ANGLE_KI, ANGLE_KD
from lab2.nodes.dead_reckoning_nav import ESCALA_TIEMPO

#Funciones auxiliares
def angulo(a):
    return math.atan2(math.sin(a), math.cos(a))

def direccion(p, q):
    return math.atan2(q[1] - p[1], q[0] - p[0])

def largo(p, q):
    return math.hypot(q[0] - p[0], q[1] - p[1])

def error_cierre(ruta):
    #Distancia entre el primer y el ultimo punto
    (x0, y0), (x1, y1) = ruta[0], ruta[-1]
    return math.hypot(x1 - x0, y1 - y0)


class TrajectoryLogger(Node):
    def __init__(self):
        super().__init__("Trajectory_logger")
        self.kp = (VEL_KP, ANGLE_KP)
        self.ki = (VEL_KI, ANGLE_KI)
        self.kd = (VEL_KD, ANGLE_KD)

        self.landmarks = self.load_landmarks()
        self.registro_pos = [] #Ruta completa de /real_pose (x, y)

        #Señales
        self.vel_signals = []
        self.angle_signals = []

        self.create_subscription(Pose, "/real_pose", self.real_cb, 10)
        self.create_subscription(String, "/trajectory_events", self.evento_cb, 10)
        self.create_subscription(Twist, "/Signal_logger", self.signal_cb, 10)
        self.get_logger().info(f"Esperando datos...")


    def load_landmarks(self):
        #Cargar lista de poses
        poses_list = [(1.0, 1.0)]
        path = os.path.join(pkg_dir('lab2'), 'config', 'pose_list.txt')
        with open(path) as poses_file:
            for line in poses_file:
                line = line.strip().strip("()")
                if line:
                    x, y, _ = map(float, line.split(","))
                    poses_list.append((x, y))
        return poses_list

    #Callbacks
    def real_cb(self, msg):
        self.registro_pos.append((msg.position.x, msg.position.y))

    def signal_cb(self, msg):
        self.vel_signals.append((msg.linear.x, msg.linear.y, msg.linear.z))
        self.angle_signals.append((msg.angular.x, msg.angular.y, msg.angular.z))

    def evento_cb(self, msg):
        if not self.registro_pos:
            return

        self.get_logger().info(f"Recibi: {msg.data}")
        
        if msg.data == "Listo":
            self.get_logger().info(f"Experimento terminado")
            self.graficar_senales()
            self.graficar_trayectoria()
            rclpy.shutdown()


    #Graficos
    def graficar_senales(self):
        if len(self.vel_signals) < 5 or len(self.angle_signals) < 5:
            print("No llegaron datos suficientes para el grafico de señales :c")
            return

        #Hacer escala de tiempo y preparar datos
        tiempo = [i*ESCALA_TIEMPO for i in range(len(self.vel_signals))]
        vel_setpoints, vel_acts, vel_states = zip(*self.vel_signals)
        angle_setpoints, angle_acts, angle_states = zip(*self.angle_signals)

        #Graficar todo
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        #Grafico 1 (senales vel)
        ax1.plot(tiempo, vel_setpoints, label="Referencia", linestyle="-", color="crimson", linewidth=2)
        ax1.plot(tiempo, vel_acts, label="Actuacion", linestyle="--", color="darkblue", linewidth=2)
        ax1.plot(tiempo, vel_states, label="Estado", linestyle="-.", color="forestgreen", linewidth=2)
        ax1.set_ylabel("Magnitud")
        ax1.set_title(f"Velocidad(t) con Kp= {self.kp[0]}, Ki= {self.ki[0]}, Kd= {self.kd[0]}")
        ax1.legend(loc="upper right")
        ax1.grid(True, linestyle=":", alpha=0.6)

        #Grafico 2 (senales angle)
        ax2.plot(tiempo, angle_setpoints, label="Referencia", linestyle="-", color="crimson", linewidth=2)
        ax2.plot(tiempo, angle_acts, label="Actuacion", linestyle="--", color="darkblue", linewidth=2)
        ax2.plot(tiempo, angle_states, label="Estado", linestyle="-.", color="forestgreen", linewidth=2)
        ax2.set_xlabel("Tiempo [s]")
        ax2.set_ylabel("Magnitud")
        ax2.set_title(f"Angulo(t) con Kp= {self.kp[1]}, Ki= {self.ki[1]}, Kd= {self.kd[1]}")
        ax2.legend(loc="upper right")
        ax2.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        plt.show()


    def graficar_trayectoria(self):
        if len(self.registro_pos) < 2 or len(self.landmarks) < 2:
            print("No llegaron datos suficientes para el grafico de trayectoria :c")
            return

        fig, ax = plt.subplots(figsize=(10, 8))
        ideal_xy = [(x, y) for x, y in self.landmarks]
 
        ax.plot(*zip(*ideal_xy), "k--", label="Trayectoria ideal (pose_list.txt)")
        ax.plot(*zip(*self.registro_pos), color="tab:blue", lw=1, alpha=0.6, label="Ruta recorrida")
        
        ax.set_title("Trayectoria")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.axis("equal")
        ax.grid(True)
        ax.legend(fontsize=8)

        texto = (f"Error: {error_cierre(self.registro_pos):.3f} m\n")
        ax.text(0.02, 0.02, texto, transform=ax.transAxes, va="bottom", ha="right", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
 
        titulo = f"Trayectoria con Kp={self.kp}, Ki={self.ki}, Kd={self.kd}"
        fig.suptitle(titulo)
        plt.show()


if __name__ == "__main__":
    rclpy.init()
    node = TrajectoryLogger()
    rclpy.spin(node)

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()