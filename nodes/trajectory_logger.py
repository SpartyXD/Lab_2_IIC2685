#!/usr/bin/env python3
import math
import os
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory as pkg_dir
import matplotlib.pyplot as plt


from dead_reckoning_nav import FACTOR_CORRECCION

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

def error_promedio_angulo(ideales, puntos):
    errores = []
    for i in range(min(len(puntos), len(ideales)) - 1):
        errores.append(angulo(direccion(puntos[i], puntos[i+1]) - direccion(ideales[i], ideales[i+1])))

    cambios = [angulo(b - a) for a, b in zip(errores, errores[1:])]

    if(len(cambios) == 0):
        return 0
    else:
        return sum(cambios)/len(cambios)

def factor_sugerido(error_prom):
    try:
        return FACTOR_CORRECCION*(math.pi/2)/(math.pi/2 + error_prom)
    except ZeroDivisionError:
        return 1.0


class TrajectoryLogger(Node):
    def __init__(self):
        super().__init__("trajectory_logger_node")
        self.nombre_exp = f"Factor de correccion: {FACTOR_CORRECCION}"

        self.landmarks = self.load_landmarks()
        self.real = [] #Ruta completa de /real_pose
        self.odom = [] #Ruta completa de /odom
        self.puntos = [] #Un (x_real, y_real, x_odom, y_odom) por cada "Landmark"

        self.create_subscription(Pose, "/real_pose", self.real_cb, 10)
        self.create_subscription(Odometry, "/odom", self.odom_cb, 10)
        self.create_subscription(String, "/trajectory_events", self.evento_cb, 10)
        self.get_logger().info(f"Listo experimento {self.nombre_exp}. Esperando datos...")


    def load_landmarks(self):
        #Cargar lista de poses
        poses_list = [(0.0, 0.0)]
        path = os.path.join(pkg_dir('lab1'), 'config', 'pose_list.txt')
        with open(path) as poses_file:
            for line in poses_file:
                line = line.strip().strip("()")
                if line:
                    x, y, _ = map(float, line.split(","))
                    poses_list.append((x, y))
        return poses_list

    #Callbacks
    def real_cb(self, msg):
        self.real.append((msg.position.x, msg.position.y))

    def odom_cb(self, msg):
        pos = msg.pose.pose.position
        self.odom.append((pos.x, pos.y))

    def evento_cb(self, msg):
        if not self.real or not self.odom:
            return
        
        if msg.data == "Landmark":
            self.puntos.append(self.real[-1] + self.odom[-1])   # (x_real, y_real, x_odom, y_odom)
            self.get_logger().info(f"Punto {len(self.puntos)} anotado")
        elif msg.data == "Listo":
            self.puntos.append(self.real[-1] + self.odom[-1]) 
            self.get_logger().info(f"Punto {len(self.puntos)} anotado")
            self.analizar_y_graficar()
            rclpy.shutdown()

        
    #Guardar
    def analizar_y_graficar(self):
        if len(self.real) < 2 or len(self.odom) < 2 or len(self.landmarks) < 2:
            print("No llegaron datos suficientes para el experimento :c")
            return
        
        puntos_real = [p[0:2] for p in self.puntos]
        puntos_odom = [p[2:4] for p in self.puntos]
        self.error_prom_angulos_real = error_promedio_angulo(self.landmarks, puntos_real)
        self.error_prom_angulos_odom = error_promedio_angulo(self.landmarks, puntos_odom)
        self.factor_sugerido = factor_sugerido(self.error_prom_angulos_real)

        self.get_logger().info(f"Factor sugerido: {self.factor_sugerido:.3f}")
        self.graficar()


    #Graficos
    def graficar(self):
        fig, ax = plt.subplots(figsize=(10, 8))

        x0, y0 = self.real[0]
        ideal_xy = [(x0 + x, y0 + y) for x, y in self.landmarks]
        puntos_real = [p[0:2] for p in self.puntos]
 
        ax.plot(*zip(*ideal_xy), "k--", label="Trayectoria ideal (pose_list.txt)")
        ax.plot(*zip(*self.real), color="tab:blue", lw=1, alpha=0.6, label="Ruta recorrida")
        ax.plot(*zip(*puntos_real), "o", color="tab:red", label="Landmarks anotados")
        for i, (x, y) in enumerate(puntos_real):
            ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(5, 5))
        
        ax.set_title("Trayectoria")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.axis("equal")
        ax.grid(True)
        ax.legend(fontsize=8)

        texto = (f"Error de cierre (real_pose): {error_cierre(self.real):.3f} m\n"
                f"Error de cierre (odom): {error_cierre(self.odom):.3f} m")
        ax.text(0.02, 0.02, texto, transform=ax.transAxes, va="bottom", ha="left", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
 
        titulo = f"{self.nombre_exp}"
        if self.factor_sugerido != 1.0:
            titulo += f"\nError promedio {math.degrees(self.error_prom_angulos_real):.1f}° (por giro)  →  factor sugerido {self.factor_sugerido:.3f}"
        fig.suptitle(titulo)
        plt.show()


if __name__ == "__main__":
    rclpy.init()
    node = TrajectoryLogger()
    rclpy.spin(node)

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()