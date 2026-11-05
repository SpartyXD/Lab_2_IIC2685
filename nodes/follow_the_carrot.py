#!/usr/bin/env python3
import math
import time
import rclpy
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

def normalizar(a):
    # Lleva cualquier angulo a (-pi, pi)
    return math.atan2(math.sin(a), math.cos(a))

class FollowTheCarrotSimple(Node):
    def __init__(self):
        super().__init__("follow_the_carrot_simple")

        # Parametros estandar
        self.declare_parameter("lookahead", 0.3)
        self.declare_parameter("ventana", 50)
        self.declare_parameter("kp", 1.0)
        self.declare_parameter("ki", 0.0)
        self.declare_parameter("kd", 0.0)
        self.declare_parameter("v_lineal", 0.1)
        self.declare_parameter("max_w", 1.0)
        self.declare_parameter("eps_final", 0.1)
        self.declare_parameter("control_hz", 20.0)

        self.lookahead = float(self.get_parameter("lookahead").value)
        self.ventana = int(self.get_parameter("ventana").value)
        self.kp = float(self.get_parameter("kp").value)
        self.ki = float(self.get_parameter("ki").value)
        self.kd = float(self.get_parameter("kd").value)
        self.v_lineal = float(self.get_parameter("v_lineal").value)
        self.max_w = float(self.get_parameter("max_w").value)
        self.eps_final = float(self.get_parameter("eps_final").value)
        control_hz = float(self.get_parameter("control_hz").value)

        # Variables para guardar la ruta
        self.ruta = []        # Lista de puntos [x, y]
        self.arco = []        # Distancia acumulada hasta cada punto
        self.largo_total = 0.0
        self.idx = 0          # Indice del dx actual
        self.s_cercano = 0.0  # Distancia recorrida actual

        # Pose del robot
        self.x = None
        self.y = None
        self.yaw = None

        # Variables para el control PID
        self.integral = 0.0
        self.error_previo = 0.0
        self.t_previo = None
        self.terminado = False

        # Tópicos
        self.create_subscription(Path, "nav_plan", self.path_cb, 10)
        self.create_subscription(Odometry, "/odom", self.odom_cb, 10)
        self.carrot_pub = self.create_publisher(PointStamped, "/carrot", 10)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_timer(1.0 / control_hz, self.control_step)

    def path_cb(self, msg): # Desglosamos el mensaje (callback de posición)
        # Convertimos los mensajes a una lista simple de listas [x, y]
        self.ruta = [[p.pose.position.x, p.pose.position.y] for p in msg.poses]
        
        if len(self.ruta) < 2:
            return

        # Calculamos la distancia acumulada punto por punto con Pitágoras
        self.arco = [0.0]
        distancia_acumulada = 0.0
        
        for i in range(1, len(self.ruta)):
            x_anterior, y_anterior = self.ruta[i-1]
            x_actual, y_actual = self.ruta[i]
            
            # Distancia entre dos puntos: raiz((x2-x1)^2 + (y2-y1)^2)
            tramo = math.hypot(x_actual - x_anterior, y_actual - y_anterior)
            distancia_acumulada += tramo
            self.arco.append(distancia_acumulada)

        self.largo_total = distancia_acumulada
        self.idx = 0
        self.s_cercano = 0.0
        self.integral = 0.0
        self.error_previo = 0.0
        self.t_previo = None
        self.terminado = False

    def odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        ori = msg.pose.pose.orientation
        self.yaw = normalizar(2.0 * math.atan2(ori.z, ori.w))

    def buscar_cercano(self): # definimos a donde se va a mover la zanahora
        inicio = self.idx
        # Solo miramos una "ventana" de segmentos hacia adelante para no revisar toda la ruta
        fin = min(inicio + self.ventana, len(self.ruta) - 1)

        mejor_distancia = float("inf")
        mejor_s = self.s_cercano
        mejor_indice = inicio

        # Revisamos cada segmento de línea frente al robot
        for k in range(inicio, fin):
            ax, ay = self.ruta[k]
            bx, by = self.ruta[k+1]
            
            # Vectores básicos
            ab_x = bx - ax
            ab_y = by - ay
            ap_x = self.x - ax
            ap_y = self.y - ay
            
            # Proyectamos al robot sobre el segmento usando producto punto
            largo_segmento_al_cuadrado = ab_x**2 + ab_y**2
            if largo_segmento_al_cuadrado == 0:
                porcentaje = 0.0
            else:
                porcentaje = (ap_x * ab_x + ap_y * ab_y) / largo_segmento_al_cuadrado
            
            # Limitamos el porcentaje entre 0 (punto A) y 1 (punto B)
            porcentaje = max(0.0, min(1.0, porcentaje))
            
            # Coordenadas exactas del punto proyectado
            proy_x = ax + porcentaje * ab_x
            proy_y = ay + porcentaje * ab_y
            
            # ¿A qué distancia está el robot de esa línea?
            distancia_a_linea = math.hypot(self.x - proy_x, self.y - proy_y)
            
            # Nos quedamos con el segmento que esté más cerca
            if distancia_a_linea < mejor_distancia:
                mejor_distancia = distancia_a_linea
                mejor_indice = k
                
                # Calculamos cuánta ruta hemos recorrido hasta ese punto
                largo_segmento = math.sqrt(largo_segmento_al_cuadrado)
                distancia_en_ruta = self.arco[k] + (porcentaje * largo_segmento)
                # control de casos para que el robot nunca retroceda
                mejor_s = max(self.s_cercano, distancia_en_ruta)

        self.idx = mejor_indice
        self.s_cercano = mejor_s
        return self.s_cercano, mejor_distancia

    def punto_en(self, distancia_objetivo): # Integramos la función buscar_cercano para sumar al lookahead
        distancia_objetivo = max(0.0, min(distancia_objetivo, self.largo_total))
        
        # Buscamos en qué segmento cae esta distancia
        for j in range(len(self.arco) - 1):
            if self.arco[j] <= distancia_objetivo <= self.arco[j+1]:
                # Interpolación lineal simple
                largo_tramo = self.arco[j+1] - self.arco[j]
                if largo_tramo == 0:
                    porcentaje = 0.0
                else:
                    porcentaje = (distancia_objetivo - self.arco[j]) / largo_tramo
                
                ax, ay = self.ruta[j]
                bx, by = self.ruta[j+1]
                
                x_interpolado = ax + porcentaje * (bx - ax)
                y_interpolado = ay + porcentaje * (by - ay)
                return x_interpolado, y_interpolado
                
        # Si algo falla, devolvemos el último punto
        return self.ruta[-1][0], self.ruta[-1][1]

    def pid(self, error): #PID de toda la vida
        ahora = self.get_clock().now().nanoseconds / 1e9
        if self.t_previo is None:
            dt = 0.0
            derivada = 0.0
        else:
            dt = ahora - self.t_previo
            derivada = (error - self.error_previo) / dt if dt > 0 else 0.0
            
        self.t_previo = ahora
        self.error_previo = error

        if dt > 0 and self.ki > 0:
            self.integral += error * dt
            limite = self.max_w / self.ki
            self.integral = max(-limite, min(limite, self.integral))

        u = self.kp * error + self.ki * self.integral + self.kd * derivada
        return max(-self.max_w, min(self.max_w, u))

    def control_step(self): # Se encarga de accionar todo
        if self.terminado or not self.ruta or self.x is None:
            return

        # 1. Dónde estamos en la ruta
        s_cerca, dist_ruta = self.buscar_cercano()
        
        # 2. Ponemos la zanahoria más adelante
        s_carrot = min(s_cerca + self.lookahead, self.largo_total)
        xc, yc = self.punto_en(s_carrot)
        
        # 3. ¿Hacia dónde tenemos que mirar para ver la zanahoria?
        theta_carrot = math.atan2(yc - self.y, xc - self.x)
        al_final = (s_carrot >= self.largo_total - 1e-9)

        # Publicar zanahoria para depuración (RViz2)
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.point.x = xc
        msg.point.y = yc
        self.carrot_pub.publish(msg)

        # 4. Condición de término: si la zanahoria está al final y nosotros cerca del final
        dist_final = math.hypot(self.x - self.ruta[-1][0], self.y - self.ruta[-1][1])
        if al_final and dist_final < self.eps_final:
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd) # Manda 0,0
            self.terminado = True
            return

        # 5. Girar usando PID
        error = normalizar(theta_carrot - self.yaw)
        w = self.pid(error)
        
        cmd = Twist()
        cmd.linear.x = self.v_lineal
        cmd.angular.z = float(w)
        self.cmd_vel_pub.publish(cmd)


def main():
    # tanta cosa es solo para que cuando haya ctrl + c se pitee y no quede dando vuelta
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = FollowTheCarrotSimple()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cmd = Twist()
        node.cmd_vel_pub.publish(cmd)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()