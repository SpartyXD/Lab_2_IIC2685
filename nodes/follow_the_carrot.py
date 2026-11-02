#!/usr/bin/env python3
"""Follow The Carrot, etapas 1 a 4.

El nodo:
  - recibe la ruta por nav_plan y precalcula el largo de arco acumulado,
  - guarda la pose del robot desde /odom,
  - busca el punto mas cercano de la ruta (solo hacia adelante),
  - ubica la zanahoria `lookahead` metros mas adelante sobre la ruta,
  - controla la orientacion con un PID: referencia = angulo hacia la
    zanahoria, salida = yaw del robot, actuacion = velocidad angular,
  - avanza con velocidad lineal fija y se detiene al llegar al final.
"""
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions


def normalizar(a):
    # Lleva cualquier angulo a (-pi, pi)
    return math.atan2(math.sin(a), math.cos(a))


class FollowTheCarrot(Node):
    def __init__(self):
        super().__init__('follow_the_carrot')

        # Parametros
        self.declare_parameter('lookahead', 0.3)
        self.declare_parameter('ventana', 50)
        self.declare_parameter('kp', 1.0)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.0)
        self.declare_parameter('v_lineal', 0.1)
        self.declare_parameter('max_w', 1.0)
        self.declare_parameter('eps_final', 0.1)
        self.declare_parameter('control_hz', 20.0)

        self.lookahead = float(self.get_parameter('lookahead').value)
        self.ventana = int(self.get_parameter('ventana').value)
        self.kp = float(self.get_parameter('kp').value)
        self.ki = float(self.get_parameter('ki').value)
        self.kd = float(self.get_parameter('kd').value)
        self.v_lineal = float(self.get_parameter('v_lineal').value)
        self.max_w = float(self.get_parameter('max_w').value)
        self.eps_final = float(self.get_parameter('eps_final').value)
        control_hz = float(self.get_parameter('control_hz').value)

        # Ruta: puntos (N, 2) y largo de arco acumulado (N,)
        self.ruta = None
        self.arco = None
        self.largo_total = 0.0
        self.idx = 0          # segmento del punto mas cercano (solo avanza)
        self.s_cercano = 0.0  # largo de arco del punto mas cercano

        # Pose del robot segun /odom
        self.x = None
        self.y = None
        self.yaw = None

        # Estado del PID
        self.integral = 0.0
        self.error_previo = 0.0
        self.t_previo = None
        self.terminado = False

        self.create_subscription(Path, 'nav_plan', self.path_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.carrot_pub = self.create_publisher(PointStamped, '/carrot', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(1.0 / control_hz, self.control_step)

        self.get_logger().info(
            f'Iniciando: lookahead={self.lookahead} ventana={self.ventana} '
            f'kp={self.kp} ki={self.ki} kd={self.kd} v={self.v_lineal} '
            f'max_w={self.max_w}'
        )

    # Callbacks
    def path_cb(self, msg):
        puntos = np.array(
            [[p.pose.position.x, p.pose.position.y] for p in msg.poses],
            dtype=float,
        )
        if len(puntos) < 2:
            self.get_logger().warn('Ruta con menos de 2 puntos, la ignoro.')
            return

        # arco[i] = distancia sobre la ruta desde el punto 0 hasta el punto i
        tramos = np.linalg.norm(np.diff(puntos, axis=0), axis=1)
        self.arco = np.concatenate(([0.0], np.cumsum(tramos)))
        self.ruta = puntos
        self.largo_total = float(self.arco[-1])
        self.idx = 0
        self.s_cercano = 0.0
        self.reiniciar_pid()
        self.terminado = False

        self.get_logger().info(
            f'Ruta recibida con {len(puntos)} puntos, '
            f'largo total {self.largo_total:.2f} m'
        )

    def odom_cb(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        self.x = pos.x
        self.y = pos.y
        # En 2D basta con z y w del cuaternion
        self.yaw = normalizar(2.0 * math.atan2(ori.z, ori.w))

    # Geometria
    def buscar_cercano(self):
        """Punto de la ruta mas cercano al robot, buscando solo hacia adelante.

        Proyecta al robot sobre los segmentos de una ventana que parte en el
        segmento anterior. Devuelve (largo de arco del punto, distancia).
        """
        n = len(self.ruta)
        i0 = self.idx
        i1 = min(i0 + self.ventana, n - 1)

        a = self.ruta[i0:i1]
        b = self.ruta[i0 + 1:i1 + 1]
        ab = b - a
        largo2 = np.einsum('ij,ij->i', ab, ab)
        p = np.array([self.x, self.y])

        t = np.einsum('ij,ij->i', p - a, ab) / np.maximum(largo2, 1e-12)
        t = np.clip(t, 0.0, 1.0)
        proy = a + t[:, None] * ab
        dist = np.linalg.norm(proy - p, axis=1)

        k = int(np.argmin(dist))
        self.idx = i0 + k
        s = float(self.arco[i0 + k] + t[k] * math.sqrt(largo2[k]))
        # La zanahoria nunca retrocede
        self.s_cercano = max(self.s_cercano, s)
        return self.s_cercano, float(dist[k])

    def punto_en(self, s):
        """Punto (x, y) de la ruta a largo de arco s (interpolado)."""
        s = min(max(s, 0.0), self.largo_total)
        j = int(np.searchsorted(self.arco, s, side='right')) - 1
        j = min(max(j, 0), len(self.ruta) - 2)
        seg = self.arco[j + 1] - self.arco[j]
        t = 0.0 if seg < 1e-12 else (s - self.arco[j]) / seg
        return self.ruta[j] + t * (self.ruta[j + 1] - self.ruta[j])

    def calcular_zanahoria(self):
        """Devuelve (xc, yc, theta_carrot, dist_ruta, al_final)."""
        s_cerca, dist_ruta = self.buscar_cercano()
        s_carrot = min(s_cerca + self.lookahead, self.largo_total)
        xc, yc = self.punto_en(s_carrot)
        theta = math.atan2(yc - self.y, xc - self.x)
        al_final = s_carrot >= self.largo_total - 1e-9
        return float(xc), float(yc), theta, dist_ruta, al_final

    # Controlador PID de orientacion
    def reiniciar_pid(self):
        self.integral = 0.0
        self.error_previo = 0.0
        self.t_previo = None

    def pid(self, error):
        """PID sobre el error angular. Devuelve la velocidad angular."""
        ahora = self.get_clock().now().nanoseconds / 1e9
        if self.t_previo is None:
            dt = 0.0
            derivada = 0.0
        else:
            dt = ahora - self.t_previo
            derivada = (error - self.error_previo) / dt if dt > 0 else 0.0
        self.t_previo = ahora
        self.error_previo = error

        # Integral con anti-windup: ki * integral nunca supera max_w
        if dt > 0 and self.ki > 0:
            self.integral += error * dt
            limite = self.max_w / self.ki
            self.integral = max(-limite, min(limite, self.integral))

        u = self.kp * error + self.ki * self.integral + self.kd * derivada
        return max(-self.max_w, min(self.max_w, u))

    # Actuacion
    def publicar_velocidad(self, v, w):
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        self.cmd_vel_pub.publish(cmd)

    def detener(self):
        self.publicar_velocidad(0.0, 0.0)

    def detener_final(self):
        # Varias veces y con una pausa, para que el cero alcance a salir
        for _ in range(5):
            self.detener()
            time.sleep(0.05)

    # Control
    def listo(self):
        return self.ruta is not None and self.x is not None

    def control_step(self):
        if self.terminado:
            return

        if not self.listo():
            self.get_logger().info(
                'Esperando ruta (nav_plan) y odometria (/odom)...',
                throttle_duration_sec=2.0,
            )
            return

        xc, yc, theta_carrot, dist_ruta, al_final = self.calcular_zanahoria()

        # Publicar la zanahoria para verla (ros2 topic echo /carrot)
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.point.x = xc
        msg.point.y = yc
        self.carrot_pub.publish(msg)

        # Etapa 4: condicion de termino
        dist_final = math.hypot(
            self.x - self.ruta[-1][0], self.y - self.ruta[-1][1]
        )
        if al_final and dist_final < self.eps_final:
            self.detener()
            self.terminado = True
            self.get_logger().info(
                f'Trayectoria terminada: a {dist_final:.3f} m del punto final.'
            )
            return

        # Etapa 3: PID de orientacion
        # referencia r(t) = theta_carrot, salida y(t) = yaw, u(t) = w
        error = normalizar(theta_carrot - self.yaw)
        w = self.pid(error)
        self.publicar_velocidad(self.v_lineal, w)

        self.get_logger().info(
            f'robot=({self.x:.2f}, {self.y:.2f}, yaw={self.yaw:.2f}) '
            f'carrot=({xc:.2f}, {yc:.2f}) theta_carrot={theta_carrot:.2f} '
            f'error_ang={error:.2f} w={w:.2f} dist_ruta={dist_ruta:.2f} '
            f'dist_final={dist_final:.2f} seg={self.idx}',
            throttle_duration_sec=0.5,
        )


def main():
    # Con NO, Ctrl+C llega como KeyboardInterrupt y el contexto sigue vivo
    # en el finally, asi que alcanzamos a detener el robot.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = FollowTheCarrot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.detener_final()
        except Exception:  # noqa: BLE001
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()