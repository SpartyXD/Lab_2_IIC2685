#!/usr/bin/env python3
"""Ventana en vivo para Follow The Carrot.

Dibuja la ruta original (gris), el rastro real (azul), el robot y la
zanahoria (rojo). En el titulo muestra dos errores:
  - ECM total: todo el recorrido del robot.
  - ECM desde la carrot: desde que la zanahoria empieza a moverse, asi no
    cuenta el tramo desde donde parte el robot hasta la ruta.
ECM = promedio de (distancia minima del robot a la ruta)^2.

Al cerrar la ventana (o con Ctrl+C) guarda UNA imagen:
    <output_dir>/<run_name>_<fecha>.png

Parametros:
    run_name    nombre de la corrida (va en el nombre de la imagen)
    output_dir  carpeta de salida. Por defecto 'imagenes', dentro del
                proyecto (src/lab2/imagenes). Una ruta relativa se interpreta
                desde la raiz del proyecto; una absoluta se usa tal cual.
    lookahead, kp, ki, kd   solo informativos: salen en el titulo.
"""
import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from ament_index_python.packages import get_package_prefix
from geometry_msgs.msg import PointStamped
from matplotlib.figure import Figure
from nav_msgs.msg import Odometry, Path
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

PASO_MIN = 0.01       # m minimos entre puntos del rastro
REFRESCO = 0.1        # s entre redibujados
CARROT_MOVIO = 0.01   # m que debe moverse la zanahoria para decir que partio
PARAMETROS = ('lookahead', 'kp', 'ki', 'kd')


def carpeta_proyecto():
    """Raiz del paquete lab2 en src/ (donde estan package.xml y nodes/)."""
    aqui = os.path.dirname(os.path.realpath(__file__))
    candidatos = [os.path.dirname(aqui)]
    try:
        prefijo = get_package_prefix('lab2')   # ~/ros2_ws/install/lab2
        ws = os.path.dirname(os.path.dirname(prefijo))
        candidatos.append(os.path.join(ws, 'src', 'lab2'))
    except Exception:  # noqa: BLE001
        pass
    for candidato in candidatos:
        if os.path.isfile(os.path.join(candidato, 'package.xml')):
            return candidato
    return os.getcwd()


def distancia_a_ruta(a, ab, largo2, p):
    """Distancia minima del punto p a la polilinea (segmentos a -> a + ab)."""
    t = np.clip(np.einsum('ij,ij->i', p - a, ab) / largo2, 0.0, 1.0)
    proy = a + t[:, None] * ab
    return float(np.sqrt(np.min(np.sum((p - proy) ** 2, axis=1))))


def texto_ecm(nombre, d):
    if len(d) == 0:
        return f'{nombre}: sin datos'
    mse = float(np.mean(np.square(d)))
    return f'{nombre} = {mse:.4f} m$^2$ (RMSE = {math.sqrt(mse):.3f} m)'


class TrajectoryViewer(Node):
    def __init__(self):
        super().__init__('trajectory_viewer')
        self.declare_parameter('run_name', 'run')
        self.declare_parameter('output_dir', 'imagenes')
        libre = ParameterDescriptor(dynamic_typing=True)
        for nombre in PARAMETROS:
            self.declare_parameter(nombre, 0.0, libre)

        self.run_name = str(self.get_parameter('run_name').value)
        self.parametros = {
            n: float(self.get_parameter(n).value) for n in PARAMETROS
        }

        salida = os.path.expanduser(self.get_parameter('output_dir').value)
        if not os.path.isabs(salida):
            salida = os.path.join(carpeta_proyecto(), salida)
        self.output_dir = salida
        self.get_logger().info(f'La imagen se guarda en {self.output_dir}')

        # Ruta y sus segmentos
        self.ruta = None
        self.seg_a = None
        self.seg_ab = None
        self.seg_l2 = None

        # Rastro real, distancia a la ruta y tiempo de cada punto
        self.traza = []
        self.dist = []
        self.tiempos = []

        # Zanahoria: posicion actual, primera posicion y punto del rastro
        # en que empezo a moverse
        self.carrot = None
        self.carrot_ini = None
        self.i_carrot = None

        self.create_subscription(Path, 'nav_plan', self.path_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(PointStamped, '/carrot', self.carrot_cb, 10)

        plt.ion()
        self.fig, (self.ax, self.ax_err) = plt.subplots(1, 2, figsize=(14, 6))

    # Callbacks
    def path_cb(self, msg):
        pts = [[p.pose.position.x, p.pose.position.y] for p in msg.poses]
        if len(pts) < 2:
            return
        self.ruta = np.array(pts, dtype=float)
        self.seg_a = self.ruta[:-1]
        self.seg_ab = self.ruta[1:] - self.seg_a
        self.seg_l2 = np.maximum(
            np.einsum('ij,ij->i', self.seg_ab, self.seg_ab), 1e-12
        )
        # Una ruta nueva empieza todo de nuevo
        self.traza = []
        self.dist = []
        self.tiempos = []
        self.carrot = None
        self.carrot_ini = None
        self.i_carrot = None
        self.get_logger().info(f'Ruta recibida ({len(pts)} puntos)')

    def odom_cb(self, msg):
        if self.ruta is None:
            return
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        if self.traza and math.hypot(
            x - self.traza[-1][0], y - self.traza[-1][1]
        ) < PASO_MIN:
            return
        self.tiempos.append(self.get_clock().now().nanoseconds / 1e9)
        self.traza.append((x, y))
        self.dist.append(distancia_a_ruta(
            self.seg_a, self.seg_ab, self.seg_l2, np.array([x, y])
        ))

    def carrot_cb(self, msg):
        c = (msg.point.x, msg.point.y)
        self.carrot = c
        if self.carrot_ini is None:
            self.carrot_ini = c
        elif self.i_carrot is None and math.hypot(
            c[0] - self.carrot_ini[0], c[1] - self.carrot_ini[1]
        ) > CARROT_MOVIO:
            # Desde el proximo punto del rastro cuenta el segundo error
            self.i_carrot = len(self.traza)
            self.get_logger().info('La carrot empezo a moverse: mido el ECM desde aqui')

    # Dibujo (sirve para la ventana en vivo y para la imagen final)
    def titulo(self):
        p = self.parametros
        t = (f'Follow The Carrot: {self.run_name}\n'
             f'lookahead={p["lookahead"]:g}  kp={p["kp"]:g}  '
             f'ki={p["ki"]:g}  kd={p["kd"]:g}\n')
        t += texto_ecm('ECM total', self.dist)
        if self.i_carrot is None:
            t += '\nECM desde la carrot: aun no empieza'
        else:
            t += '\n' + texto_ecm('ECM desde la carrot',
                                  self.dist[self.i_carrot:])
        return t

    def pintar(self, ax):
        puntos = []
        if self.ruta is not None:
            ax.plot(self.ruta[:, 0], self.ruta[:, 1], '--', color='gray',
                    lw=2, label='Trayectoria original')
            puntos.append(self.ruta)
        if self.traza:
            xy = np.array(self.traza)
            ax.plot(xy[:, 0], xy[:, 1], '-', color='tab:blue', lw=1.5,
                    label='Trayectoria real')
            ax.plot(xy[-1, 0], xy[-1, 1], 'o', color='tab:blue', ms=9,
                    label='Robot')
            if self.i_carrot is not None and self.i_carrot < len(xy):
                ax.plot(xy[self.i_carrot, 0], xy[self.i_carrot, 1], '^',
                        color='k', ms=9, label='La carrot empieza a moverse')
            puntos.append(xy)
        if self.carrot is not None:
            ax.plot(self.carrot[0], self.carrot[1], '*', color='tab:red',
                    ms=14, label='Zanahoria')

        if puntos:
            todos = np.vstack(puntos)
            ax.set_xlim(todos[:, 0].min() - 0.5, todos[:, 0].max() + 0.5)
            ax.set_ylim(todos[:, 1].min() - 0.5, todos[:, 1].max() + 0.5)
        ax.set_title(self.titulo(), fontsize=9)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_aspect('equal')
        ax.grid(True)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc='best', fontsize=8)

    def pintar_error(self, ax):
        """Distancia a la ruta en el tiempo, con el tiempo partiendo en 0."""
        if self.dist:
            t = np.array(self.tiempos) - self.tiempos[0]
            ax.plot(t, self.dist, color='tab:red', lw=1.2,
                    label='Distancia a la ruta')
            if self.i_carrot is not None and self.i_carrot < len(t):
                ax.axvline(t[self.i_carrot], color='k', ls=':', lw=1,
                           label='La carrot empieza a moverse')
        ax.set_title('Error en el tiempo', fontsize=9)
        ax.set_xlabel('Tiempo [s]')
        ax.set_ylabel('Distancia a la ruta [m]')
        ax.grid(True)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc='best', fontsize=8)

    def refresh(self):
        self.ax.cla()
        self.pintar(self.ax)
        self.ax_err.cla()
        self.pintar_error(self.ax_err)

    def guardar(self):
        if not self.traza:
            self.get_logger().warn('No hay rastro que guardar.')
            return

        os.makedirs(self.output_dir, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archivo = os.path.join(self.output_dir, f'{self.run_name}_{stamp}.png')

        # Figura aparte: la de la ventana puede estar ya cerrada
        fig = Figure(figsize=(14, 6))
        a1, a2 = fig.subplots(1, 2)
        self.pintar(a1)
        self.pintar_error(a2)
        fig.tight_layout()
        fig.savefig(archivo, dpi=150, bbox_inches='tight')

        self.get_logger().info(f'Imagen guardada en {archivo}')
        self.get_logger().info(texto_ecm('ECM total', self.dist))
        if self.i_carrot is not None:
            self.get_logger().info(
                texto_ecm('ECM desde la carrot', self.dist[self.i_carrot:])
            )


def main():
    # Con NO, Ctrl+C llega como KeyboardInterrupt y alcanzamos a guardar
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = TrajectoryViewer()
    ultimo = 0.0
    try:
        while rclpy.ok() and plt.fignum_exists(node.fig.number):
            rclpy.spin_once(node, timeout_sec=0.02)
            ahora = node.get_clock().now().nanoseconds / 1e9
            if ahora - ultimo >= REFRESCO:
                ultimo = ahora
                node.refresh()
            plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.guardar()
        except Exception as exc:  # noqa: BLE001
            node.get_logger().error(f'No se pudo guardar: {exc}')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()