#!/usr/bin/env python3
"""Ventana en vivo para Follow The Carrot.

Dibuja la ruta objetivo (gris), el rastro real del robot (azul), el robot y
la zanahoria (rojo). En el titulo muestra el error cuadratico medio entre el
rastro real y la ruta.

Al cerrar la ventana (o con Ctrl+C) guarda en output_dir:
  <run_name>_<fecha>.csv   columnas t, x, y del rastro real
  <run_name>_<fecha>.png   imagen del grafico, lista para las slides

Parametros:
    run_name    prefijo de los archivos (por ejemplo, el nombre de la ruta)
    output_dir  carpeta de salida. Por defecto 'imagenes', dentro del
                proyecto (src/lab2/imagenes). Una ruta relativa se interpreta
                desde la raiz del proyecto; una absoluta se usa tal cual.
"""
import csv
import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from ament_index_python.packages import get_package_prefix
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

PASO_MIN = 0.01        # m minimos entre puntos del rastro
REFRESCO = 0.1         # s entre redibujados
REFRESCO_ERROR = 1.0   # s entre calculos del error en vivo
MAX_PUNTOS_ERROR = 500


def carpeta_proyecto():
    """Raiz del paquete lab2 en src/ (donde estan package.xml y nodes/).

    Funciona con colcon build normal y con --symlink-install. Si no la
    encuentra, usa la carpeta desde donde se lanzo el nodo.
    """
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


def error_cuadratico(ruta, traza):
    """Error cuadratico medio: promedio de (distancia minima a la ruta)^2."""
    a = ruta[:-1]
    ab = ruta[1:] - a
    largo2 = np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-12)

    p = np.asarray(traza, dtype=float)
    ap = p[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum('msj,sj->ms', ap, ab) / largo2[None, :], 0.0, 1.0)
    proy = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    d2 = np.sum((p[:, None, :] - proy) ** 2, axis=2)
    return float(d2.min(axis=1).mean())


class TrajectoryViewer(Node):
    def __init__(self):
        super().__init__('trajectory_viewer')
        self.declare_parameter('run_name', 'run')
        self.declare_parameter('output_dir', 'imagenes')
        self.run_name = self.get_parameter('run_name').value
        salida = os.path.expanduser(self.get_parameter('output_dir').value)
        if not os.path.isabs(salida):
            salida = os.path.join(carpeta_proyecto(), salida)
        self.output_dir = salida
        self.get_logger().info(f'Las corridas se guardan en {self.output_dir}')

        self.ruta = None
        self.traza = []      # (t, x, y)
        self.carrot = None
        self.error_txt = ''
        self.t_error = 0.0

        self.create_subscription(Path, 'nav_plan', self.path_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(PointStamped, '/carrot', self.carrot_cb, 10)

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 7))
        (self.l_ruta,) = self.ax.plot([], [], '--', color='gray', lw=2,
                                      label='Trayectoria original')
        (self.l_traza,) = self.ax.plot([], [], '-', color='tab:blue', lw=1.5,
                                       label='Trayectoria real')
        (self.p_robot,) = self.ax.plot([], [], 'o', color='tab:blue', ms=9,
                                       label='Robot')
        (self.p_carrot,) = self.ax.plot([], [], '*', color='tab:red', ms=14,
                                        label='Zanahoria')
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.legend(loc='best', fontsize=8)
        self.ax.set_title(f'Follow The Carrot: {self.run_name}')

    # Callbacks
    def path_cb(self, msg):
        pts = [[p.pose.position.x, p.pose.position.y] for p in msg.poses]
        if len(pts) >= 2:
            self.ruta = np.array(pts, dtype=float)
            self.traza = []
            self.get_logger().info(f'Ruta recibida ({len(pts)} puntos)')

    def odom_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        t = self.get_clock().now().nanoseconds / 1e9
        if not self.traza or math.hypot(
            x - self.traza[-1][1], y - self.traza[-1][2]
        ) >= PASO_MIN:
            self.traza.append((t, x, y))

    def carrot_cb(self, msg):
        self.carrot = (msg.point.x, msg.point.y)

    # Dibujo
    def refresh(self):
        if self.ruta is not None:
            self.l_ruta.set_data(self.ruta[:, 0], self.ruta[:, 1])

        if self.traza:
            arr = np.array(self.traza)
            self.l_traza.set_data(arr[:, 1], arr[:, 2])
            self.p_robot.set_data([arr[-1, 1]], [arr[-1, 2]])
        if self.carrot is not None:
            self.p_carrot.set_data([self.carrot[0]], [self.carrot[1]])

        # Limites: ruta + rastro + margen
        puntos = []
        if self.ruta is not None:
            puntos.append(self.ruta)
        if self.traza:
            puntos.append(np.array(self.traza)[:, 1:3])
        if puntos:
            todos = np.vstack(puntos)
            margen = 0.5
            self.ax.set_xlim(todos[:, 0].min() - margen,
                             todos[:, 0].max() + margen)
            self.ax.set_ylim(todos[:, 1].min() - margen,
                             todos[:, 1].max() + margen)

        self.actualizar_error()
        titulo = f'Follow The Carrot: {self.run_name}'
        if self.error_txt:
            titulo += f'\n{self.error_txt}'
        self.ax.set_title(titulo)

    def actualizar_error(self, forzar=False):
        ahora = self.get_clock().now().nanoseconds / 1e9
        if not forzar and ahora - self.t_error < REFRESCO_ERROR:
            return
        self.t_error = ahora
        if self.ruta is None or len(self.traza) < 2:
            return

        pts = np.array(self.traza)[:, 1:3]
        if not forzar and len(pts) > MAX_PUNTOS_ERROR:
            paso = len(pts) // MAX_PUNTOS_ERROR + 1
            pts = pts[::paso]
        mse = error_cuadratico(self.ruta, pts)
        self.error_txt = (
            f'Error cuadratico medio = {mse:.4f} m^2 '
            f'(RMSE = {math.sqrt(mse):.3f} m)'
        )

    # Guardado
    def guardar(self):
        if not self.traza:
            self.get_logger().warn('No hay rastro que guardar.')
            return

        os.makedirs(self.output_dir, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base = os.path.join(self.output_dir, f'{self.run_name}_{stamp}')

        t0 = self.traza[0][0]
        with open(base + '.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['t', 'x', 'y'])
            for t, x, y in self.traza:
                w.writerow([f'{t - t0:.3f}', f'{x:.4f}', f'{y:.4f}'])

        # Error final con todos los puntos
        self.actualizar_error(forzar=True)
        self.refresh()
        self.fig.savefig(base + '.png', dpi=150, bbox_inches='tight')
        self.get_logger().info(f'Guardado en {base}.csv y {base}.png')
        if self.error_txt:
            self.get_logger().info(self.error_txt)


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