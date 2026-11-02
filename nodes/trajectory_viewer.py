#!/usr/bin/env python3
"""Ventana en vivo y analisis de error para Follow The Carrot.

Mientras corre el robot dibuja:
  - a la izquierda: ruta original (gris), rastro real (azul), robot y
    zanahoria (rojo),
  - a la derecha: distancia del robot a la ruta en el tiempo.

Al cerrar la ventana (o con Ctrl+C) guarda en output_dir:
  <run_name>_<fecha>.csv     columnas t, x, y, d (d = distancia a la ruta)
  <run_name>_<fecha>.png     trayectoria real vs original + error en el tiempo
  tabla_corridas.csv         una fila por corrida: parametros y error
  mejor_<ruta>.png           copia del grafico si es la mejor corrida de esa ruta
  resumen_errores.png        las mejores corridas de todas las rutas juntas

Error: para cada punto real, distancia minima a la polilinea de la ruta.
MSE = promedio de esas distancias al cuadrado; RMSE = su raiz.

Parametros:
    run_name    nombre de la corrida. Debe empezar con el nombre del txt de la
                ruta (por ejemplo path_sin_L03_kp10) para asociarla a su ruta.
    output_dir  carpeta de salida. Por defecto 'imagenes', dentro del
                proyecto (src/lab2/imagenes). Una ruta relativa se interpreta
                desde la raiz del proyecto; una absoluta se usa tal cual.
    lookahead, kp, ki, kd   solo informativos: se anotan en la tabla.
"""
import csv
import glob
import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from ament_index_python.packages import get_package_prefix
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry, Path
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

PASO_MIN = 0.01   # m minimos entre puntos del rastro
REFRESCO = 0.1    # s entre redibujados
PARAMETROS = ('lookahead', 'kp', 'ki', 'kd')
COLUMNAS = ['ruta', 'corrida', 'lookahead', 'kp', 'ki', 'kd',
            'mse_m2', 'rmse_m', 'max_m', 'tiempo_s', 'archivo_csv']


# Utilidades
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


def cargar_ruta(archivo):
    puntos = []
    with open(archivo, encoding='utf-8') as f:
        for linea in f:
            partes = linea.strip().strip('()').replace(',', ' ').split()
            try:
                nums = [float(p) for p in partes]
            except ValueError:
                continue
            if len(nums) >= 2:
                puntos.append(nums[:2])
    return np.array(puntos, dtype=float)


def distancia_a_ruta(a, ab, largo2, p):
    """Distancia minima del punto p a la polilinea (segmentos a -> a + ab)."""
    t = np.clip(np.einsum('ij,ij->i', p - a, ab) / largo2, 0.0, 1.0)
    proy = a + t[:, None] * ab
    return float(np.sqrt(np.min(np.sum((p - proy) ** 2, axis=1))))


def metricas_desde(t, d):
    d = np.asarray(d, dtype=float)
    mse = float(np.mean(d ** 2))
    return {
        'mse': mse,
        'rmse': math.sqrt(mse),
        'max': float(d.max()),
        'dur': float(t[-1] - t[0]),
    }


def leer_tabla(archivo):
    if not os.path.isfile(archivo):
        return []
    with open(archivo, newline='', encoding='utf-8') as f:
        return [fila for fila in csv.DictReader(f) if fila.get('archivo_csv')]


# Graficos
def dibujar_trayectoria(ax, titulo, ruta, xy, m, leyenda=False):
    ax.plot(ruta[:, 0], ruta[:, 1], '--', color='gray', lw=2,
            label='Trayectoria original')
    ax.plot(xy[:, 0], xy[:, 1], '-', color='tab:blue', lw=1.5,
            label='Trayectoria real')
    ax.plot(*xy[0], 'o', color='tab:green', label='Inicio')
    ax.plot(*xy[-1], 's', color='tab:red', label='Fin')
    ax.set_title(f'{titulo}\nMSE = {m["mse"]:.4f} m$^2$  '
                 f'(RMSE = {m["rmse"]:.3f} m)')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_aspect('equal')
    ax.grid(True)
    if leyenda:
        ax.legend(fontsize=8, loc='best')


def dibujar_error(ax, t, d, m):
    ax.plot(t, d, color='tab:red', lw=1.2)
    ax.axhline(m['rmse'], color='k', ls='--', lw=1,
               label=f'RMSE = {m["rmse"]:.3f} m')
    ax.set_title(f'Error en el tiempo (max = {m["max"]:.3f} m)')
    ax.set_xlabel('Tiempo [s]')
    ax.set_ylabel('Distancia a la ruta [m]')
    ax.grid(True)
    ax.legend(fontsize=8, loc='best')


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

        self.proyecto = carpeta_proyecto()
        self.config_dir = os.path.join(self.proyecto, 'config')
        salida = os.path.expanduser(self.get_parameter('output_dir').value)
        if not os.path.isabs(salida):
            salida = os.path.join(self.proyecto, salida)
        self.output_dir = salida
        self.get_logger().info(f'Las corridas se guardan en {self.output_dir}')

        # Nombres de rutas conocidas (config/path_*.txt)
        self.rutas_conocidas = sorted(
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(self.config_dir, 'path_*.txt'))
        )

        # Ruta actual y segmentos precalculados
        self.ruta = None
        self.seg_a = None
        self.seg_ab = None
        self.seg_l2 = None

        # Rastro real: (t, x, y), distancia a la ruta y suma de d^2
        self.traza = []
        self.dist = []
        self.suma_d2 = 0.0
        self.carrot = None

        self.create_subscription(Path, 'nav_plan', self.path_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(PointStamped, '/carrot', self.carrot_cb, 10)

        plt.ion()
        self.fig, (self.ax, self.ax_err) = plt.subplots(
            1, 2, figsize=(14, 6)
        )
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

        (self.l_err,) = self.ax_err.plot([], [], color='tab:red', lw=1.2)
        self.l_rmse = self.ax_err.axhline(0.0, color='k', ls='--', lw=1)
        self.ax_err.set_xlabel('Tiempo [s]')
        self.ax_err.set_ylabel('Distancia a la ruta [m]')
        self.ax_err.set_title('Error en el tiempo')
        self.ax_err.grid(True)
        self.fig.tight_layout()

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
        # Una ruta nueva empieza un rastro nuevo
        self.traza = []
        self.dist = []
        self.suma_d2 = 0.0
        self.get_logger().info(f'Ruta recibida ({len(pts)} puntos)')

    def odom_cb(self, msg):
        if self.ruta is None:
            return
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        if self.traza and math.hypot(
            x - self.traza[-1][1], y - self.traza[-1][2]
        ) < PASO_MIN:
            return

        t = self.get_clock().now().nanoseconds / 1e9
        d = distancia_a_ruta(
            self.seg_a, self.seg_ab, self.seg_l2, np.array([x, y])
        )
        self.traza.append((t, x, y))
        self.dist.append(d)
        self.suma_d2 += d * d

    def carrot_cb(self, msg):
        self.carrot = (msg.point.x, msg.point.y)

    # Dibujo en vivo
    def metricas_en_vivo(self):
        n = len(self.dist)
        if n < 2:
            return None
        mse = self.suma_d2 / n
        return {'mse': mse, 'rmse': math.sqrt(mse), 'max': max(self.dist)}

    def refresh(self):
        if self.ruta is not None:
            self.l_ruta.set_data(self.ruta[:, 0], self.ruta[:, 1])

        titulo = f'Follow The Carrot: {self.run_name}'
        if self.traza:
            arr = np.array(self.traza)
            self.l_traza.set_data(arr[:, 1], arr[:, 2])
            self.p_robot.set_data([arr[-1, 1]], [arr[-1, 2]])
            self.l_err.set_data(arr[:, 0] - arr[0, 0], self.dist)

            m = self.metricas_en_vivo()
            if m:
                self.l_rmse.set_ydata([m['rmse'], m['rmse']])
                titulo += (f'\nMSE = {m["mse"]:.4f} m$^2$  '
                           f'(RMSE = {m["rmse"]:.3f} m)')
            self.ax_err.relim()
            self.ax_err.autoscale_view()
        self.ax.set_title(titulo)

        if self.carrot is not None:
            self.p_carrot.set_data([self.carrot[0]], [self.carrot[1]])

        # Limites de la trayectoria: ruta + rastro + margen
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

    # Analisis y guardado
    def nombre_ruta(self):
        # El nombre mas largo primero, por si uno es prefijo de otro
        for nombre in sorted(self.rutas_conocidas, key=len, reverse=True):
            if self.run_name.startswith(nombre):
                return nombre
        return self.run_name

    def guardar(self):
        if len(self.traza) < 2 or self.ruta is None:
            self.get_logger().warn('No hay rastro que guardar.')
            return

        os.makedirs(self.output_dir, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        corrida = f'{self.run_name}_{stamp}'
        base = os.path.join(self.output_dir, corrida)

        arr = np.array(self.traza)
        t = arr[:, 0] - arr[0, 0]
        xy = arr[:, 1:3]
        d = np.array(self.dist)
        m = metricas_desde(t, d)

        # CSV del rastro (con la distancia a la ruta de cada punto)
        with open(base + '.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['t', 'x', 'y', 'd'])
            for ti, (xi, yi), di in zip(t, xy, d):
                w.writerow([f'{ti:.3f}', f'{xi:.4f}', f'{yi:.4f}', f'{di:.5f}'])

        # Grafico de la corrida
        ruta_nombre = self.nombre_ruta()
        p = self.parametros
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))
        dibujar_trayectoria(a1, ruta_nombre, self.ruta, xy, m, leyenda=True)
        dibujar_error(a2, t, d, m)
        fig.suptitle(
            f'Corrida {corrida}   lookahead={p["lookahead"]:g}  '
            f'kp={p["kp"]:g}  ki={p["ki"]:g}  kd={p["kd"]:g}', fontsize=9
        )
        fig.tight_layout()
        fig.savefig(base + '.png', dpi=150, bbox_inches='tight')

        # Tabla de corridas
        tabla = os.path.join(self.output_dir, 'tabla_corridas.csv')
        nuevo = not os.path.isfile(tabla)
        with open(tabla, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=COLUMNAS)
            if nuevo:
                w.writeheader()
            w.writerow({
                'ruta': ruta_nombre,
                'corrida': corrida,
                'lookahead': p['lookahead'],
                'kp': p['kp'],
                'ki': p['ki'],
                'kd': p['kd'],
                'mse_m2': f'{m["mse"]:.6f}',
                'rmse_m': f'{m["rmse"]:.4f}',
                'max_m': f'{m["max"]:.4f}',
                'tiempo_s': f'{m["dur"]:.1f}',
                'archivo_csv': corrida + '.csv',
            })

        # Si es la mejor corrida de su ruta, guardar copia del grafico
        filas = [f for f in leer_tabla(tabla) if f['ruta'] == ruta_nombre]
        mejor = min(filas, key=lambda f: float(f['mse_m2']))
        if mejor['corrida'] == corrida:
            fig.savefig(
                os.path.join(self.output_dir, f'mejor_{ruta_nombre}.png'),
                dpi=150, bbox_inches='tight'
            )
            self.get_logger().info(f'Mejor corrida hasta ahora de {ruta_nombre}')
        plt.close(fig)

        self.get_logger().info(
            f'Guardado {base}.csv y .png | '
            f'MSE={m["mse"]:.4f} m2  RMSE={m["rmse"]:.3f} m  '
            f'max={m["max"]:.3f} m  t={m["dur"]:.1f} s'
        )
        self.generar_resumen(tabla)

    def generar_resumen(self, tabla):
        """Mejor corrida de cada ruta, todas juntas en una imagen."""
        mejores = {}
        for fila in leer_tabla(tabla):
            r = fila['ruta']
            if r not in self.rutas_conocidas:
                continue
            if r not in mejores or float(fila['mse_m2']) < float(
                mejores[r]['mse_m2']
            ):
                mejores[r] = fila
        nombres = [r for r in self.rutas_conocidas if r in mejores]
        if not nombres:
            return

        fig, axs = plt.subplots(
            2, len(nombres), figsize=(5.5 * len(nombres), 9), squeeze=False
        )
        for i, nombre in enumerate(nombres):
            try:
                ruta = cargar_ruta(
                    os.path.join(self.config_dir, nombre + '.txt')
                )
                datos = np.loadtxt(
                    os.path.join(self.output_dir, mejores[nombre]['archivo_csv']),
                    delimiter=',', skiprows=1, ndmin=2,
                )
            except (OSError, ValueError) as exc:
                self.get_logger().warn(f'Resumen: salto {nombre} ({exc})')
                axs[0][i].axis('off')
                axs[1][i].axis('off')
                continue
            t, xy, d = datos[:, 0], datos[:, 1:3], datos[:, 3]
            m = metricas_desde(t, d)
            dibujar_trayectoria(axs[0][i], nombre, ruta, xy, m, leyenda=(i == 0))
            dibujar_error(axs[1][i], t, d, m)

        fig.suptitle('Follow The Carrot: mejor corrida de cada trayectoria')
        fig.tight_layout()
        salida = os.path.join(self.output_dir, 'resumen_errores.png')
        fig.savefig(salida, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.get_logger().info(f'Resumen actualizado en {salida}')


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