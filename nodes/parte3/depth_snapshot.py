#!/usr/bin/env python3
"""Captura pares color/profundidad para inspeccionar los valores de depth.

SPACE guarda color.png, depth.npy (valores crudos), depth_vis.png y stats.txt.
Un clic sobre la ventana de color imprime la profundidad de ese píxel.
ESC o Q cierran el nodo.
"""

import os
from datetime import datetime

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class DepthSnapshot(Node):

    def __init__(self):
        super().__init__('depth_snapshot')

        self.color_topic = self.declare_parameter(
            'color_topic', '/image_raw'
        ).value
        self.depth_topic = self.declare_parameter(
            'depth_topic', '/depth/image_raw'
        ).value
        self.output_dir = os.path.expanduser(
            self.declare_parameter('output_dir', '~/depth_snapshots').value
        )
        self.display_width = int(
            self.declare_parameter('display_width', 960).value
        )

        self.bridge = CvBridge()
        self.color_bgr = None
        self.depth_raw = None
        self.depth_encoding = ''
        self.click_pixel = None
        self.capture_count = 0
        self.quit_requested = False
        self._color_scale = 1.0

        self.create_subscription(Image, self.color_topic, self.color_cb, 2)
        self.create_subscription(Image, self.depth_topic, self.depth_cb, 2)

        self.gui_available = self._init_windows()
        if self.gui_available:
            self.get_logger().info(
                f'Esperando {self.color_topic} y {self.depth_topic}. '
                'SPACE = capturar, ESC/Q = salir. '
                'Clic en la ventana de color = leer profundidad.'
            )
        else:
            self.get_logger().warning(
                'No se pudo abrir la ventana de OpenCV; se capturará el '
                'primer par recibido y el nodo terminará.'
            )

    def _init_windows(self):
        try:
            cv2.namedWindow('Color', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('Profundidad', cv2.WINDOW_AUTOSIZE)
        except cv2.error as exc:
            self.get_logger().warning(f'OpenCV sin GUI: {exc}')
            return False
        cv2.setMouseCallback('Color', self.mouse_cb)
        return True

    def color_cb(self, msg):
        try:
            self.color_bgr = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8'
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                f'No se pudo convertir la imagen de color: {exc}',
                throttle_duration_sec=5.0,
            )

    def depth_cb(self, msg):
        try:
            self.depth_raw = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='passthrough'
            )
            self.depth_encoding = msg.encoding
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                f'No se pudo convertir la imagen de profundidad: {exc}',
                throttle_duration_sec=5.0,
            )

    def depth_meters(self):
        """Devuelve la profundidad en metros asumiendo 16UC1 en milímetros."""
        depth = self.depth_raw.astype(np.float32)
        if self.depth_encoding in ('16UC1', 'mono16'):
            return depth / 1000.0
        return depth

    def unit_label(self):
        if self.depth_encoding in ('16UC1', 'mono16'):
            return 'mm (uint16, asumido)'
        return 'm (float, asumido)'

    def stats_lines(self):
        meters = self.depth_meters()
        valid = np.isfinite(meters) & (meters > 0.0)
        count = int(valid.sum())
        total = int(meters.size)

        lines = [
            f'color_topic: {self.color_topic}',
            f'depth_topic: {self.depth_topic}',
            f'encoding: {self.depth_encoding}',
            f'dtype: {self.depth_raw.dtype}',
            f'shape: {self.depth_raw.shape}',
            f'unidades: {self.unit_label()}',
            f'pixeles validos: {count}/{total} '
            f'({100.0 * count / total:.1f}%)',
        ]

        if count:
            values = meters[valid]
            labels = ['min', 'p1', 'p5', 'p25', 'mediana', 'p75',
                      'p95', 'p99', 'max']
            percentiles = np.percentile(
                values, [0, 1, 5, 25, 50, 75, 95, 99, 100]
            )
            for label, value in zip(labels, percentiles):
                lines.append(f'{label}: {value:.4f} m')

        alto, ancho = meters.shape[:2]
        cx, cy = ancho // 2, alto // 2
        raw_centro = self.depth_raw[cy, cx].item()
        lines.append(
            f'centro ({cx},{cy}): raw={raw_centro} '
            f'-> {meters[cy, cx]:.4f} m'
        )
        return lines

    def depth_visualization(self):
        meters = self.depth_meters()
        valid = np.isfinite(meters) & (meters > 0.0)
        vis = np.zeros((*meters.shape, 3), dtype=np.uint8)
        if not valid.any():
            return vis

        values = meters[valid]
        lo, hi = np.percentile(values, [2, 98])
        if hi <= lo:
            hi = lo + 1e-6

        norm = (meters - lo) / (hi - lo)
        norm = np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)
        norm = (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)
        vis = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        vis[~valid] = (0, 0, 0)
        cv2.putText(
            vis, f'{lo:.2f} - {hi:.2f} m (p2-p98)', (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
        return vis

    def save_matrix_txt(self, path):
        """Escribe la matriz cruda de profundidad en texto legible."""
        depth = self.depth_raw
        if depth.dtype.kind == 'f':
            precision = 9 if depth.dtype.itemsize <= 4 else 17
            fmt = f'%.{precision}g'
        elif depth.dtype.kind in 'iu':
            fmt = '%d'
        else:
            fmt = '%s'

        header = '\n'.join([
            f'topic: {self.depth_topic}',
            f'encoding: {self.depth_encoding}',
            f'dtype: {depth.dtype}',
            f'shape: {depth.shape}',
            f'unidades: {self.unit_label()}',
            'Una fila por fila de imagen; valores crudos sin convertir.',
            'NaN/Inf se escriben como nan/inf.',
        ])
        np.savetxt(
            path, depth, fmt=fmt, delimiter=' ', header=header,
            comments='# ',
        )

    def mouse_cb(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or self.depth_raw is None:
            return
        if self.color_bgr is None:
            return

        alto_c, ancho_c = self.color_bgr.shape[:2]
        x_color = x / self._color_scale
        y_color = y / self._color_scale
        x_color = min(max(x_color, 0.0), ancho_c - 1)
        y_color = min(max(y_color, 0.0), alto_c - 1)

        alto, ancho = self.depth_raw.shape[:2]
        px = int(round(x_color * ancho / ancho_c))
        py = int(round(y_color * alto / alto_c))
        px = min(max(px, 0), ancho - 1)
        py = min(max(py, 0), alto - 1)

        self.click_pixel = (px, py)
        raw = self.depth_raw[py, px].item()
        meters = self.depth_meters()[py, px]
        self.get_logger().info(
            f'Pixel ({px},{py}): raw={raw} ({self.unit_label()}) '
            f'-> {meters:.4f} m'
        )

    def capture(self):
        if self.color_bgr is None or self.depth_raw is None:
            self.get_logger().warning(
                'Aún no hay par color+profundidad; espera un momento.'
            )
            return

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        folder = os.path.join(
            self.output_dir, f'{stamp}_{self.capture_count:02d}'
        )
        self.capture_count += 1
        os.makedirs(folder, exist_ok=True)

        cv2.imwrite(os.path.join(folder, 'color.png'), self.color_bgr)
        np.save(os.path.join(folder, 'depth.npy'), self.depth_raw)
        self.save_matrix_txt(os.path.join(folder, 'depth.txt'))
        cv2.imwrite(
            os.path.join(folder, 'depth_vis.png'),
            self.depth_visualization(),
        )
        stats = '\n'.join(self.stats_lines()) + '\n'
        with open(
            os.path.join(folder, 'stats.txt'), 'w', encoding='utf-8'
        ) as stats_file:
            stats_file.write(stats)

        self.get_logger().info(f'Captura guardada en {folder}\n{stats}')

    def update_gui(self):
        if not self.gui_available:
            if self.color_bgr is not None and self.depth_raw is not None:
                self.capture()
                self.quit_requested = True
            return

        if self.color_bgr is not None:
            display = self.color_bgr.copy()
            ancho = display.shape[1]
            if ancho > self.display_width:
                self._color_scale = self.display_width / ancho
                display = cv2.resize(
                    display, None,
                    fx=self._color_scale, fy=self._color_scale,
                    interpolation=cv2.INTER_AREA,
                )
            else:
                self._color_scale = 1.0

            if self.click_pixel is not None and self.depth_raw is not None:
                px, py = self.click_pixel
                alto_d, ancho_d = self.depth_raw.shape[:2]
                alto_c, ancho_c = self.color_bgr.shape[:2]
                if px < ancho_d and py < alto_d:
                    dx = int(round(
                        px * ancho_c / ancho_d * self._color_scale
                    ))
                    dy = int(round(
                        py * alto_c / alto_d * self._color_scale
                    ))
                    raw = self.depth_raw[py, px].item()
                    meters = self.depth_meters()[py, px]
                    cv2.drawMarker(
                        display, (dx, dy), (0, 255, 0),
                        cv2.MARKER_CROSS, 20, 2,
                    )
                    cv2.putText(
                        display, f'{meters:.3f} m (raw {raw})',
                        (dx + 12, dy - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 2, cv2.LINE_AA,
                    )
            try:
                cv2.imshow('Color', display)
            except cv2.error as exc:
                self._gui_failed(exc)
                return

        if self.depth_raw is not None:
            try:
                cv2.imshow('Profundidad', self.depth_visualization())
            except cv2.error as exc:
                self._gui_failed(exc)
                return

        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            self.capture()
        elif key in (27, ord('q')):  # ESC o Q
            self.quit_requested = True

    def _gui_failed(self, exc):
        self.get_logger().warning(
            f'No se pudo mostrar la ventana ({exc}); '
            'se capturará el próximo par automáticamente.'
        )
        self.gui_available = False


def main():
    rclpy.init()
    node = DepthSnapshot()
    try:
        while rclpy.ok() and not node.quit_requested:
            rclpy.spin_once(node, timeout_sec=0.03)
            node.update_gui()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
