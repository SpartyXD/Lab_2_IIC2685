#!/usr/bin/env python3
"""Obtiene rangos HSV para un mask a partir de una región de /image_raw.

Controles:
    espacio    congela o descongela la imagen
    click izq  dibuja el borde del objeto sobre la imagen congelada
    r          borra el trazo y los rangos
    s          guarda los rangos y sobrescribe el archivo JSON
    a          fusiona los rangos con los del archivo y lo sobrescribe
    c          vuelve a copiar los rangos al portapapeles
    q / Esc    cierra

Al soltar el mouse, los rangos se amplían con un margen de seguridad y se
copian al portapapeles, pero no se guardan. Usa s o a para escribir
config/hsv_ranges.json. Reinicia detector_de_objeto.py para que cargue los
nuevos rangos.
"""

import os
import subprocess

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory as pkg_dir
from cv_bridge import CvBridge, CvBridgeError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from hsv_ranges import (
    build_mask,
    format_hsv_ranges,
    mask_ranges,
    pad_pairs,
    save_ranges,
)

MASK_WINDOW = 'Mask'
INSTRUCTIONS_LIVE = ('espacio: congelar  |  q: salir',)
INSTRUCTIONS_FROZEN = (
    'r: borrar  |  s: guardar  |  a: agregar  |  c: copiar',
    'espacio: volver  |  q: salir',
)


class ColorRangePicker(Node):
    def __init__(self):
        super().__init__('color_range_picker')
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('hue_margin', 5)
        self.declare_parameter('saturation_margin', 30)
        self.declare_parameter('value_margin', 40)
        self.declare_parameter('ranges_file', '')
        self.topic = self.get_parameter('image_topic').value
        self.hue_margin = int(self.get_parameter('hue_margin').value)
        self.saturation_margin = int(
            self.get_parameter('saturation_margin').value
        )
        self.value_margin = int(self.get_parameter('value_margin').value)
        self.ranges_file = self.configured_ranges_file()

        self.bridge = CvBridge()
        self.latest = None
        self.frozen = None
        self.points = []
        self.drawing = False
        self.pairs = []
        self.mask_preview = None
        self.mask_window_open = False
        self.clipboard_text = ''
        self.save_note = ''
        self.tk_root = None
        self.window_name = f'Rangos HSV - {self.topic}'

        self.create_subscription(
            Image, self.topic, self.image_callback, qos_profile_sensor_data
        )

        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.get_logger().info(
            f'Escuchando {self.topic}. Espacio para congelar, q para salir.'
        )
        self.get_logger().info(
            f'Los rangos HSV se guardarán en {self.ranges_file}.'
        )

    def configured_ranges_file(self):
        path = self.get_parameter('ranges_file').value
        if path:
            return os.path.expanduser(path)

        config_dir = os.path.join(pkg_dir('lab2'), 'config')
        path = os.path.join(config_dir, 'hsv_ranges.json')
        if not os.path.exists(path):
            pose_list = os.path.join(config_dir, 'pose_list.txt')
            if os.path.islink(pose_list):
                path = os.path.join(
                    os.path.dirname(os.path.realpath(pose_list)),
                    'hsv_ranges.json',
                )
        return path

    def image_callback(self, msg):
        try:
            self.latest = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8'
            )
        except CvBridgeError as error:
            self.get_logger().error(f'No se pudo convertir la imagen: {error}')

    def mouse_callback(self, event, x, y, _flags, _param):
        if self.frozen is None:
            return

        height, width = self.frozen.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.close_mask_window()
            self.points = [(x, y)]
            self.drawing = True
            self.pairs = []
            self.mask_preview = None
            self.clipboard_text = ''
            self.save_note = ''
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            last_x, last_y = self.points[-1]
            if abs(x - last_x) + abs(y - last_y) >= 2:
                self.points.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            if self.points[-1] != (x, y):
                self.points.append((x, y))
            if len(self.points) >= 3:
                self.calculate_ranges()

    def calculate_ranges(self):
        hsv = cv2.cvtColor(self.frozen, cv2.COLOR_BGR2HSV)
        pairs = mask_ranges(hsv, self.points)
        if not pairs:
            self.get_logger().warn('La región no cubre ningún píxel.')
            return

        self.pairs = pad_pairs(
            pairs,
            hue_margin=self.hue_margin,
            saturation_margin=self.saturation_margin,
            value_margin=self.value_margin,
        )
        self.update_preview(self.pairs)

        self.clipboard_text = format_hsv_ranges(self.pairs)
        self.save_note = ''
        self.get_logger().info(
            'Rangos HSV con margen de seguridad:\n' + self.clipboard_text
        )
        self.get_logger().info(
            'Presiona s para sobrescribir o a para fusionar con el archivo.'
        )
        self.copy_current()

    def update_preview(self, pairs):
        hsv = cv2.cvtColor(self.frozen, cv2.COLOR_BGR2HSV)
        mask = build_mask(hsv, pairs)
        mask_view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        result_view = cv2.bitwise_and(self.frozen, self.frozen, mask=mask)
        separator = np.full((mask.shape[0], 2, 3), 255, dtype=np.uint8)
        self.mask_preview = np.hstack([mask_view, separator, result_view])

    def toggle_freeze(self):
        if self.frozen is None:
            if self.latest is None:
                self.get_logger().warn('Todavía no llega ninguna imagen.')
                return
            self.frozen = self.latest.copy()
        else:
            self.frozen = None
        self.reset_selection()

    def reset_selection(self):
        self.points = []
        self.drawing = False
        self.pairs = []
        self.mask_preview = None
        self.clipboard_text = ''
        self.save_note = ''
        self.close_mask_window()

    def close_mask_window(self):
        if not self.mask_window_open:
            return
        try:
            cv2.destroyWindow(MASK_WINDOW)
        except cv2.error:
            pass
        self.mask_window_open = False

    def save_current(self, append=False):
        if not self.pairs:
            self.get_logger().warn('Todavía no hay rangos para guardar.')
            return

        if append and not os.path.exists(self.ranges_file):
            self.get_logger().warn(
                f'No existe {self.ranges_file}; se guardará como nuevo.'
            )

        try:
            saved = save_ranges(
                self.ranges_file, self.pairs, append=append
            )
        except (OSError, ValueError) as error:
            self.get_logger().error(
                f'No se pudieron guardar los rangos en '
                f'{self.ranges_file}: {error}'
            )
            return

        self.update_preview(saved)
        self.clipboard_text = format_hsv_ranges(saved)
        mode = 'fusionado con los anteriores' if append else 'sobrescrito'
        self.save_note = (
            f'Guardado: {os.path.basename(self.ranges_file)} ({mode})'
        )
        self.get_logger().info(
            f'Rangos guardados en {self.ranges_file} ({mode}):\n'
            f'{self.clipboard_text}'
        )
        self.get_logger().info(
            'Reinicia detector_de_objeto.py para cargar los nuevos rangos.'
        )
        self.copy_current()

    def copy_current(self):
        if not self.clipboard_text:
            self.get_logger().warn('Todavía no hay rangos para copiar.')
            return

        target = self.copy_to_clipboard(self.clipboard_text)
        if target is None:
            self.get_logger().warn(
                'No se pudo copiar al portapapeles; copia el texto de la terminal.'
            )
        else:
            self.get_logger().info(f'Rangos copiados con {target}.')

    def copy_to_clipboard(self, text):
        commands = (
            ('xclip', ['xclip', '-selection', 'clipboard']),
            ('wl-copy', ['wl-copy']),
        )
        for name, command in commands:
            try:
                subprocess.run(
                    command,
                    input=text.encode('utf-8'),
                    check=True,
                    timeout=2,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return name
            except (FileNotFoundError, subprocess.SubprocessError):
                continue

        try:
            import tkinter as tk
        except ImportError:
            return None

        try:
            if self.tk_root is None:
                self.tk_root = tk.Tk()
                self.tk_root.withdraw()
            self.tk_root.clipboard_clear()
            self.tk_root.clipboard_append(text)
            self.tk_root.update()
            return 'tkinter'
        except tk.TclError:
            return None

    def refresh(self):
        if self.frozen is None:
            lines = list(INSTRUCTIONS_LIVE)
            if self.latest is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    frame,
                    f'Esperando {self.topic}...',
                    (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                frame = self.latest.copy()
        else:
            frame = self.frozen.copy()
            if len(self.points) >= 2:
                vertices = np.asarray(self.points, dtype=np.int32)
                vertices = vertices.reshape(-1, 1, 2)
                cv2.polylines(
                    frame, [vertices], not self.drawing, (0, 255, 0), 2
                )

            lines = []
            if self.clipboard_text:
                lines.extend(self.clipboard_text.splitlines())
                if self.save_note:
                    lines.append(self.save_note)
            else:
                lines.append('Dibuja el borde del objeto con click izquierdo')
            lines.extend(INSTRUCTIONS_FROZEN)

        frame = self.with_status(frame, lines)
        cv2.imshow(self.window_name, frame)
        if self.mask_preview is not None:
            cv2.imshow(MASK_WINDOW, self.mask_preview)
            self.mask_window_open = True

    @staticmethod
    def with_status(frame, lines):
        line_height = 22
        box_height = 10 + line_height * len(lines)
        overlay = frame.copy()
        cv2.rectangle(
            overlay, (0, 0), (frame.shape[1], box_height), (0, 0, 0), -1
        )
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (8, 22 + index * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        return frame


def main(args=None):
    rclpy.init(args=args)
    node = ColorRangePicker()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            node.refresh()

            try:
                visible = cv2.getWindowProperty(
                    node.window_name, cv2.WND_PROP_VISIBLE
                )
            except cv2.error:
                break
            # En GTK getWindowProperty devuelve -1 cuando no puede
            # determinar la visibilidad. Solo 0 significa ventana cerrada.
            if visible == 0:
                break

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord(' '):
                node.toggle_freeze()
            elif key == ord('r'):
                node.reset_selection()
            elif key == ord('s'):
                node.save_current()
            elif key == ord('a'):
                node.save_current(append=True)
            elif key == ord('c'):
                node.copy_current()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
