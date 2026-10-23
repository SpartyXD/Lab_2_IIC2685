#!/usr/bin/env python3
"""Obtiene rangos HSV para un mask a partir de una región de /image_raw.

Controles:
    espacio    congela o descongela la imagen
    click izq  dibuja el borde del objeto sobre la imagen congelada
    r          borra el trazo y los rangos
    c          vuelve a copiar los rangos al portapapeles
    q / Esc    cierra

Al soltar el mouse, los rangos se copian al portapapeles y se imprimen en la
terminal.
"""

import subprocess

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from hsv_ranges import build_mask, format_hsv_ranges, mask_ranges

MASK_WINDOW = 'Mask'
INSTRUCTIONS_LIVE = 'espacio: congelar  |  q: salir'
INSTRUCTIONS_FROZEN = 'r: borrar  |  c: copiar  |  espacio: volver  |  q: salir'


class ColorRangePicker(Node):
    def __init__(self):
        super().__init__('color_range_picker')
        self.declare_parameter('image_topic', '/image_raw')
        self.topic = self.get_parameter('image_topic').value

        self.bridge = CvBridge()
        self.latest = None
        self.frozen = None
        self.points = []
        self.drawing = False
        self.pairs = []
        self.mask_preview = None
        self.mask_window_open = False
        self.clipboard_text = ''
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

        self.pairs = pairs
        mask = build_mask(hsv, pairs)
        mask_view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        result_view = cv2.bitwise_and(self.frozen, self.frozen, mask=mask)
        separator = np.full((mask.shape[0], 2, 3), 255, dtype=np.uint8)
        self.mask_preview = np.hstack([mask_view, separator, result_view])

        self.clipboard_text = format_hsv_ranges(pairs)
        self.get_logger().info('Rangos HSV:\n' + self.clipboard_text)
        self.copy_current()

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
        self.close_mask_window()

    def close_mask_window(self):
        if not self.mask_window_open:
            return
        try:
            cv2.destroyWindow(MASK_WINDOW)
        except cv2.error:
            pass
        self.mask_window_open = False

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
            lines = [INSTRUCTIONS_LIVE]
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
            else:
                lines.append('Dibuja el borde del objeto con click izquierdo')
            lines.append(INSTRUCTIONS_FROZEN)

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
            if visible < 1:
                break

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord(' '):
                node.toggle_freeze()
            elif key == ord('r'):
                node.reset_selection()
            elif key == ord('c'):
                node.copy_current()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
