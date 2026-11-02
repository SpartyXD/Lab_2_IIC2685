#!/usr/bin/env python3
from collections import deque
import json
import os

import rclpy
import numpy as np
from ament_index_python.packages import get_package_share_directory as pkg_dir
from rclpy.node import Node
from std_msgs.msg import Int16, Bool

import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

DISTANCIA_MIN = 100
DISTANCIA_MAX = 1500
ESPESOR_MAX = 80
EPS_VERTICES = 0.04
SHOW_MASKS = True
SHOW_IMG = True


class ObjectDetector(Node):
    def __init__( self ):
        super().__init__('Obstacle_detector')

        ranges_path = os.path.join(
            pkg_dir('lab2'), 'config', 'hsv_ranges.json'
        )
        with open(ranges_path, encoding='utf-8') as ranges_file:
            ranges = json.load(ranges_file)

        self.lower_blue = np.array(ranges['lower'], dtype=np.uint8)
        self.upper_blue = np.array(ranges['upper'], dtype=np.uint8)
        self.get_logger().info(
            f'Rangos HSV cargados desde {ranges_path}: '
            f'{self.lower_blue.tolist()} - {self.upper_blue.tolist()}'
        )

        self.queue_rgb = deque(maxlen=2)
        self.queue_depth = deque(maxlen=2)

        self.get_logger().info("Iniciando ...")

        self.depth_sub = self.create_subscription(Image, '/depth/image_raw', self.depth_callback, 2)
        self.rgb_sub = self.create_subscription(Image, '/image_raw', self.rgb_callback, 2)

        self.blue_square_position_pub = self.create_publisher(Int16, '/blue_square/position', 2)
        self.blue_square_detected_pub = self.create_publisher(Bool, '/blue_square/detected', 2)
        self.bridge = CvBridge()

    def rgb_callback(self, rgb_data):
        self.get_logger().info("Imagen RGB Recibida")
        self.queue_rgb.append(rgb_data)
        self.detect_blue_square_cb()

    def depth_callback(self, depth_data):
        self.get_logger().info("Depth Recibida")
        self.queue_depth.append(depth_data)
        self.detect_blue_square_cb()

    def distancia_valida(self, array):
        valid = ~(np.isnan(array)) & (array >= DISTANCIA_MIN) & (array <= DISTANCIA_MAX)
        return valid.astype(np.uint8) * 255

    def detect_blue_square_cb(self):
        if not(len(self.queue_rgb)) or not(len(self.queue_depth)):
            return

        self.get_logger().info("Imagenes recibidas :p")
        rgb_data = self.queue_rgb.popleft()
        depth_data = self.queue_depth.popleft()

        rgb_array = self.bridge.imgmsg_to_cv2(rgb_data)
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        hsv_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)

        depth_array = self.bridge.imgmsg_to_cv2(depth_data)
        alto, ancho = depth_array.shape

        mask_distancia = self.distancia_valida(depth_array)
        mask_color = cv2.inRange(
            hsv_array, self.lower_blue, self.upper_blue
        )

        imagen_distancia_valida = cv2.bitwise_and(bgr_array, bgr_array, mask= mask_distancia)
        imagen_color_valido= cv2.bitwise_and(bgr_array, bgr_array, mask= mask_distancia & mask_color)

        mask = cv2.bitwise_and(mask_distancia, mask_color)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mejor = None
        for contorno in contornos:
            aprox = cv2.approxPolyDP(
                contorno, EPS_VERTICES * cv2.arcLength(contorno, True), True
            )
            if len(aprox) != 4:
                continue

            relleno = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(relleno, [contorno], -1, 255, cv2.FILLED)
            z = depth_array[relleno > 0].astype(np.float32)
            z = z[(z >= DISTANCIA_MIN) & (z <= DISTANCIA_MAX)]
            if z.size < 30:
                continue
            if np.percentile(z, 90) - np.percentile(z, 10) > ESPESOR_MAX:
                continue

            M = cv2.moments(contorno)
            if M['m00'] == 0:
                continue
            area = cv2.contourArea(contorno)
            if mejor is None or area > mejor[0]:
                mejor = (area, M['m10'] / M['m00'], M['m01'] / M['m00'], contorno, aprox)

        x = 0
        detected = False
        if mejor is not None:
            _, cx, cy, contorno, aprox = mejor
            coords = (int(cx), int(cy))
            x = int(coords[0] - ancho / 2)
            detected = True
            cv2.drawContours(bgr_array, [contorno], -1, (0, 255, 0), 2)
            for punto in aprox.reshape(-1, 2):
                cv2.circle(bgr_array, (int(punto[0]), int(punto[1])), 4, (0, 0, 255), -1)
            cv2.putText(bgr_array, str(x), coords, 1, 2, (0, 0, 0), 1)
            cv2.line(bgr_array, (int(ancho/2), coords[1]), coords, (0, 0, 0), 2)

        cv2.line(bgr_array, (int(ancho/2), 0), (int(ancho/2), alto), (0, 0, 0), 2)

        pos_x = Int16()
        square_detected = Bool()
        pos_x.data = x
        square_detected.data = detected
        self.blue_square_position_pub.publish(pos_x)
        self.blue_square_detected_pub.publish(square_detected)

        if SHOW_MASKS:
            cv2.imshow('Distancia Válida', imagen_distancia_valida)
            cv2.imshow('Colores Válidos (dentro de la distancia valida)', imagen_color_valido)
            cv2.imshow('Mascara Contornos', mask)
        if SHOW_IMG:
            cv2.imshow('Imagen Posicion Cuadrado Azul', bgr_array)
        if SHOW_MASKS or SHOW_IMG:
            cv2.waitKey(1)


if __name__ == '__main__':
    rclpy.init()
    node = ObjectDetector()
    rclpy.spin(node)
    rclpy.shutdown()
