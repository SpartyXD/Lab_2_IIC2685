#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

'''
Generado por IA
Publica /camera/rgb/image_color usando la webcam del pc
Publica /camera/depth/image_raw como un cuadrado valido en el centro y no valido en la demas parte segun los criterios del detector de objeto
'''


class CameraSimulatorNode(Node):
    def __init__(self):
        super().__init__('camera_simulator_node')

        # Publicadores
        self.rgb_pub = self.create_publisher(Image, '/camera/rgb/image_color/test', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw/test', 10)

        self.bridge = CvBridge()

        # Inicializar webcam (índice 0 por defecto)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error('No se pudo abrir la cámara web.')
            exit(1)

        # Timer a 30 FPS (~0.033 segundos)
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
        self.get_logger().info('Publicando RGB desde webcam y Depth sintético a 30 FPS...')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('No se pudo leer el frame de la webcam.')
            return

        alto, ancho = frame.shape[:2]

        # 1. Crear mapa de profundidad sintético (float32 en metros)
        # Fondo por defecto a 2.0 metros (fuera del rango de 0.70 m)
        depth_frame = np.full((alto, ancho), 2.0, dtype=np.float32)

        # Definir una región central cuadrada con distancia válida (ej. 0.45 metros)
        tamano_cuadrado = int(min(alto, ancho) / 1.5)
        centro_y, centro_x = alto // 2, ancho // 2
        y1 = centro_y - tamano_cuadrado // 2
        y2 = centro_y + tamano_cuadrado // 2
        x1 = centro_x - tamano_cuadrado // 2
        x2 = centro_x + tamano_cuadrado // 2

        # Rango válido para tu detector (0.45 m <= 0.70 m)
        depth_frame[y1:y2, x1:x2] = 0.45

        # 2. Obtener timestamp sincronizado para ambos mensajes
        now = self.get_clock().now().to_msg()

        # 3. Convertir y publicar imagen RGB
        rgb_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        rgb_msg.header.stamp = now
        rgb_msg.header.frame_id = 'camera_optical_frame'
        self.rgb_pub.publish(rgb_msg)

        # 4. Convertir y publicar imagen de Profundidad (encoding 32FC1 para float32 en metros)
        depth_msg = self.bridge.cv2_to_imgmsg(depth_frame, encoding='32FC1')
        depth_msg.header.stamp = now
        depth_msg.header.frame_id = 'camera_optical_frame'
        self.depth_pub.publish(depth_msg)

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraSimulatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()