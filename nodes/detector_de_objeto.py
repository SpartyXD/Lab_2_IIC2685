#!/usr/bin/env python3
from collections import deque
import rclpy
import message_filters
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Int16, Bool

import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


DISTANCIA_MAX = 0.70
SHOW_MASKS = True
SHOW_IMG = True

class ObjectDetector(Node):
    def __init__( self ):
        super().__init__('Obstacle_detector')

        self.queue_rgb = deque(maxlen=2)
        self.queue_depth = deque(maxlen=2)

        # https://stackoverflow.com/questions/57119292/how-to-merge-2-ros-topics-into-1#comment100853370_57120257
        self.get_logger().info("Iniciando ...")

        self.depth_sub = self.create_subscription(Image, '/depth/image_raw', self.depth_callback, 2)
        self.rgb_sub = self.create_subscription(Image, '/image_raw', self.rgb_callback, 2)
        
        self.blue_square_position_pub = self.create_publisher(Int16, '/blue_square/position', 2)
        self.blue_square_detected_pub = self.create_publisher(Bool, '/blue_square/detected', 2)
        self.bridge = CvBridge()


        #https://opencv.org/blob-detection-using-opencv/
        blob_params = cv2.SimpleBlobDetector_Params()
        
        # Filter by Area
        blob_params.filterByArea = True
        blob_params.minArea = 500
        blob_params.maxArea = 100000
        
        # Filter by Circularity
        blob_params.filterByCircularity = False
        
        # Filter by Convexity
        blob_params.filterByConvexity = True
        blob_params.minConvexity = 0.7
        
        blob_params.filterByColor = True
        blob_params.blobColor = 255

        # Filter by Inertia
        blob_params.filterByInertia = True
        blob_params.minInertiaRatio = 0.3

        # Create a detector with the parameters
        self.detector = cv2.SimpleBlobDetector_create(blob_params)


        self.lower_blue = np.array([97,134,129] )
        self.upper_blue = np.array([121,255,253])

    def rgb_callback(self, rgb_data):
        self.get_logger().info("Imagen RGB Recibida")
        self.queue_rgb.append(rgb_data)
        self.detect_blue_square_cb()

    def depth_callback(self, depth_data):
        self.get_logger().info("Depth Recibida")
        self.queue_depth.append(depth_data)
        self.detect_blue_square_cb()
    

    def distancia_valida(self, array):
        valid = (array <= DISTANCIA_MAX) | (np.isnan(array))
        return valid.astype(np.uint8) * 255

    def detect_blue_square_cb(self):
        #https://docs.opencv.org/4.13.0/df/d9d/tutorial_py_colorspaces.html
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
        mask_color = cv2.inRange(hsv_array, self.lower_blue, self.upper_blue)

        imagen_distancia_valida = cv2.bitwise_and(bgr_array, bgr_array, mask= mask_distancia)
        imagen_color_valido= cv2.bitwise_and(bgr_array, bgr_array, mask= mask_distancia & mask_color)
        
        # Detect blob
        x=0
        detected = False
        keypoints = self.detector.detect(mask_distancia & mask_color)
        if keypoints:
            blob = max(keypoints, key=lambda kp: kp.size)
            coords = tuple(map(int, blob.pt))
            x =  int(coords[0]-ancho/2)
            detected = True
            cv2.putText(bgr_array, str(x), coords, 1, 2, (0, 0, 0),1)
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
        if SHOW_IMG:
            cv2.imshow('Imagen Posicion Cuadrado Azul', bgr_array)
        if SHOW_MASKS or SHOW_IMG:
            cv2.waitKey(1)
            


if __name__ == '__main__':
    rclpy.init()
    node = ObjectDetector()
    rclpy.spin(node)
    rclpy.shutdown()

