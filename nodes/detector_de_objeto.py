#!/usr/bin/env python3
"""Detector del cuadrado azul con adaptación automática a la iluminación.

Publica exactamente lo mismo que la versión anterior, así que
seguidor_de_objeto.py no cambia:
    /blue_square/position  (Int16)  desplazamiento horizontal desde el centro
    /blue_square/detected  (Bool)

Qué cambia respecto a la versión anterior:
  * Modelo de color adaptativo: cada vez que el cuadrado se confirma (forma +
    profundidad), el tono, la tolerancia, la saturación mínima y el brillo
    mínimo se ajustan de a poco a cómo se ve el cuadrado en ese momento.
    El tono nunca se aleja más de max_hue_drift del valor calibrado, así no
    puede "derivar" hacia otro color.
  * Compensación global de luz: si la escena completa se oscurece, el brillo
    mínimo baja en la misma proporción.
  * Recuperación: si el cuadrado se pierde por un rato, el modelo vuelve a la
    calibración y los umbrales se relajan gradualmente hasta reencontrarlo.
  * Solo se usan cotas inferiores de S y V (sin techo): un reflejo o una zona
    más iluminada del cuadrado ya no lo saca de la máscara.
  * Detección por contornos (área, relleno, solidez, proporción) en vez de
    SimpleBlobDetector, que fallaba con los hoyos que deja la luz en la máscara.
  * Recarga config/hsv_ranges.json automáticamente cuando cambia: ya no hay
    que reiniciar el nodo después de usar color_range_picker.py.
  * Calibración con un click: click izquierdo sobre el cuadrado en la ventana
    principal y aprende el color al tiro (queda en memoria).
  * La imagen se pide en bgr8 y se convierte con BGR2HSV, igual que
    color_range_picker.py, así los rangos calculados allá calzan acá
    sin importar la codificación que publique la cámara.

Teclas (con alguna ventana de OpenCV enfocada):
    click izq  aprende el color desde el punto clickeado
    r          vuelve a la calibración (archivo o último click)
"""

from collections import deque
import json
import os

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory as pkg_dir
from cv_bridge import CvBridge, CvBridgeError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int16

WINDOW_MAIN = 'Imagen Posicion Cuadrado Azul'
WINDOW_DISTANCE = 'Distancia Valida'
WINDOW_COLOR = 'Colores Validos (dentro de la distancia valida)'

# Pisos absolutos: con menos saturación o brillo el tono es puro ruido.
S_FLOOR = 40.0
V_FLOOR = 25.0
HUE_TOL_MIN = 6.0
HUE_TOL_MAX = 25.0
# Calibración por defecto si no se puede leer el JSON: un azul genérico.
DEFAULT_SEED = (110.0, 15.0, 80.0, 40.0)  # (tono, tolerancia, S mín, V mín)

# Profundidad válida y planitud del cuadrado (mm).
DISTANCIA_MIN = 100
DISTANCIA_MAX = 1200
ESPESOR_MAX = 80
# Filtros de forma.
AREA_MIN = 1000
AREA_MAX = 100000
RELLENO_MIN = 0.80        # área / rectángulo mínimo que la contiene
PROPORCION_MIN = 0.5      # lado corto / lado largo
CIRCULARIDAD_MAX = 0.90   # cuadrado ~0.785, círculo ~1.0
# Adaptación a la luz.
TASA_ADAPTACION = 0.05    # qué tan rápido sigue los cambios de luz
DERIVA_TONO_MAX = 12.0    # cuánto puede alejarse el tono de la calibración
FRAMES_PARA_RELAJAR = 15  # frames perdido antes de volver a la calibración
FRAMES_DE_GRACIA = 2      # frames perdido en que se mantiene la última x
BALANCE_BLANCOS = False
# Ventanas.
SHOW_MASKS = True
SHOW_IMG = True


def hue_distance(hue, center):
    """Distancia circular entre tonos de OpenCV (0-179)."""
    diff = np.abs(hue.astype(np.int16) - int(round(center)) % 180)
    return np.minimum(diff, 180 - diff)


def wrap_hue(delta):
    """Lleva una diferencia de tonos al rango [-90, 90)."""
    return (delta + 90.0) % 180.0 - 90.0


def circular_hue_stats(hues):
    """Media y desviación circular de un arreglo de tonos (0-179)."""
    angles = hues.astype(np.float32) * (np.pi / 90.0)
    sin_mean = float(np.mean(np.sin(angles)))
    cos_mean = float(np.mean(np.cos(angles)))
    mean = (np.degrees(np.arctan2(sin_mean, cos_mean)) / 2.0) % 180.0
    resultant = min(max(np.hypot(sin_mean, cos_mean), 1e-6), 1.0)
    std = np.degrees(np.sqrt(-2.0 * np.log(resultant))) / 2.0
    return float(mean), float(std)


def seed_from_ranges(data):
    """Convierte {'lower': [...], 'upper': [...]} en (tono, tol, S mín, V mín).

    Acepta un solo par o una lista de pares. Supone que el rango no cruza el
    tono 0, lo que se cumple para el azul.
    """
    lower = np.atleast_2d(np.asarray(data['lower'], dtype=np.float32))
    upper = np.atleast_2d(np.asarray(data['upper'], dtype=np.float32))
    if lower.shape != upper.shape or lower.shape[1] != 3:
        raise ValueError('lower y upper deben ser tripletas HSV')
    hue_low = float(lower[:, 0].min())
    hue_high = float(upper[:, 0].max())
    hue = (hue_low + hue_high) / 2.0
    tolerance = float(np.clip((hue_high - hue_low) / 2.0,
                              HUE_TOL_MIN, HUE_TOL_MAX))
    s_min = max(float(lower[:, 1].min()), S_FLOOR)
    v_min = max(float(lower[:, 2].min()), V_FLOOR)
    return hue, tolerance, s_min, v_min


def gray_world(bgr):
    """Balance de blancos 'gray world': iguala el promedio de los 3 canales."""
    means = np.array(cv2.mean(bgr)[:3], dtype=np.float32)
    gains = means.mean() / np.maximum(means, 1.0)
    return np.clip(bgr.astype(np.float32) * gains, 0, 255).astype(np.uint8)


class ObjectDetector(Node):
    def __init__(self):
        super().__init__('Obstacle_detector')

        self.min_distance = DISTANCIA_MIN
        self.max_distance = DISTANCIA_MAX
        self.max_thickness = ESPESOR_MAX
        self.min_area = AREA_MIN
        self.max_area = AREA_MAX
        self.min_fill = RELLENO_MIN
        self.min_aspect = PROPORCION_MIN
        self.max_circularity = CIRCULARIDAD_MAX
        self.adapt_rate = TASA_ADAPTACION
        self.max_hue_drift = DERIVA_TONO_MAX
        self.lost_frames_relax = FRAMES_PARA_RELAJAR
        self.hold_frames = FRAMES_DE_GRACIA
        self.white_balance = BALANCE_BLANCOS
        self.show_masks = SHOW_MASKS
        self.show_image = SHOW_IMG

        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # Calibración (semilla) y modelo adaptativo.
        self.lost_frames = 0
        self.last_x = None
        self.scene_v_ref = None
        self.last_scene_v = None
        self.last_hsv = None
        self.ranges_file = self.configured_ranges_file()
        self.ranges_mtime = None
        self.seed = DEFAULT_SEED
        self.reset_model()
        if not self.load_ranges():
            self.get_logger().warn(
                f'Usando un azul genérico {DEFAULT_SEED}. Haz click sobre el '
                'cuadrado en la ventana principal para calibrar.'
            )

        self.queue_rgb = deque(maxlen=2)
        self.queue_depth = deque(maxlen=2)
        self.bridge = CvBridge()

        self.depth_sub = self.create_subscription(
            Image, '/depth/image_raw', self.depth_callback, 2)
        self.rgb_sub = self.create_subscription(
            Image, '/image_raw', self.rgb_callback, 2)
        self.blue_square_position_pub = self.create_publisher(
            Int16, '/blue_square/position', 2)
        self.blue_square_detected_pub = self.create_publisher(
            Bool, '/blue_square/detected', 2)

        self.create_timer(1.0, self.check_ranges_file)

        if self.show_image:
            cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(WINDOW_MAIN, self.mouse_callback)

        self.get_logger().info('Iniciando ...')

    # ------------------------------------------------------------------
    # Calibración desde archivo
    # ------------------------------------------------------------------
    @staticmethod
    def configured_ranges_file():
        """Misma lógica que color_range_picker.py para encontrar el JSON."""
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

    def load_ranges(self):
        try:
            mtime = os.path.getmtime(self.ranges_file)
            with open(self.ranges_file, encoding='utf-8') as ranges_file:
                seed = seed_from_ranges(json.load(ranges_file))
        except (OSError, ValueError, KeyError, TypeError) as error:
            # Si el picker estaba escribiendo, se reintenta en el próximo tick.
            self.get_logger().warn(
                f'No se pudo leer {self.ranges_file}: {error}')
            return False

        self.ranges_mtime = mtime
        self.seed = seed
        self.reset_model()
        self.get_logger().info(
            f'Calibración cargada desde {self.ranges_file}: '
            f'H {seed[0]:.0f}±{seed[1]:.0f}, S≥{seed[2]:.0f}, V≥{seed[3]:.0f}'
        )
        return True

    def check_ranges_file(self):
        try:
            mtime = os.path.getmtime(self.ranges_file)
        except OSError:
            return
        if mtime != self.ranges_mtime:
            self.load_ranges()

    # ------------------------------------------------------------------
    # Modelo de color
    # ------------------------------------------------------------------
    def reset_model(self):
        """Vuelve el modelo adaptativo a la calibración."""
        self.hue, self.hue_tol, self.s_min, self.v_min = self.seed
        self.scene_v_ref = self.last_scene_v

    def thresholds(self, scene_v):
        """Umbrales efectivos para este frame."""
        if self.scene_v_ref is None:
            self.scene_v_ref = scene_v

        # Solo se compensa hacia abajo: si la escena se aclara, el cuadrado
        # también, y como no hay techo de V no hace falta subir el umbral.
        light = float(np.clip(scene_v / max(self.scene_v_ref, 1.0), 0.3, 1.0))

        relax = 0.0
        extra = self.lost_frames - self.lost_frames_relax
        if extra > 0:
            relax = min(1.0, extra / (2.0 * self.lost_frames_relax))

        tolerance = self.hue_tol * (1.0 + 0.5 * relax)
        s_min = self.s_min * (1.0 - 0.4 * relax)
        v_min = self.v_min * light * (1.0 - 0.5 * relax)
        return (
            self.hue,
            float(np.clip(tolerance, HUE_TOL_MIN, HUE_TOL_MAX * 1.5)),
            max(s_min, S_FLOOR),
            float(np.clip(v_min, V_FLOOR, 220.0)),
            relax,
        )

    def adapt(self, hsv, region, scene_v):
        """Ajusta el modelo con los píxeles del cuadrado recién confirmado."""
        pixels = hsv[region > 0]
        hues, sats, vals = pixels[:, 0], pixels[:, 1], pixels[:, 2]
        near = ((hue_distance(hues, self.hue) <= self.hue_tol * 1.5)
                & (sats >= S_FLOOR))
        if np.count_nonzero(near) < 50:
            return
        hues, sats, vals = hues[near], sats[near], vals[near]

        rate = self.adapt_rate
        mean_hue, std_hue = circular_hue_stats(hues)
        new_hue = (self.hue + rate * wrap_hue(mean_hue - self.hue)) % 180.0
        seed_hue = self.seed[0]
        drift = np.clip(wrap_hue(new_hue - seed_hue),
                        -self.max_hue_drift, self.max_hue_drift)
        self.hue = float((seed_hue + drift) % 180.0)

        target_tol = float(np.clip(2.5 * std_hue + 3.0,
                                   HUE_TOL_MIN, HUE_TOL_MAX))
        target_s = max(S_FLOOR, 0.7 * float(np.percentile(sats, 10)))
        target_v = max(V_FLOOR, 0.6 * float(np.percentile(vals, 10)))
        # Los umbrales bajan rápido y suben lento: perder el cuadrado por
        # un umbral muy estricto es peor que aceptar un poco más de ruido.
        step = lambda current, target: rate * (  # noqa: E731
            2.0 if target < current else 0.5) * (target - current)
        self.hue_tol += step(self.hue_tol, target_tol)
        self.s_min += step(self.s_min, target_s)
        self.v_min += step(self.v_min, target_v)
        self.scene_v_ref += rate * (scene_v - self.scene_v_ref)

    # ------------------------------------------------------------------
    # Calibración con click
    # ------------------------------------------------------------------
    def mouse_callback(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.learn_from_point(x, y)

    def learn_from_point(self, x, y):
        if self.last_hsv is None:
            self.get_logger().warn('Todavía no llega ninguna imagen.')
            return
        hsv = self.last_hsv
        height, width = hsv.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            return

        # Crece una región desde el click con píxeles de color parecido.
        flood_mask = np.zeros((height + 2, width + 2), np.uint8)
        flags = (4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
                 | (255 << 8))
        cv2.floodFill(hsv.copy(), flood_mask, (x, y), (0, 0, 0),
                      (8, 70, 90), (8, 70, 90), flags)
        region = flood_mask[1:-1, 1:-1]
        count = int(np.count_nonzero(region))
        if count < 100 or count > 0.5 * region.size:
            self.get_logger().warn(
                f'La región del click tiene {count} píxeles; haz click más '
                'al centro del cuadrado.')
            return

        pixels = hsv[region > 0]
        mean_hue, std_hue = circular_hue_stats(pixels[:, 0])
        tolerance = float(np.clip(2.5 * std_hue + 4.0,
                                  HUE_TOL_MIN, HUE_TOL_MAX))
        s_min = max(S_FLOOR, 0.7 * float(np.percentile(pixels[:, 1], 5)))
        v_min = max(V_FLOOR, 0.6 * float(np.percentile(pixels[:, 2], 5)))

        self.seed = (mean_hue, tolerance, s_min, v_min)
        self.reset_model()
        self.lost_frames = 0
        self.get_logger().info(
            f'Color aprendido con click: H {mean_hue:.0f}±{tolerance:.0f}, '
            f'S≥{s_min:.0f}, V≥{v_min:.0f}. Queda en memoria; usa '
            'color_range_picker.py si quieres guardarlo en el archivo.'
        )

    # ------------------------------------------------------------------
    # Procesamiento
    # ------------------------------------------------------------------
    def rgb_callback(self, rgb_data):
        self.queue_rgb.append(rgb_data)
        self.try_process()

    def depth_callback(self, depth_data):
        self.queue_depth.append(depth_data)
        self.try_process()

    def try_process(self):
        if not self.queue_rgb or not self.queue_depth:
            return
        rgb_data = self.queue_rgb.popleft()
        depth_data = self.queue_depth.popleft()
        try:
            self.process(rgb_data, depth_data)
        except CvBridgeError as error:
            self.get_logger().error(f'No se pudo convertir la imagen: {error}')

    def distancia_valida(self, array):
        valid = (~np.isnan(array) & (array >= self.min_distance)
                 & (array <= self.max_distance))
        return valid.astype(np.uint8) * 255

    def find_square(self, mask, depth_array):
        """Devuelve (contorno, región, centro) del mejor cuadrado o None."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(contour)
            if area < self.min_area:
                break
            if area > self.max_area:
                continue

            (_, _), (width, height), _ = cv2.minAreaRect(contour)
            if min(width, height) <= 0:
                continue
            if min(width, height) / max(width, height) < self.min_aspect:
                continue
            if area / (width * height) < self.min_fill:
                continue

            # La envoltura convexa suaviza el borde ruidoso: un cuadrado da
            # circularidad ~0.785 y un círculo ~1.0.
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            hull_perimeter = cv2.arcLength(hull, True)
            if hull_area <= 0 or area / hull_area < 0.9:
                continue
            circularity = 4.0 * np.pi * hull_area / hull_perimeter ** 2
            if circularity > self.max_circularity:
                continue

            # Chequeo de profundidad: el cuadrado debe ser plano.
            region = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(region, [contour], -1, 255, cv2.FILLED)
            region = cv2.erode(region, self.kernel_close)
            z = depth_array[region > 0].astype(np.float32)
            z = z[(z >= self.min_distance) & (z <= self.max_distance)]
            if z.size < 30:
                continue
            if np.percentile(z, 90) - np.percentile(z, 10) > self.max_thickness:
                continue

            moments = cv2.moments(contour)
            center = (int(moments['m10'] / moments['m00']),
                      int(moments['m01'] / moments['m00']))
            return contour, region, center
        return None

    def process(self, rgb_data, depth_data):
        bgr_array = self.bridge.imgmsg_to_cv2(rgb_data, desired_encoding='bgr8')
        depth_array = self.bridge.imgmsg_to_cv2(depth_data)
        alto, ancho = bgr_array.shape[:2]
        if depth_array.shape[:2] != (alto, ancho):
            depth_array = cv2.resize(depth_array, (ancho, alto),
                                     interpolation=cv2.INTER_NEAREST)

        source = gray_world(bgr_array) if self.white_balance else bgr_array
        hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
        scene_v = float(cv2.mean(hsv[:, :, 2])[0])
        self.last_hsv = hsv
        self.last_scene_v = scene_v

        hue, tolerance, s_min, v_min, relax = self.thresholds(scene_v)

        mask_distancia = self.distancia_valida(depth_array)
        mask_color = ((hue_distance(hsv[:, :, 0], hue) <= tolerance)
                      & (hsv[:, :, 1] >= s_min)
                      & (hsv[:, :, 2] >= v_min)).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask_color, mask_distancia)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)

        found = self.find_square(mask, depth_array)
        if found is not None:
            contour, region, center = found
            self.lost_frames = 0
            x = int(center[0] - ancho / 2)
            self.last_x = x
            detected = True
            self.adapt(hsv, region, scene_v)
        else:
            contour, center = None, None
            self.lost_frames += 1
            if self.lost_frames == self.lost_frames_relax:
                self.get_logger().info(
                    'Cuadrado perdido: volviendo a la calibración y '
                    'relajando umbrales.')
                self.reset_model()
            # Aguanta unos pocos frames perdidos (parpadeos por la luz)
            # para que el seguidor no frene de golpe.
            if self.last_x is not None and self.lost_frames <= self.hold_frames:
                x, detected = self.last_x, True
            else:
                x, detected = 0, False

        pos_x = Int16()
        pos_x.data = x
        square_detected = Bool()
        square_detected.data = detected
        self.blue_square_position_pub.publish(pos_x)
        self.blue_square_detected_pub.publish(square_detected)

        if self.show_masks:
            cv2.imshow(WINDOW_DISTANCE, cv2.bitwise_and(
                bgr_array, bgr_array, mask=mask_distancia))
            cv2.imshow(WINDOW_COLOR, cv2.bitwise_and(
                bgr_array, bgr_array, mask=mask))
        if self.show_image:
            view = bgr_array.copy()
            cv2.line(view, (ancho // 2, 0), (ancho // 2, alto), (0, 0, 0), 2)
            if contour is not None:
                cv2.drawContours(view, [contour], -1, (0, 255, 0), 2)
                cv2.line(view, (ancho // 2, center[1]), center, (0, 0, 0), 2)
                cv2.putText(view, str(x), center, 1, 2, (0, 0, 0), 1)
            if detected and contour is not None:
                state = f'DETECTADO x={x}'
            elif detected:
                state = f'manteniendo x={x}'
            elif relax > 0:
                state = f'buscando (relajado {relax:.0%})'
            else:
                state = 'buscando'
            lines = (
                f'H {hue:.0f}+-{tolerance:.0f}  S>={s_min:.0f}  '
                f'V>={v_min:.0f}',
                state,
                'click: aprender color  |  r: volver a calibracion',
            )
            self.draw_status(view, lines)
            cv2.imshow(WINDOW_MAIN, view)
        if self.show_masks or self.show_image:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                self.reset_model()
                self.get_logger().info('Modelo reiniciado a la calibración.')

    @staticmethod
    def draw_status(frame, lines):
        line_height = 22
        box_height = 10 + line_height * len(lines)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], box_height),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)
        for index, line in enumerate(lines):
            cv2.putText(frame, line, (8, 22 + index * line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,
                        cv2.LINE_AA)


def main():
    rclpy.init()
    node = ObjectDetector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()