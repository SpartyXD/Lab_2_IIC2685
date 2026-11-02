#!/usr/bin/env python3
"""Detector del cuadrado azul con la profundidad del Kinect como fuente principal.

Publica lo mismo de siempre (seguidor_de_objeto.py no cambia):
    /blue_square/position  (Int16)  desplazamiento horizontal desde el centro
    /blue_square/detected  (Bool)

Cómo funciona:
  1. La profundidad se convierte en una nube de puntos con los intrínsecos de
     /depth/camera_info (o los típicos del Kinect si no llega).
  2. Segmentación geométrica, sin color: la imagen se corta donde hay un salto
     de distancia o un quiebre en la orientación de la superficie, así cada
     objeto o cara queda como una región separada.
  3. Cada región se ajusta a un plano y se mide en milímetros sobre ese plano:
     planitud, lados reales, proporción y relleno (cuadrado vs. círculo).
     Como se mide sobre el plano, la perspectiva no afecta.
  4. El color solo confirma: entre las regiones que geométricamente son el
     cuadrado, se acepta la que tiene suficientes píxeles azulados.
  5. Con cada detección confirmada, el modelo de color aprende cómo se ve el
     cuadrado con la luz del momento.

Ventanas:
    Imagen Posicion Cuadrado Azul  RGB con el cuadrado detectado
    Profundidad                    regiones planas con su medida en mm:
                                   verde = aceptada, amarillo = forma correcta
                                   pero sin azul, rojo = forma incorrecta
    Colores Azulados               píxeles que el modelo considera azules

Teclas:
    click izq  (ventana RGB) aprende el color desde el punto clickeado
    r          vuelve a la calibración de color del archivo
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
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Int16

DEPTH_TOPIC = '/depth/image_raw'
RGB_TOPIC = '/image_raw'
CAMERA_INFO_TOPIC = '/depth/camera_info'

# --- Rango de trabajo (mm). Ojo: el Kinect no mide bajo ~450 mm. ---
DISTANCIA_MIN = 100
DISTANCIA_MAX = 1200

# --- Geometría del cuadrado, medida en mm sobre su propio plano ---
LADO_MIN_MM = 100         # Ajusta a tu cuadrado real; el nodo imprime en la
LADO_MAX_MM = 400         # terminal los lados que mide cuando lo detecta.
PROPORCION_MIN = 0.70     # lado corto / lado largo
RELLENO_MIN = 0.85        # área / rectángulo (cuadrado ~1.0, círculo ~0.79)
PLANITUD_MAX_MM = 8.0     # error RMS contra el plano (+1% de la distancia)
INCLINACION_MAX = 65.0    # grados entre la normal del cuadrado y la cámara

# --- Color como apoyo ---
COLOR_MIN = 0.35          # fracción mínima de píxeles azulados en la región
S_FLOOR = 40.0            # bajo esta saturación el tono es ruido
V_FLOOR = 25.0            # bajo este brillo el tono es ruido
HUE_TOL_MIN = 6.0
HUE_TOL_MAX = 25.0
DEFAULT_SEED = (110.0, 15.0, 80.0)  # (tono, tolerancia, S mín) si no hay JSON
TASA_ADAPTACION = 0.05    # qué tan rápido aprende el color con la luz actual
DERIVA_TONO_MAX = 12.0    # cuánto puede alejarse el tono de la calibración
FRAMES_PARA_RESETEAR_COLOR = 30

# --- Segmentación de profundidad ---
ESCALA = 2                # se procesa a la mitad de la resolución
SALTO_MIN_MM = 25.0       # salto de distancia que separa objetos...
SALTO_REL = 0.04          # ...o 4% de la distancia, lo que sea mayor
ANGULO_QUIEBRE = 35.0     # grados de cambio de normal que separan caras
AREA_MIN_PX = 80          # px de la imagen reducida
MAX_PUNTOS_AJUSTE = 3000

# --- Alineación RGB-profundidad ---
DESFASE_X = 0             # px en la imagen RGB. Si el contorno verde aparece
DESFASE_Y = 0             # corrido respecto al cuadrado real, corrígelo aquí.
FX_DEFAULT = 525.0        # focal del Kinect a 640x480 si no llega camera_info

# --- Comportamiento ---
FRAMES_DE_GRACIA = 2      # frames perdido en que se mantiene la última x
SHOW_MASKS = True
SHOW_IMG = True

WINDOW_MAIN = 'Imagen Posicion Cuadrado Azul'
WINDOW_DEPTH = 'Profundidad'
WINDOW_COLOR = 'Colores Azulados'

KERNEL_3 = np.ones((3, 3), np.uint8)
KERNEL_7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


# ----------------------------------------------------------------------
# Utilidades de color
# ----------------------------------------------------------------------
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
    """Convierte {'lower': [...], 'upper': [...]} en (tono, tol, S mín).

    Acepta un solo par o una lista de pares. Supone que el rango no cruza el
    tono 0, lo que se cumple para el azul.
    """
    lower = np.atleast_2d(np.asarray(data['lower'], dtype=np.float32))
    upper = np.atleast_2d(np.asarray(data['upper'], dtype=np.float32))
    if lower.shape != upper.shape or lower.shape[1] != 3:
        raise ValueError('lower y upper deben ser tripletas HSV')
    hue_low = float(lower[:, 0].min())
    hue_high = float(upper[:, 0].max())
    tolerance = float(np.clip((hue_high - hue_low) / 2.0,
                              HUE_TOL_MIN, HUE_TOL_MAX))
    s_min = max(float(lower[:, 1].min()), S_FLOOR)
    return (hue_low + hue_high) / 2.0, tolerance, s_min


# ----------------------------------------------------------------------
# Utilidades de geometría
# ----------------------------------------------------------------------
def depth_to_mm(depth):
    """Profundidad en mm como float32, con 0 donde no hay medida."""
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if np.issubdtype(depth.dtype, np.floating):
        z = depth.astype(np.float32)
        z[~np.isfinite(z)] = 0.0
        if z.max() < 20.0:  # viene en metros
            z *= 1000.0
        return z
    return depth.astype(np.float32)


def fit_plane(points):
    """Ajuste robusto de plano. Devuelve (centro, e1, e2, normal, rms, inliers)."""
    if len(points) < 30:
        return None
    center = points.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov((points - center).T))
    normal = vectors[:, 0]
    threshold = max(10.0, 0.015 * float(center[2]))
    inliers = np.abs((points - center) @ normal) < threshold
    if np.count_nonzero(inliers) < 30:
        return None

    kept = points[inliers]
    center = kept.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov((kept - center).T))
    normal = vectors[:, 0]
    residuals = (kept - center) @ normal
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    return (center, vectors[:, 2], vectors[:, 1], normal, rms,
            float(np.mean(inliers)))


class ObjectDetector(Node):
    def __init__(self):
        super().__init__('Obstacle_detector')

        self.bridge = CvBridge()
        self.intrinsics = None
        self.queue_rgb = deque(maxlen=2)
        self.queue_depth = deque(maxlen=2)

        # Seguimiento
        self.lost_frames = 0
        self.last_x = None
        self.last_center = None
        self.last_hsv = None

        # Modelo de color (solo apoyo)
        self.ranges_file = self.configured_ranges_file()
        self.ranges_mtime = None
        self.seed = DEFAULT_SEED
        self.reset_color()
        if not self.load_ranges():
            self.get_logger().warn(
                f'Usando un azul genérico {DEFAULT_SEED}. Haz click sobre el '
                'cuadrado en la ventana RGB para calibrar el color.')

        self.create_subscription(
            Image, DEPTH_TOPIC, self.depth_callback, 2)
        self.create_subscription(
            Image, RGB_TOPIC, self.rgb_callback, 2)
        self.create_subscription(
            CameraInfo, CAMERA_INFO_TOPIC, self.camera_info_callback, 2)
        self.blue_square_position_pub = self.create_publisher(
            Int16, '/blue_square/position', 2)
        self.blue_square_detected_pub = self.create_publisher(
            Bool, '/blue_square/detected', 2)
        self.create_timer(1.0, self.check_ranges_file)

        if SHOW_IMG:
            cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(WINDOW_MAIN, self.mouse_callback)

        self.get_logger().info('Iniciando ...')

    # ------------------------------------------------------------------
    # Entradas
    # ------------------------------------------------------------------
    def camera_info_callback(self, msg):
        if msg.k[0] > 0 and msg.width > 0:
            if self.intrinsics is None:
                self.get_logger().info(
                    f'Intrínsecos de profundidad: fx={msg.k[0]:.1f}, '
                    f'fy={msg.k[4]:.1f}, {msg.width}x{msg.height}')
            self.intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5],
                               msg.width, msg.height)

    def depth_intrinsics(self, width, height):
        """(fx, fy, cx, cy) para una imagen de profundidad de este tamaño."""
        if self.intrinsics is not None:
            fx, fy, cx, cy, info_w, info_h = self.intrinsics
            sx, sy = width / info_w, height / info_h
            return (fx * sx, fy * sy,
                    (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5)
        focal = FX_DEFAULT * width / 640.0
        return focal, focal, (width - 1) / 2.0, (height - 1) / 2.0

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

    # ------------------------------------------------------------------
    # Calibración de color: archivo y click
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
            self.get_logger().warn(
                f'No se pudo leer {self.ranges_file}: {error}')
            return False

        self.ranges_mtime = mtime
        self.seed = seed
        self.reset_color()
        self.get_logger().info(
            f'Color cargado desde {self.ranges_file}: '
            f'H {seed[0]:.0f}±{seed[1]:.0f}, S≥{seed[2]:.0f}')
        return True

    def check_ranges_file(self):
        try:
            mtime = os.path.getmtime(self.ranges_file)
        except OSError:
            return
        if mtime != self.ranges_mtime:
            self.load_ranges()

    def reset_color(self):
        self.hue, self.hue_tol, self.s_min = self.seed

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
        self.seed = (mean_hue, tolerance, s_min)
        self.reset_color()
        self.get_logger().info(
            f'Color aprendido con click: H {mean_hue:.0f}±{tolerance:.0f}, '
            f'S≥{s_min:.0f}. Queda en memoria hasta reiniciar el nodo.')

    # ------------------------------------------------------------------
    # Modelo de color
    # ------------------------------------------------------------------
    def blue_mask(self, hsv):
        """Píxeles azulados, con umbrales holgados: el color solo apoya."""
        mask = ((hue_distance(hsv[:, :, 0], self.hue) <= self.hue_tol * 1.5)
                & (hsv[:, :, 1] >= max(S_FLOOR, 0.6 * self.s_min))
                & (hsv[:, :, 2] >= V_FLOOR))
        return mask.astype(np.uint8) * 255

    def adapt_color(self, hsv, region):
        """Aprende el color con los píxeles del cuadrado confirmado."""
        pixels = hsv[region > 0]
        if len(pixels) < 50:
            return
        hues, sats = pixels[:, 0], pixels[:, 1]
        near = ((hue_distance(hues, self.hue) <= self.hue_tol * 2.0)
                & (sats >= S_FLOOR) & (pixels[:, 2] >= V_FLOOR))
        if np.count_nonzero(near) < 50:
            return
        hues, sats = hues[near], sats[near]

        rate = TASA_ADAPTACION
        mean_hue, std_hue = circular_hue_stats(hues)
        new_hue = (self.hue + rate * wrap_hue(mean_hue - self.hue)) % 180.0
        drift = np.clip(wrap_hue(new_hue - self.seed[0]),
                        -DERIVA_TONO_MAX, DERIVA_TONO_MAX)
        self.hue = float((self.seed[0] + drift) % 180.0)

        target_tol = float(np.clip(2.5 * std_hue + 3.0,
                                   HUE_TOL_MIN, HUE_TOL_MAX))
        target_s = max(S_FLOOR, 0.7 * float(np.percentile(sats, 10)))
        self.hue_tol += rate * (target_tol - self.hue_tol)
        self.s_min += rate * (target_s - self.s_min)

    # ------------------------------------------------------------------
    # Geometría
    # ------------------------------------------------------------------
    def segment(self, z, fx, fy, cx, cy):
        """Corta la profundidad reducida en regiones suaves y continuas."""
        height, width = z.shape
        offset = ESCALA // 2
        us = np.arange(width, dtype=np.float32) * ESCALA + offset
        vs = np.arange(height, dtype=np.float32) * ESCALA + offset

        valid = (z >= DISTANCIA_MIN) & (z <= DISTANCIA_MAX)
        z = np.where(valid, z, 0.0).astype(np.float32)
        points = np.dstack([
            (us[None, :] - cx) * z / fx,
            (vs[:, None] - cy) * z / fy,
            z,
        ]).astype(np.float32)

        # Saltos de distancia entre vecinos.
        threshold = np.maximum(SALTO_MIN_MM, SALTO_REL * z)
        jump = np.zeros((height, width), bool)
        jump_x = np.abs(np.diff(z, axis=1)) > threshold[:, :-1]
        jump_y = np.abs(np.diff(z, axis=0)) > threshold[:-1, :]
        jump[:, :-1] |= jump_x
        jump[:, 1:] |= jump_x
        jump[:-1, :] |= jump_y
        jump[1:, :] |= jump_y

        # Normales por diferencias centrales, suavizadas.
        dx = np.zeros_like(points)
        dy = np.zeros_like(points)
        dx[:, 1:-1] = points[:, 2:] - points[:, :-2]
        dy[1:-1, :] = points[2:, :] - points[:-2, :]
        normals = cv2.blur(np.cross(dx, dy), (3, 3))
        length = np.linalg.norm(normals, axis=2, keepdims=True)
        normals /= np.maximum(length, 1e-6)

        # Quiebres de orientación (aristas entre caras).
        cos_limit = np.cos(np.radians(ANGULO_QUIEBRE))
        crease = np.zeros((height, width), bool)
        crease[:, 2:-2] |= np.sum(normals[:, 4:] * normals[:, :-4],
                                  axis=2) < cos_limit
        crease[2:-2, :] |= np.sum(normals[4:, :] * normals[:-4, :],
                                  axis=2) < cos_limit

        core = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        free = core & ~jump & ~crease
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            free.astype(np.uint8), connectivity=4)
        return points, valid, free, count, labels, stats

    def evaluate(self, index, points, valid, labels, stats, blue_small):
        """Mide una región y devuelve un candidato (dict) o None."""
        x0, y0, box_w, box_h, area = stats[index]
        if area < AREA_MIN_PX:
            return None
        height, width = labels.shape
        margin = 4
        xa, ya = max(x0 - margin, 0), max(y0 - margin, 0)
        xb = min(x0 + box_w + margin, width)
        yb = min(y0 + box_h + margin, height)
        component = labels[ya:yb, xa:xb] == index
        sub_points = points[ya:yb, xa:xb]

        sample = sub_points[component]
        if len(sample) > MAX_PUNTOS_AJUSTE:
            sample = sample[::len(sample) // MAX_PUNTOS_AJUSTE + 1]
        plane = fit_plane(sample)
        if plane is None:
            return None
        center, axis_1, axis_2, normal, rms, inlier_ratio = plane
        distance = float(center[2])
        if rms > PLANITUD_MAX_MM + 0.01 * distance or inlier_ratio < 0.85:
            return None

        # Recupera el borde que perdió la segmentación: píxeles cercanos que
        # están sobre el mismo plano.
        tolerance = max(10.0, 0.02 * distance)
        off_plane = np.abs((sub_points - center) @ normal)
        grown = cv2.dilate(component.astype(np.uint8), KERNEL_3,
                           iterations=3) > 0
        region = (component | (grown & valid[ya:yb, xa:xb])) \
            & (off_plane < tolerance)

        flat = sub_points[region] - center
        if len(flat) > MAX_PUNTOS_AJUSTE:
            flat = flat[::len(flat) // MAX_PUNTOS_AJUSTE + 1]
        uv = np.stack([flat @ axis_1, flat @ axis_2], axis=1)
        uv = uv.astype(np.float32)
        _, (side_a, side_b), _ = cv2.minAreaRect(uv)
        short, long_ = sorted((side_a, side_b))
        if short <= 0:
            return None
        fill = cv2.contourArea(cv2.convexHull(uv)) / (side_a * side_b)
        aspect = short / long_
        view = center / np.linalg.norm(center)
        tilt = float(np.degrees(np.arccos(min(abs(normal @ view), 1.0))))

        shape_ok = (LADO_MIN_MM <= short and long_ <= LADO_MAX_MM
                    and aspect >= PROPORCION_MIN and fill >= RELLENO_MIN
                    and tilt <= INCLINACION_MAX)

        inner = cv2.erode(region.astype(np.uint8), KERNEL_3) > 0
        if not inner.any():
            inner = region
        color = float(blue_small[ya:yb, xa:xb][inner].mean())

        ys, xs = np.nonzero(region)
        return {
            'origin': (xa, ya),
            'region': region,
            'center': (float(xs.mean() + xa), float(ys.mean() + ya)),
            'sides': (long_, short),
            'distance': distance,
            'fill': fill,
            'aspect': aspect,
            'tilt': tilt,
            'color': color,
            'shape_ok': shape_ok,
            'accepted': shape_ok and color >= COLOR_MIN,
        }

    # ------------------------------------------------------------------
    # Procesamiento principal
    # ------------------------------------------------------------------
    def process(self, rgb_data, depth_data):
        bgr = self.bridge.imgmsg_to_cv2(rgb_data, desired_encoding='bgr8')
        depth = depth_to_mm(self.bridge.imgmsg_to_cv2(depth_data))
        rgb_h, rgb_w = bgr.shape[:2]
        depth_h, depth_w = depth.shape[:2]
        fx, fy, cx, cy = self.depth_intrinsics(depth_w, depth_h)
        offset = ESCALA // 2
        scale_x, scale_y = rgb_w / depth_w, rgb_h / depth_h

        def to_rgb(u, v):
            """Píxel de la profundidad reducida -> píxel RGB."""
            return ((u * ESCALA + offset) * scale_x + DESFASE_X,
                    (v * ESCALA + offset) * scale_y + DESFASE_Y)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        self.last_hsv = hsv
        blue = self.blue_mask(hsv)

        z_small = cv2.medianBlur(depth, 5)[offset::ESCALA, offset::ESCALA]
        small_h, small_w = z_small.shape
        shift = np.float32([[1, 0, -DESFASE_X], [0, 1, -DESFASE_Y]])
        blue_aligned = cv2.warpAffine(blue, shift, (rgb_w, rgb_h))
        blue_small = cv2.resize(blue_aligned, (small_w, small_h),
                                interpolation=cv2.INTER_AREA)
        blue_small = blue_small.astype(np.float32) / 255.0

        points, valid, free, count, labels, stats = self.segment(
            z_small, fx, fy, cx, cy)
        candidates = []
        for index in range(1, count):
            candidate = self.evaluate(
                index, points, valid, labels, stats, blue_small)
            if candidate is not None:
                candidates.append(candidate)

        best, best_score = None, 0.0
        for candidate in candidates:
            if not candidate['accepted']:
                continue
            score = candidate['color'] * candidate['fill'] * candidate['aspect']
            if self.last_center is not None:
                jump = np.hypot(candidate['center'][0] - self.last_center[0],
                                candidate['center'][1] - self.last_center[1])
                score /= 1.0 + jump / 50.0
            if score > best_score:
                best, best_score = candidate, score

        rgb_contour = None
        if best is not None:
            self.lost_frames = 0
            self.last_center = best['center']
            center_rgb = to_rgb(*best['center'])
            x = int(round(center_rgb[0] - rgb_w / 2))
            self.last_x = x
            detected = True

            # Región del cuadrado en la imagen RGB.
            full = np.zeros((small_h, small_w), np.uint8)
            xa, ya = best['origin']
            region = best['region']
            full[ya:ya + region.shape[0], xa:xa + region.shape[1]] = \
                region.astype(np.uint8) * 255
            region_rgb = cv2.resize(full, (rgb_w, rgb_h),
                                    interpolation=cv2.INTER_NEAREST)
            region_rgb = cv2.warpAffine(
                region_rgb, np.float32([[1, 0, DESFASE_X], [0, 1, DESFASE_Y]]),
                (rgb_w, rgb_h))
            contours, _ = cv2.findContours(
                region_rgb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                rgb_contour = max(contours, key=cv2.contourArea)
            self.adapt_color(hsv, cv2.erode(region_rgb, KERNEL_7))

            long_, short = best['sides']
            self.get_logger().info(
                f'Cuadrado: {long_:.0f}x{short:.0f} mm a '
                f'{best["distance"]:.0f} mm, azul {best["color"]:.0%}, x={x}',
                throttle_duration_sec=2.0)
        else:
            self.lost_frames += 1
            if self.lost_frames == FRAMES_PARA_RESETEAR_COLOR:
                self.reset_color()
                self.last_center = None
            if self.last_x is not None and self.lost_frames <= FRAMES_DE_GRACIA:
                x, detected = self.last_x, True
            else:
                x, detected = 0, False

            near_misses = [c for c in candidates if c['shape_ok']]
            if near_misses and self.lost_frames > FRAMES_DE_GRACIA:
                miss = max(near_misses, key=lambda c: c['color'])
                self.get_logger().info(
                    f'Hay un cuadrado de {miss["sides"][0]:.0f}x'
                    f'{miss["sides"][1]:.0f} mm pero solo '
                    f'{miss["color"]:.0%} azul (mínimo {COLOR_MIN:.0%}).',
                    throttle_duration_sec=2.0)

        pos_x = Int16()
        pos_x.data = x
        square_detected = Bool()
        square_detected.data = detected
        self.blue_square_position_pub.publish(pos_x)
        self.blue_square_detected_pub.publish(square_detected)

        if SHOW_MASKS:
            cv2.imshow(WINDOW_DEPTH,
                       self.depth_view(z_small, valid, free, candidates))
            cv2.imshow(WINDOW_COLOR,
                       cv2.bitwise_and(bgr, bgr, mask=blue))
        if SHOW_IMG:
            view = bgr.copy()
            cv2.line(view, (rgb_w // 2, 0), (rgb_w // 2, rgb_h), (0, 0, 0), 2)
            if rgb_contour is not None:
                center = tuple(int(c) for c in to_rgb(*best['center']))
                cv2.drawContours(view, [rgb_contour], -1, (0, 255, 0), 2)
                cv2.line(view, (rgb_w // 2, center[1]), center, (0, 0, 0), 2)
                cv2.putText(view, str(x), center, 1, 2, (0, 0, 0), 1)
                state = (f'DETECTADO x={x}  {best["sides"][0]:.0f}x'
                         f'{best["sides"][1]:.0f}mm  azul '
                         f'{best["color"]:.0%}')
            elif detected:
                state = f'manteniendo x={x}'
            else:
                state = 'buscando'
            lines = (
                state,
                f'color: H {self.hue:.0f}+-{self.hue_tol:.0f}  '
                f'S>={self.s_min:.0f}',
                'click: aprender color  |  r: volver a calibracion',
            )
            self.draw_status(view, lines)
            cv2.imshow(WINDOW_MAIN, view)
        if SHOW_MASKS or SHOW_IMG:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                self.reset_color()
                self.get_logger().info('Color reiniciado a la calibración.')

    # ------------------------------------------------------------------
    # Visualización
    # ------------------------------------------------------------------
    @staticmethod
    def depth_view(z, valid, free, candidates):
        """Profundidad en gris, bordes en rojo oscuro y regiones medidas."""
        span = float(DISTANCIA_MAX - DISTANCIA_MIN)
        gray = 255.0 - (z - DISTANCIA_MIN) / span * 200.0
        gray = np.where(valid, gray, 0).clip(0, 255).astype(np.uint8)
        view = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        view[valid & ~free] = (0, 0, 120)
        view = cv2.resize(view, None, fx=ESCALA, fy=ESCALA,
                          interpolation=cv2.INTER_NEAREST)

        for candidate in candidates:
            if candidate['accepted']:
                color = (0, 255, 0)
            elif candidate['shape_ok']:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
            contours, _ = cv2.findContours(
                candidate['region'].astype(np.uint8), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE, offset=candidate['origin'])
            contours = [c * ESCALA for c in contours]
            cv2.drawContours(view, contours, -1, color, 2)
            u, v = candidate['center']
            label = (f'{candidate["sides"][0]:.0f}x'
                     f'{candidate["sides"][1]:.0f} {candidate["color"]:.0%}')
            cv2.putText(view, label, (int(u * ESCALA) - 40, int(v * ESCALA)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        return view

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