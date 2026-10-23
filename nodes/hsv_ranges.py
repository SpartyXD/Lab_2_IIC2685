"""Rangos HSV calculados a partir de una región dibujada sobre una imagen."""

import cv2
import numpy as np

HUE_MAX = 179
"""Valor máximo de H en OpenCV (el tono va de 0 a 180)."""


def polygon_mask(shape, points):
    """Máscara uint8 con 255 dentro del polígono descrito por points."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if len(points) < 3:
        return mask

    vertices = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [vertices], 255)
    return mask


def hue_segments(hues):
    """Devuelve los tramos de H que cubren los tonos dados.

    Cada tramo es una tupla ``(minimo, maximo)`` inclusiva. Normalmente hay
    un solo tramo; si los tonos cruzan el 0 (rojo), hay dos.
    """
    values = np.unique(np.asarray(hues, dtype=np.int32))
    if values.size == 0:
        return []
    if values.size == 1:
        return [(int(values[0]), int(values[0]))]

    gaps = np.diff(values)
    wrap_gap = HUE_MAX + 1 - int(values[-1]) + int(values[0])
    largest_gap = int(gaps.max())

    if wrap_gap >= largest_gap:
        return [(int(values[0]), int(values[-1]))]

    split = int(np.argmax(gaps))
    return [
        (0, int(values[split])),
        (int(values[split + 1]), HUE_MAX),
    ]


def mask_ranges(hsv_image, points):
    """Calcula los pares ``(lower, upper)`` HSV de los píxeles del polígono.

    Devuelve una lista vacía si el polígono no cubre ningún píxel. Devuelve
    dos pares cuando el tono cruza el 0 y se necesitan dos máscaras.
    """
    mask = polygon_mask(hsv_image.shape, points)
    if cv2.countNonZero(mask) == 0:
        return []

    selected = mask > 0
    hues = hsv_image[:, :, 0][selected]
    saturations = hsv_image[:, :, 1][selected]
    values = hsv_image[:, :, 2][selected]

    s_min, s_max = int(np.min(saturations)), int(np.max(saturations))
    v_min, v_max = int(np.min(values)), int(np.max(values))

    return [
        (
            np.array([h_min, s_min, v_min], dtype=np.uint8),
            np.array([h_max, s_max, v_max], dtype=np.uint8),
        )
        for h_min, h_max in hue_segments(hues)
    ]


def build_mask(hsv_image, pairs):
    """Máscara binaria que combina todos los rangos HSV recibidos."""
    mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    for lower, upper in pairs:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv_image, lower, upper))
    return mask


def format_hsv_ranges(pairs):
    """Texto con los rangos listos para copiar y pegar en un nodo."""
    lines = []
    if len(pairs) == 1:
        lower, upper = pairs[0]
        lines.append(f'lower = np.array({_as_list(lower)})')
        lines.append(f'upper = np.array({_as_list(upper)})')
    elif pairs:
        for index, (lower, upper) in enumerate(pairs, start=1):
            lines.append(f'lower{index} = np.array({_as_list(lower)})')
            lines.append(f'upper{index} = np.array({_as_list(upper)})')
        lines.append('# combina las mascaras con cv2.bitwise_or')
    return '\n'.join(lines)


def _as_list(values):
    return '[' + ', '.join(str(int(value)) for value in values) + ']'
