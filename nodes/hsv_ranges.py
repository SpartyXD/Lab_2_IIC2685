"""Rangos HSV calculados a partir de una región dibujada sobre una imagen."""

import json
import os

import cv2
import numpy as np

HUE_MAX = 179
"""Valor máximo de H en OpenCV (el tono va de 0 a 180)."""

CHANNEL_MAX = 255
"""Valor máximo de S y V en OpenCV."""


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


def pad_pairs(pairs, hue_margin=5, saturation_margin=30, value_margin=40):
    """Amplía los rangos HSV para tolerar cambios de iluminación.

    H se trata como circular (0 a 179), así que un margen que cruza el 0 se
    divide en dos rangos. S y V se recortan al rango 0 a 255.
    """
    padded = []
    for lower, upper in pairs:
        s_min = max(0, int(lower[1]) - saturation_margin)
        s_max = min(CHANNEL_MAX, int(upper[1]) + saturation_margin)
        v_min = max(0, int(lower[2]) - value_margin)
        v_max = min(CHANNEL_MAX, int(upper[2]) + value_margin)

        for h_min, h_max in _padded_hues(
            int(lower[0]), int(upper[0]), int(hue_margin)
        ):
            padded.append(
                (
                    np.array([h_min, s_min, v_min], dtype=np.uint8),
                    np.array([h_max, s_max, v_max], dtype=np.uint8),
                )
            )
    return padded


def _padded_hues(h_min, h_max, margin):
    """Tramos de H que cubren ``[h_min - margin, h_max + margin]``."""
    start = h_min - max(0, margin)
    end = h_max + max(0, margin)
    if end - start + 1 >= HUE_MAX + 1:
        return [(0, HUE_MAX)]
    if start < 0:
        return [(0, end), (HUE_MAX + 1 + start, HUE_MAX)]
    if end > HUE_MAX:
        return [(start, HUE_MAX), (0, end - HUE_MAX - 1)]
    return [(start, end)]


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


def save_ranges(path, pairs):
    """Guarda el primer rango HSV en path con json.dump."""
    if not pairs:
        raise ValueError('No hay rangos HSV para guardar')

    lower, upper = pairs[0]
    data = {
        'lower': _as_int_list(lower),
        'upper': _as_int_list(upper),
    }
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as ranges_file:
        json.dump(data, ranges_file, indent=2)
        ranges_file.write('\n')
    return data


def load_ranges(path):
    """Lee un archivo JSON con las claves ``lower`` y ``upper``."""
    with open(path, encoding='utf-8') as ranges_file:
        data = json.load(ranges_file)

    try:
        lower = _to_hsv(data['lower'])
        upper = _to_hsv(data['upper'])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f'Rango HSV inválido en {path}: {error}') from error
    if np.any(lower > upper):
        raise ValueError(f'Rango HSV inválido en {path}: lower > upper')
    return [(lower, upper)]


def _to_hsv(values):
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError('se esperaban tres valores HSV')

    hsv = [int(value) for value in values]
    if not 0 <= hsv[0] <= HUE_MAX:
        raise ValueError('H debe estar entre 0 y 179')
    if any(value < 0 or value > CHANNEL_MAX for value in hsv[1:]):
        raise ValueError('S y V deben estar entre 0 y 255')
    return np.array(hsv, dtype=np.uint8)


def _as_int_list(values):
    return [int(value) for value in values]


def _as_list(values):
    return '[' + ', '.join(str(int(value)) for value in values) + ']'
