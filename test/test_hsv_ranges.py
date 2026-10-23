import unittest

import numpy as np

from nodes.hsv_ranges import (
    build_mask,
    format_hsv_ranges,
    hue_segments,
    mask_ranges,
)


class HueSegmentsTest(unittest.TestCase):
    def test_hues_contiguos_dan_un_solo_rango(self):
        self.assertEqual(hue_segments([10, 11, 12, 12]), [(10, 12)])

    def test_rojo_que_cruza_cero_da_dos_rangos(self):
        self.assertEqual(
            hue_segments([0, 2, 178, 179]), [(0, 2), (178, 179)]
        )

    def test_rango_amplio_sin_cruce_de_cero_sigue_siendo_uno(self):
        self.assertEqual(hue_segments([90, 100, 120]), [(90, 120)])


class MaskRangesTest(unittest.TestCase):
    def test_solo_considera_pixeles_dentro_del_poligono(self):
        hsv = np.zeros((10, 10, 3), np.uint8)
        hsv[:, :] = (100, 40, 40)
        hsv[2:8, 2:8] = (10, 200, 220)

        pairs = mask_ranges(hsv, [(3, 3), (7, 3), (7, 7), (3, 7)])

        self.assertEqual(len(pairs), 1)
        lower, upper = pairs[0]
        self.assertEqual(lower.tolist(), [10, 200, 220])
        self.assertEqual(upper.tolist(), [10, 200, 220])

    def test_poligono_con_menos_de_tres_puntos_no_devuelve_rangos(self):
        hsv = np.zeros((5, 5, 3), np.uint8)

        self.assertEqual(mask_ranges(hsv, [(0, 0), (4, 4)]), [])


class BuildMaskTest(unittest.TestCase):
    def test_dos_rangos_tapan_los_dos_lados_del_rojo(self):
        hsv = np.zeros((1, 3, 3), np.uint8)
        hsv[0, 0] = (0, 255, 255)
        hsv[0, 1] = (179, 255, 255)
        hsv[0, 2] = (90, 255, 255)

        pairs = [
            (np.array([0, 0, 0]), np.array([5, 255, 255])),
            (np.array([175, 0, 0]), np.array([179, 255, 255])),
        ]

        self.assertEqual(build_mask(hsv, pairs).tolist(), [[255, 255, 0]])


class FormatHsvRangesTest(unittest.TestCase):
    def test_un_rango(self):
        pairs = [
            (np.array([97, 134, 129]), np.array([121, 255, 253]))
        ]

        self.assertEqual(
            format_hsv_ranges(pairs),
            'lower = np.array([97, 134, 129])\n'
            'upper = np.array([121, 255, 253])',
        )

    def test_dos_rangos(self):
        pairs = [
            (np.array([0, 0, 0]), np.array([5, 255, 255])),
            (np.array([175, 0, 0]), np.array([179, 255, 255])),
        ]

        text = format_hsv_ranges(pairs)

        self.assertIn('lower1 = np.array([0, 0, 0])', text)
        self.assertIn('upper1 = np.array([5, 255, 255])', text)
        self.assertIn('lower2 = np.array([175, 0, 0])', text)
        self.assertIn('upper2 = np.array([179, 255, 255])', text)


if __name__ == '__main__':
    unittest.main()
