# Calibración y filtro temporal del detector

Bag usado: `rosbag2_2026_11_03-16_31_32` (66 s, 1833 frames RGB y 1830 de
profundidad útiles). La profundidad del Kinect está en milímetros (`16UC1`).

## Recuperar el bag

La base SQLite original está corrupta en algunas páginas. Se recuperaron 7330
de 7950 mensajes con:

```bash
sqlite3 rosbag2_2026_11_03-16_31_32_0.db3 ".recover" | sqlite3 recuperado.db
```

El resto de este documento usa `recuperado.db`.

## Snapshots

`extraer_snapshots.py` lee el SQLite y el CDR directamente, sin ROS:

```bash
python3 tools/calibracion/extraer_snapshots.py recuperado.db snapshots --cada 1.5
python3 tools/calibracion/extraer_snapshots.py recuperado.db snapshots --indices 0,611,1222
```

Genera `NNN_rgb.png`, `NNN_depth.npy`, `NNN_depth.png` e `index.csv`.

## Calibración del detector original

Se revisaron visualmente snapshots a 1.5 s. Los intervalos con cuadrado visible
son 0.0–8.8, 8.8–12.8, 31.4–40.8 y 46.4–60.0 s. La referencia de centro se
obtuvo con una máscara HSV estricta + profundidad y se validó visualmente.

Convención: un positivo detectado con `|Δx| > 80 px` cuenta como FN; los FP se
cuentan solo en frames negativos.

| Detector | Precisión | Recall | F1 | FP en negativos | Error x mediana |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original sin calibrar | 0.776 | 0.528 | 0.628 | 164 / 757 (21.7 %) | 0.6 px |
| Original calibrado | **0.998** | 0.787 | 0.880 | 2 / 757 (0.3 %) | 0.7 px |
| Original calibrado + filtro temporal | 0.995 | **0.815** | **0.896** | 4 / 757 (0.5 %) | 0.7 px |

Con el filtro temporal, la detección por intervalo queda en 155/265, 68/121,
281/282 y 377/408 para 0–8.8, 8.8–12.8, 31.4–40.8 y 46.4–60.0 s.

Cambios de calibración aplicados:

- `config/hsv_ranges.json`: `[104,145,90]–[120,255,255]`. El rango anterior
  `[92,104,89]–[126,255,255]` capturaba jeans y paredes.
- `DISTANCIA_MAX`: `1500 → 2200` mm. Entre 35 y 40 s el cuadrado está a
  1.6–1.9 m, fuera del rango anterior.
- `ESPESOR_MAX`: `60 → 500` mm. El cuadrado inclinado/lejano tiene más
  variación de profundidad de la que aceptaba el filtro.
- `minInertiaRatio`: `0.3 → 0.25`. Recupera cuadrados parcialmente ocluidos.

## Filtro temporal en `nodes/parte3/detector_de_objeto.py`

El detector asume que el cuadrado es un objeto físico que se mueve, no que
desaparece y reaparece en otro lugar. Por eso mantiene:

- `ultima_pos`: última posición válida publicada.
- `hay_ultima_pos`: distingue “sin objeto” de “objeto centrado en x=0”.
- `contador_sin_cuadrado`: frames seguidos sin detección o con una detección
  lejos de `ultima_pos`.
- `contador_candidato`: frames seguidos con una detección consistente cuando
  todavía no hay posición previa.

Reglas:

1. Si la detección está a menos de `TOLERANCIA_POS=80 px` de `ultima_pos`, se
   acepta y se publica.
2. Si no hay detección o la detección está lejos, se publica `ultima_pos` con
   `detected=True` durante `MAX_FRAMES_SIN_CUADRADO=10` frames.
3. Si se superan esos 10 frames, se publica `0` con `detected=False`.
4. Si no había posición previa, la primera detección debe repetirse
   `FRAMES_PARA_APARICION=3` frames para engancharse. Es el “contador menor”
   para una aparición.
5. Excepción: si la detección está cerca de un borde de la cámara
   (`MARGEN_BORDE=45 px`) o de los límites de distancia
   (`MARGEN_DISTANCIA=150 mm`), se acepta el salto de inmediato. Si la última
   posición era especial, la pérdida se resuelve en
   `MAX_FRAMES_SIN_CUADRADO_BORDE=3` frames.

## Estructura del paquete

Los nodos quedaron ordenados por parte del laboratorio:

```text
nodes/
├── parte1/   # control P/PI y trayectoria
├── parte2/   # follow the carrot
└── parte3/   # detección y seguimiento del cuadrado
```

El `CMakeLists.txt` instala los archivos en `lib/lab2` conservando el nombre
base. Por eso los launches siguen usando `exec="detector_de_objeto.py"`,
`exec="robot_controller.py"`, etc., sin cambios.

## Compilar y probar

```bash
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch lab2 seguimiento_de_objeto.launch
```

Para el seguimiento del cuadrado se usa el launch existente
`seguimiento_de_objeto.launch`. El seguidor `seguidor_de_objeto.py` no cambia:
consume `/blue_square/position` y `/blue_square/detected`.
