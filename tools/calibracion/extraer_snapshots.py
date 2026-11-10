#!/usr/bin/env python3
"""Extrae snapshots RGB/profundidad de un rosbag2 SQLite, sin ROS.

Uso:
    python3 extraer_snapshots.py <bag.db3|directorio_del_bag> <salida>
    python3 extraer_snapshots.py ... --cada 1.5
    python3 extraer_snapshots.py ... --indices 0,100,500

Si la base está corrupta, primero recuperarla:
    sqlite3 rosbag2_..._0.db3 ".recover" | sqlite3 recuperado.db
"""
import argparse
import csv
import os
import sqlite3
import struct

import cv2
import numpy as np


def parse_image(data):
    endian = '<' if data[1] == 1 else '>'
    off = 4
    sec, nsec = struct.unpack_from(endian + 'iI', data, off)
    off += 8
    largo = struct.unpack_from(endian + 'I', data, off)[0]
    off += 4
    frame = data[off:off + max(largo - 1, 0)].decode(errors='replace')
    off += largo
    off = (off + 3) & ~3
    alto, ancho = struct.unpack_from(endian + 'II', data, off)
    off += 8
    largo = struct.unpack_from(endian + 'I', data, off)[0]
    off += 4
    encoding = data[off:off + max(largo - 1, 0)].decode(errors='replace')
    off += largo
    off += 1  # is_bigendian
    off = (off + 3) & ~3
    step = struct.unpack_from(endian + 'I', data, off)[0]
    off += 4
    largo = struct.unpack_from(endian + 'I', data, off)[0]
    off += 4
    return sec, nsec, frame, alto, ancho, encoding, step, largo, off


def cargar_topic(conexion, topic, encoding):
    fila = conexion.execute(
        "select id from topics where name=?", (topic,)
    ).fetchone()
    if fila is None:
        raise SystemExit(f'No existe el topico {topic}')
    topic_id = fila[0]
    salida = []
    for timestamp, data in conexion.execute(
        "select timestamp, data from messages where topic_id=? "
        "order by timestamp", (topic_id,)
    ):
        if not data or len(data) < 100:
            continue
        _, _, _, alto, ancho, enc, step, _, off = parse_image(data)
        if enc != encoding:
            continue
        bytes_pixel = 3 if encoding in ('rgb8', 'bgr8') else (
            2 if encoding == '16UC1' else 4
        )
        ancho_bytes = ancho * bytes_pixel
        if step < ancho_bytes or off + step * alto > len(data):
            continue
        filas = np.frombuffer(
            data, dtype=np.uint8, count=step * alto, offset=off
        ).reshape(alto, step)
        if encoding in ('rgb8', 'bgr8'):
            imagen = filas[:, :ancho_bytes].copy().reshape(alto, ancho, 3)
            if encoding == 'rgb8':
                imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
        elif encoding == '16UC1':
            imagen = filas[:, :ancho_bytes].copy().view('<u2').reshape(
                alto, ancho
            )
        else:
            imagen = filas[:, :ancho_bytes].copy().view('<f4').reshape(
                alto, ancho
            )
            imagen = np.nan_to_num(imagen * 1000.0).astype(np.uint16)
        salida.append((timestamp, imagen))
    return salida


def buscar_db(ruta):
    if os.path.isfile(ruta):
        return ruta
    archivos = [a for a in os.listdir(ruta) if a.endswith('.db3')]
    if not archivos:
        raise SystemExit(f'No hay .db3 en {ruta}')
    return os.path.join(ruta, sorted(archivos)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bag')
    parser.add_argument('salida')
    parser.add_argument('--cada', type=float, default=1.5)
    parser.add_argument('--indices', default='')
    parser.add_argument('--max', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.salida, exist_ok=True)
    db = buscar_db(args.bag)
    conexion = sqlite3.connect('file:' + db + '?mode=ro', uri=True)
    rgb = cargar_topic(conexion, '/image_raw', 'rgb8')
    depth = cargar_topic(conexion, '/depth/image_raw', '16UC1')
    if not rgb or not depth:
        raise SystemExit('No se encontraron frames RGB/profundidad')
    tiempos_depth = np.array([t for t, _ in depth])
    inicio = rgb[0][0]

    if args.indices:
        indices = [int(v) for v in args.indices.split(',') if v.strip()]
    else:
        cantidad = int((rgb[-1][0] - inicio) / (args.cada * 1e9)) + 1
        indices = list(range(0, len(rgb), max(1, len(rgb) // max(cantidad, 1))))

    filas = []
    for numero, indice in enumerate(indices):
        if indice >= len(rgb):
            continue
        timestamp, imagen = rgb[indice]
        cercano = int(np.argmin(np.abs(tiempos_depth - timestamp)))
        timestamp_depth, profundidad = depth[cercano]
        prefijo = os.path.join(args.salida, f'{numero:03d}')
        cv2.imwrite(prefijo + '_rgb.png', imagen)
        np.save(prefijo + '_depth.npy', profundidad)
        visible = cv2.applyColorMap(
            np.clip(profundidad / 4000.0 * 255.0, 0, 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        visible[profundidad == 0] = (0, 0, 0)
        cv2.imwrite(prefijo + '_depth.png', visible)
        filas.append({
            'archivo': os.path.basename(prefijo),
            'indice_rgb': indice,
            't_rgb': timestamp,
            't_depth': timestamp_depth,
            'segundos': (timestamp - inicio) / 1e9,
        })
        if args.max and len(filas) >= args.max:
            break

    if not filas:
        print('No se generaron snapshots')
        return
    with open(os.path.join(args.salida, 'index.csv'), 'w', newline='') as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0].keys()))
        escritor.writeheader()
        escritor.writerows(filas)
    print(f'{len(filas)} snapshots en {args.salida}')


if __name__ == '__main__':
    main()
