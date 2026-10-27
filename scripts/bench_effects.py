"""Mide cuánto tarda cada efecto por frame, sin cámara, y opcionalmente guarda una hoja de capturas.

    python scripts/bench_effects.py                  # 1280x720
    python scripts/bench_effects.py --sheet out.png  # además, capturas a 25/50/75 % de cada efecto
"""

import argparse
import time

import cv2
import numpy as np

from cybervanish.effects import available, create

FPS = 30


class StaticPlate:
    def __init__(self, bg):
        self.bg = bg

    def get(self, now, frame=None, body=None):
        return self.bg


def synthetic_scene(w, h):
    """Habitación con degradado y una "persona" con textura, para que los efectos tengan detalle que mostrar."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = np.dstack([90 + 60 * xx / w, 80 + 40 * yy / h, 70 + 30 * (1 - xx / w)]).astype(np.uint8)
    cv2.rectangle(bg, (int(w * 0.07), int(h * 0.15)), (int(w * 0.25), int(h * 0.6)), (60, 120, 170), -1)
    person = np.zeros((h, w), np.float32)
    cx = w // 2
    cv2.circle(person, (cx, int(h * 0.3)), int(h * 0.12), 1.0, -1)
    cv2.ellipse(person, (cx, int(h * 0.85)), (int(h * 0.3), int(h * 0.42)), 0, 180, 360, 1.0, -1)
    person = cv2.GaussianBlur(person, (0, 0), 3)
    skin = np.dstack([110 + 40 * np.sin(xx / 23), 140 + 40 * np.sin(yy / 17), 200 + 30 * np.cos(xx / 31)])
    frame = bg.copy()
    np.copyto(frame, skin.astype(np.uint8), where=person[..., None] > 0.5)
    return frame, person, StaticPlate(bg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--sheet", help="guarda una hoja de capturas en esta ruta")
    ap.add_argument("--only", nargs="+", choices=list(available()), help="solo estos efectos")
    ap.add_argument("--appear", action="store_true", help="captura la reaparición en vez de la desaparición")
    ap.add_argument("--scale", type=float, default=1 / 3, help="tamaño de cada captura en la hoja")
    ap.add_argument("--at", nargs="+", type=float, default=[0.25, 0.5, 0.75],
                    help="en qué momentos del efecto (0..1) se hacen las capturas")
    args = ap.parse_args()

    frame, person, plate = synthetic_scene(args.width, args.height)
    rows = []
    print(f"{'efecto':10s} {'media':>8s} {'peor':>8s}   (ms por frame a {args.width}x{args.height})")
    for name in args.only or available():
        fx = create(name, args.width, args.height, duration=args.duration)
        fx.trigger(0.0, person, origin=(0.55, 0.45))
        t0 = 0.0
        if args.appear:
            for i in range(int(args.duration * FPS) + 45):
                fx.render(frame, person, plate, i / FPS)
            t0 = (int(args.duration * FPS) + 45) / FPS
            fx.trigger(t0, person, origin=(0.55, 0.45))
        times, shots = [], []
        n = int(args.duration * FPS)
        at = {round(a * n) for a in args.at}
        for i in range(n + 1):
            t = t0 + i / FPS
            start = time.perf_counter()
            out = fx.render(frame, person, plate, t)
            times.append((time.perf_counter() - start) * 1000)
            if i in at:
                shots.append(out)
        print(f"{name:10s} {np.mean(times):8.1f} {np.max(times):8.1f}")
        if args.sheet:
            size = (int(args.width * args.scale), int(args.height * args.scale))
            row = np.hstack([cv2.resize(s, size, interpolation=cv2.INTER_AREA) for s in shots])
            cv2.putText(row, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            rows.append(row)
    if args.sheet:
        cv2.imwrite(args.sheet, np.vstack(rows))
        print(f"Capturas en {args.sheet}")


if __name__ == "__main__":
    main()
