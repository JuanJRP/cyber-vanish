"""Simula desaparecer SIN haber capturado el fondo (sin pulsar B) y guarda cómo queda la habitación.

    python scripts/bench_background.py --image foto_de_tu_habitacion.jpg --out fondo.png

Una "persona" sintética tapa el centro de la foto y se mueve un poco, como en una llamada; el fondo
se aprende solo con lo que va quedando a la vista y luego desapareces. La hoja muestra la habitación
real, el relleno rápido de antes (cv2.inpaint) y el relleno actual, más lo que tarda.
"""

import argparse
import time

import cv2
import numpy as np

from cybervanish.background import BackgroundPlate
from cybervanish.effects import create

W, H = 1280, 720
FPS = 30


def load_room(path):
    if path is None:
        return _synthetic_room()
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"No se pudo leer {path}")
    s = max(W / img.shape[1], H / img.shape[0])
    img = cv2.resize(img, (int(img.shape[1] * s + 1), int(img.shape[0] * s + 1)), interpolation=cv2.INTER_AREA)
    y0, x0 = (img.shape[0] - H) // 2, (img.shape[1] - W) // 2
    return img[y0 : y0 + H, x0 : x0 + W].copy()


def _synthetic_room():
    rng = np.random.default_rng(1)
    room = np.full((H, W, 3), (150, 170, 190), np.uint8)
    room[int(H * 0.72) :] = (70, 95, 120)                                     # suelo
    cv2.rectangle(room, (160, 120), (420, 420), (60, 90, 130), -1)            # estantería
    for y in range(150, 420, 60):
        cv2.line(room, (160, y), (420, y), (40, 60, 90), 6)
    cv2.rectangle(room, (700, 90), (1100, 330), (210, 200, 180), -1)          # ventana
    cv2.line(room, (900, 90), (900, 330), (90, 90, 90), 8)
    return np.clip(room + rng.normal(0, 6, room.shape), 0, 255).astype(np.uint8)


def person_mask(dx):
    """Busto de webcam desplazado dx px, con el mismo engorde que aplica vision.py."""
    m = np.zeros((H, W), np.uint8)
    cx = W // 2 + dx
    cv2.ellipse(m, (cx, 300), (105, 135), 0, 0, 360, 255, -1)
    cv2.ellipse(m, (cx, 760), (330, 330), 0, 180, 360, 255, -1)
    cv2.rectangle(m, (cx - 60, 380), (cx + 60, 470), 255, -1)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50)))
    return np.clip(cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 5), 0, 1)


def run(room, quick_only):
    plate = BackgroundPlate(W, H)
    person_color = np.array([60, 80, 140], np.uint8)
    t = 0.0
    # Unos segundos de llamada moviéndose un poco: el fondo se aprende con lo que queda a la vista.
    for i in range(int(4 * FPS)):
        t = i / FPS
        person = person_mask(int(25 * np.sin(t * 1.5)))
        frame = room.copy()
        frame[person > 0.5] = person_color
        plate.update(frame, person, None if quick_only else t)
    start = time.perf_counter()
    if plate.job is not None:
        plate.job.result()
    wait = time.perf_counter() - start
    fx = create("polvo", W, H, duration=1.0)
    fx.trigger(t, person)
    out = None
    for i in range(int(2 * FPS)):
        out = fx.render(frame, person, plate, t + i / FPS)
    plate.close()
    return out, wait, plate.coverage


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="foto de la habitación (sin nadie); por defecto, una sintética")
    ap.add_argument("--out", default="fondo.png", help="dónde guardar la comparación")
    args = ap.parse_args()

    room = load_room(args.image)
    before, _, coverage = run(room, quick_only=True)
    after, wait, _ = run(room, quick_only=False)
    print(f"Fondo visto: {coverage:.0%}. El relleno bueno tardó {wait:.1f} s más tras 4 s de llamada.")
    tiles = []
    for img, label in [(room, "habitacion real"), (before, "antes: cv2.inpaint"), (after, "ahora")]:
        img = img.copy()
        cv2.putText(img, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4, cv2.LINE_AA)
        tiles.append(cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA))
    cv2.imwrite(args.out, np.hstack(tiles))
    print(f"Comparación en {args.out}")


if __name__ == "__main__":
    main()
