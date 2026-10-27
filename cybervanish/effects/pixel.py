"""Pixel: el cuerpo se pixela a saltos con paleta retro, se vuelve un mosaico de bloques con relieve
y los bloques se sueltan, giran y rebotan. Al volver, los bloques suben volando a su sitio."""

import math

import cv2
import numpy as np

from .base import COLOR, KIND, LIFE, SIZE, VX, VY, Vanish, X, Y, normalize, register, smoothstep

BLOCK = 32                       # tamaño final (px) de los bloques
STEPS = (1, 2, 4, 8, 16, BLOCK)  # el pixelado va a saltos, como en los juegos de 8 bits
SPLIT = 0.5                      # parte del efecto que dura el pixelado; el resto, la caída de bloques
LEAD = 0.45                      # segundos que tarda un bloque en volver volando a su sitio


@register
class Pixel(Vanish):
    name = "pixel"
    key = "6"
    summary = "te pixelas a saltos en bloques retro con relieve que se sueltan, giran y rebotan"

    GRAVITY = 1600.0
    DRAG = 1.0
    MAX_PARTICLES = 1500

    def __init__(self, width, height, **options):
        super().__init__(width, height, **options)
        self.gw, self.gh = -(-width // BLOCK), -(-height // BLOCK)
        self.block_order = np.zeros((self.gh, self.gw), np.float32)
        self.block_drift = np.ones((self.gh, self.gw), np.float32)
        self.land_at = np.zeros((self.gh, self.gw))
        self.prev_level = 0.0
        self.bevels = {}

    def _plan(self, person, direction, origin):
        pos, drift = self._travel(person, self.gw, self.gh, direction, origin)
        self.block_order = normalize(0.6 * pos + 0.4 * self.rng.random((self.gh, self.gw)))
        self.block_drift = drift
        appearing = self.state == "vanished"     # _plan se llama justo antes de cambiar de estado
        self.prev_level = 2.0 if appearing else 0.0   # al aparecer, por encima de todo: ningún bloque se queda fuera
        self.land_at = np.full((self.gh, self.gw), np.inf if appearing else 0.0)

    def _transition(self, frame, person, bg, now, t, p, vanishing):
        q = p if vanishing else 1 - p
        size = STEPS[min(len(STEPS) - 1, int(smoothstep(0, SPLIT, q) * len(STEPS)))]
        retro = _posterize(frame, int(1 + 56 * smoothstep(0.05, SPLIT, q)))
        pix, mask = _pixelate(retro, size), _pixelate(person, size) > 0.3
        if size >= 8:
            pix = cv2.multiply(pix, self._bevel(size), dtype=cv2.CV_8U)

        level = float(smoothstep(SPLIT, 1.0, q))
        loose = self.block_order < level
        if vanishing and size == BLOCK:
            self._drop(retro, person, loose & (self.block_order >= self.prev_level))
        elif not vanishing:
            settled = ~loose & (self.block_order < self.prev_level)
            self.land_at[settled] = now + LEAD
            self._fly_in(retro, person, settled)
        self.prev_level = level

        shown = ~loose & (self.land_at <= now)
        shown = cv2.resize(shown.astype(np.uint8), (self.gw * BLOCK, self.gh * BLOCK),
                           interpolation=cv2.INTER_NEAREST)[: self.h, : self.w]
        mask &= shown > 0

        out = self._background(frame, bg, person, np.ones_like(person), p, vanishing)
        np.copyto(out, pix, where=mask[..., None])
        return out

    def _bevel(self, size):
        """Relieve de cada bloque: borde de arriba e izquierda claro, de abajo y derecha oscuro."""
        if size not in self.bevels:
            edge = max(1, size // 8)
            tile = np.ones((size, size), np.float32)
            tile[:edge, :] = tile[:, :edge] = 1.3
            tile[-edge:, :] = tile[:, -edge:] = 0.65
            reps = (-(-self.h // size), -(-self.w // size))
            full = np.tile(tile, reps)[: self.h, : self.w]
            self.bevels[size] = cv2.merge([full, full, full])
        return self.bevels[size]

    def _blocks(self, retro, person, which):
        colors = cv2.resize(retro, (self.gw, self.gh), interpolation=cv2.INTER_AREA)
        body = cv2.resize(person, (self.gw, self.gh), interpolation=cv2.INTER_AREA) > 0.3
        gy, gx = np.nonzero(which & body)
        return gx, gy, colors[gy, gx]

    def _drop(self, retro, person, new):
        """Los bloques que se acaban de soltar salen despedidos girando."""
        gx, gy, colors = self._blocks(retro, person, new)
        k = len(gx)
        if not k:
            return
        vx = self.block_drift[gy, gx] * self.rng.uniform(20, 180, k) + self.rng.normal(0, 50, k)
        vy = self.rng.uniform(-420, -80, k)
        spin = self.rng.uniform(-5, 5, k)
        self._emit(gx * BLOCK + 1, gy * BLOCK + 1, vx, vy, np.full(k, 2.2), np.full(k, BLOCK - 2), colors, spin)

    def _fly_in(self, retro, person, settled):
        """Al volver, cada bloque sube desde abajo y aterriza en su sitio."""
        gx, gy, colors = self._blocks(retro, person, settled)
        k = len(gx)
        if not k:
            return
        tx, ty = gx * BLOCK + 1, gy * BLOCK + 1
        sx = tx + self.rng.normal(0, 160, k)
        sy = self.h + self.rng.uniform(40, 260, k)
        self._emit_homing(sx, sy, tx, ty, np.full(k, LEAD), np.full(k, BLOCK - 2), colors, self.rng.uniform(-8, 8, k))

    def _forces(self, p, free, dt):
        # Rebotan contra el borde de abajo, perdiendo fuerza.
        floor = free & (p[:, Y] + p[:, SIZE] > self.h - 2) & (p[:, VY] > 0)
        p[floor, VY] *= -0.42
        p[floor, VX] *= 0.8
        p[floor, KIND] *= 0.6

    def _draw_particles(self, out, p):
        out = out.copy()
        for row in p.tolist():
            s = row[SIZE]
            angle = math.degrees(row[KIND] * row[LIFE])   # en los que vuelan a su sitio, llega a 0 al aterrizar
            box = cv2.boxPoints(((row[X] + s / 2, row[Y] + s / 2), (s, s), angle)).astype(np.int32)
            color = tuple(row[COLOR])
            light = tuple(min(255.0, c * 1.3) for c in color)
            dark = tuple(c * 0.6 for c in color)
            cv2.fillConvexPoly(out, box, color, cv2.LINE_AA)
            cv2.polylines(out, [box], True, dark, 2, cv2.LINE_AA)
            cv2.polylines(out, [box[:3]], False, light, 2, cv2.LINE_AA)
        return out


def _pixelate(img, size):
    if size <= 1:
        return img
    h, w = img.shape[:2]
    gw, gh = -(-w // size), -(-h // size)
    small = cv2.resize(img, (gw, gh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (gw * size, gh * size), interpolation=cv2.INTER_NEAREST)[:h, :w]


def _posterize(img, step):
    """Reduce los colores a niveles de `step` en `step`: paleta de consola antigua."""
    if step <= 1:
        return img
    lut = np.minimum((np.arange(256) // step) * step + step // 2, 255).astype(np.uint8)
    return cv2.LUT(img, lut)
