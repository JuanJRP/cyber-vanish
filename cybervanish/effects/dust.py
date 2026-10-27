"""Polvo: el cuerpo se vuelve ceniza y se deshace en granos que se lleva un viento racheado.

Al volver, los granos llegan desde el aire y aterrizan cada uno en su sitio justo cuando aparece.
"""

import math

import cv2
import numpy as np

from .base import COLOR, SIZE, VX, VY, Vanish, X, Y, band, below, gray3, register

EDGE = 0.03    # banda (en progreso 0..1) que se está deshaciendo en granos
ASH = 0.11     # banda por delante donde el color se apaga a gris ceniza
RIM = 0.012    # filo de carbón justo donde se suelta el polvo
CHARCOAL = (26, 28, 32)
LEAD = 0.7     # segundos que tardan los granos en volver a su sitio


@register
class Dust(Vanish):
    name = "polvo"
    key = "2"
    summary = "te vuelves ceniza y te deshaces en polvo que se lleva el viento"

    GRID = 3
    MIX = (0.55, 0.3, 0.15)
    BLOB_SIGMA = 2
    GRAVITY = -18.0          # el viento levanta los granos poco a poco
    DRAG = 0.99
    MAX_PARTICLES = 9000

    def __init__(self, width, height, **options):
        super().__init__(width, height, **options)
        self.charcoal = np.full((height, width, 3), CHARCOAL, np.uint8)
        self.phase = self.rng.uniform(0, 2 * math.pi, 3)

    def _transition(self, frame, person, bg, now, t, p, vanishing):
        d = self._front(p, vanishing, ASH)
        gone = below(d, 0)
        present = 1 - gone
        out = self._background(frame, bg, person, gone, p, vanishing)

        # Justo antes de irse, la piel se apaga a gris ceniza...
        ashen = cv2.add(cv2.convertScaleAbs(gray3(frame), alpha=0.72), (30, 37, 46, 0))
        a = cv2.multiply(np.clip(1 - d / ASH, 0, 1) * present, person) * np.float32(0.85)
        out = cv2.blendLinear(out, ashen, 1 - a, a)
        # ...y el filo que se está soltando se ve carbonizado.
        rim = cv2.multiply(band(d, 0, RIM), person) * np.float32(0.7)
        out = cv2.blendLinear(out, self.charcoal, 1 - rim, rim)

        if vanishing:
            edge = cv2.multiply(band(d, 0, EDGE), person)
            self._blow(edge, frame, ashen, 520)
        else:
            self._gather(d, person, frame, ashen, 260)
        return out

    def _colors(self, frame, ashen, xs, ys):
        # Mezcla del color real y de la ceniza: los granos siguen recordando de qué parte salen.
        mix = self.rng.uniform(0.25, 0.9, (len(xs), 1))
        return frame[ys, xs] * (1 - mix) + ashen[ys, xs] * mix

    def _sizes(self, k):
        # Sobre todo granos finos; algún trozo más grande que se desprende entero.
        return np.where(self.rng.random(k) < 0.06, self.rng.integers(4, 6, k), self.rng.choice([1, 1, 2, 2, 3], k))

    def _blow(self, edge, frame, ashen, n):
        xs, ys = self._sample(edge, n)
        k = len(xs)
        if not k:
            return
        vx = self.drift[ys, xs] * self.rng.uniform(50, 260, k) + self.rng.normal(0, 30, k)
        vy = self.rng.uniform(-120, 20, k)
        self._emit(xs, ys, vx, vy, self.rng.uniform(0.9, 2.2, k), self._sizes(k), self._colors(frame, ashen, xs, ys))

    def _gather(self, d, person, frame, ashen, n):
        arriving, speed = self._arriving(d, ASH, LEAD)
        xs, ys = self._sample(cv2.multiply(arriving, person), n)
        k = len(xs)
        if not k:
            return
        life = np.maximum(-d[ys, xs] / speed, 0.05)
        # Vienen del lado hacia el que se fueron, arrastrados por el mismo viento.
        sx = xs + self.drift[ys, xs] * self.rng.uniform(120, 420, k) + self.rng.normal(0, 40, k)
        sy = ys + self.rng.uniform(-160, 30, k)
        self._emit_homing(sx, sy, xs, ys, life, self._sizes(k), self._colors(frame, ashen, xs, ys))

    def _forces(self, p, free, dt):
        # Viento racheado: remolinos que cambian con el tiempo y la posición.
        x, y, t = p[free, X], p[free, Y], self.now
        p[free, VX] += (70 * np.sin(y / 37 + t * 1.3 + self.phase[0]) + 35 * np.sin(y / 11 - t * 2.3)) * dt * 2
        p[free, VY] += (55 * np.cos(x / 43 - t * 1.1 + self.phase[1]) + 25 * np.sin(x / 9 + self.phase[2])) * dt * 2

    def _draw_particles(self, out, p):
        # Los granos no se oscurecen al morir: se encogen.
        q = p.copy()
        q[:, SIZE] *= np.sqrt(np.clip(q[:, 4] / q[:, 5], 0, 1))
        out = self._splat(out.copy(), q, fade=False, max_size=5)
        # Un poco de luz en los granos grandes para que se lean como trozos con volumen.
        big = q[q[:, SIZE] >= 3.5]
        if len(big):
            hi = big.copy()
            hi[:, COLOR] = np.minimum(hi[:, COLOR] * 1.35 + 20, 255)
            hi[:, SIZE] = 1
            self._splat(out, hi, fade=False, max_size=1)
        return out
