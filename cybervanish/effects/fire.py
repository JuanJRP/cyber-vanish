"""Fuego: un borde de brasa avanza por el cuerpo, chamuscándolo, y suelta chispas que suben."""

import cv2
import numpy as np

from .base import Vanish, below, glow, register

EDGE = 0.03    # anchura de la brasa (en progreso 0..1)
CHAR = 0.11    # banda por delante de la brasa que se chamusca
SOOT = (12, 22, 38)


@register
class Fire(Vanish):
    name = "fuego"
    key = "3"
    summary = "un borde de brasa te consume dejando chispas que suben"

    MIX = (0.35, 0.5, 0.15)
    BLOB_SIGMA = 5
    GRAVITY = -90.0          # las chispas suben cada vez más rápido
    DRAG = 0.96
    MAX_PARTICLES = 2000

    def __init__(self, width, height, **options):
        super().__init__(width, height, **options)
        self.soot = np.full((height, width, 3), SOOT, np.uint8)

    def _transition(self, frame, person, bg, now, t, p, vanishing):
        d = self._front(p, vanishing, CHAR)
        gone = below(d, 0)
        out = self._background(frame, bg, person, gone, p, vanishing)

        # Chamuscado: oscuro y marrón justo detrás de la brasa, normal más lejos.
        c = cv2.multiply(np.clip(1 - (d - EDGE) / (CHAR - EDGE), 0, 1) * (1 - gone), person) * np.float32(0.9)
        out = cv2.blendLinear(out, self.soot, 1 - c, c)

        # Brasa: naranja que parpadea; el halo la satura a amarillo blanco donde es más intensa.
        flicker = self.rng.uniform(0.55, 1.25, (self.h // 16 + 1, self.w // 16 + 1)).astype(np.float32)
        flicker = cv2.resize(flicker, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        edge = cv2.multiply(below(d, EDGE) - gone, person)
        e = cv2.convertScaleAbs(cv2.multiply(edge, flicker), alpha=255)
        layer = cv2.merge([cv2.convertScaleAbs(e, alpha=0.12), cv2.convertScaleAbs(e, alpha=0.55), e])
        out = cv2.add(cv2.add(out, layer), glow(layer, 2.2, 4))

        self._spawn(edge, 45 if vanishing else 20)
        return out

    def _spawn(self, edge, n):
        xs, ys = self._sample(edge, n, stride=2)
        k = len(xs)
        if not k:
            return
        vx = self.rng.normal(0, 45, k) + self.drift[ys, xs] * 15
        vy = self.rng.uniform(-230, -70, k)
        hot = self.rng.random((k, 1))
        colors = (1 - hot) * np.array([0, 110, 255]) + hot * np.array([150, 235, 255])
        self._emit(xs, ys, vx, vy, self.rng.uniform(0.5, 1.5, k), self.rng.choice([1, 2, 3], k), colors)

    def _draw_particles(self, out, p):
        layer = self._splat(np.zeros_like(out), p)
        return cv2.add(cv2.add(out, layer), glow(layer, 1.6, 2))
