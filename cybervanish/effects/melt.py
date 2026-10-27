"""Derretir: el cuerpo se escurre hacia abajo como cera, en chorretones de distinta velocidad."""

import cv2
import numpy as np

from .base import Vanish, register, smoothstep


@register
class Melt(Vanish):
    name = "derretir"
    key = "5"
    summary = "te derrites y escurres hacia abajo como cera"

    def __init__(self, width, height, **options):
        super().__init__(width, height, **options)
        self.xs = np.arange(width, dtype=np.float32).reshape(1, -1)
        self.ys = np.arange(height, dtype=np.float32).reshape(-1, 1)
        self.speed = np.ones((1, width), np.float32)

    def _plan(self, person, direction, origin):
        # Velocidad de caída por columna: ondas suaves más algunos chorretones rápidos y estrechos.
        n = self.w // 6
        speed = cv2.GaussianBlur(self.rng.random((1, n)).astype(np.float32), (0, 0), 4)
        speed = (speed - speed.min()) / (np.ptp(speed) + 1e-6)
        drips = cv2.GaussianBlur((self.rng.random((1, n)) < 0.08).astype(np.float32), (0, 0), 1) * 1.5
        speed = 0.5 + speed + drips
        self.speed = cv2.resize(speed, (self.w, 1), interpolation=cv2.INTER_LINEAR)

    def _transition(self, frame, person, bg, now, t, p, vanishing):
        q = p if vanishing else 1 - p
        k = np.float32(q * q)
        # Cada punto baja más cuanto más abajo está: el cuerpo se estira en chorretones.
        dy = k * self.h * self.speed * (0.25 + 0.75 * self.ys / self.h)
        map_y = self.ys - dy
        map_x = self.xs + np.sin(self.ys * 0.05 + t * 3) * (6 * k)
        map_x, map_y = (m.astype(np.float32) for m in np.broadcast_arrays(map_x, map_y))
        body = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        mask = cv2.remap(person, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # Lo derretido se oscurece un poco y al final se desvanece.
        body = cv2.convertScaleAbs(body, alpha=1 - 0.3 * float(k))
        mask = mask * np.float32(1 - smoothstep(0.55, 1.0, q))

        out = self._background(frame, bg, person, np.ones_like(person), p, vanishing)
        return cv2.blendLinear(out, body, 1 - mask, mask)
