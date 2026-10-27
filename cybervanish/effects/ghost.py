"""Fantasma: el cuerpo se vuelve translúcido y ondula como un espejismo hasta desvanecerse."""

import math

import cv2
import numpy as np

from .base import Vanish, glow, register, smoothstep


@register
class Ghost(Vanish):
    name = "fantasma"
    key = "7"
    summary = "te vuelves translúcido y ondulas como un espejismo hasta desvanecerte"

    def __init__(self, width, height, **options):
        super().__init__(width, height, **options)
        self.xs = np.arange(width, dtype=np.float32).reshape(1, -1)
        self.ys = np.arange(height, dtype=np.float32).reshape(-1, 1)

    def _plan(self, person, direction, origin):
        pass   # se desvanece todo a la vez

    def _transition(self, frame, person, bg, now, t, p, vanishing):
        q = p if vanishing else 1 - p
        amp = 20 * float(smoothstep(0, 0.7, q))
        map_x, map_y = np.broadcast_arrays(
            self.xs + amp * np.sin(self.ys / 18 + t * 8),
            self.ys + 0.4 * amp * np.sin(self.xs / 25 + t * 6),
        )
        map_x, map_y = map_x.astype(np.float32), map_y.astype(np.float32)
        body = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mask = cv2.remap(person, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # Más frío y luminoso a medida que se va.
        body = cv2.add(cv2.convertScaleAbs(body, alpha=1 - 0.15 * q), (45 * q, 25 * q, 0, 0))

        out = self._background(frame, bg, person, np.ones_like(person), p, vanishing)
        a = mask * np.float32(1 - smoothstep(0.15, 1.0, q))
        out = cv2.blendLinear(out, body, 1 - a, a)
        # Halo etéreo alrededor de la silueta.
        aura = cv2.multiply(body, cv2.merge([a, a, a]), dtype=cv2.CV_8U)
        return cv2.add(out, glow(aura, 0.7 * math.sin(math.pi * q), 6))
