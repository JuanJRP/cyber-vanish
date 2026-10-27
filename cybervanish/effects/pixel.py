"""Pixel: el cuerpo se pixela cada vez más grueso y luego los bloques se sueltan y caen."""

import cv2
import numpy as np

from .base import Vanish, normalize, register, smoothstep

BLOCK = 40     # tamaño final (px) de los bloques
SPLIT = 0.5    # parte del efecto que dura el pixelado; el resto, la caída de bloques


@register
class Pixel(Vanish):
    name = "pixel"
    key = "6"
    summary = "te pixelas en bloques cada vez más grandes que se sueltan y caen"

    GRAVITY = 1500.0
    DRAG = 1.0
    MAX_PARTICLES = 3000

    def __init__(self, width, height, **options):
        super().__init__(width, height, **options)
        self.gw, self.gh = -(-width // BLOCK), -(-height // BLOCK)
        self.block_order = np.zeros((self.gh, self.gw), np.float32)
        self.block_drift = np.ones((self.gh, self.gw), np.float32)
        self.prev_level = 0.0

    def _plan(self, person, direction, origin):
        pos, drift = self._travel(person, self.gw, self.gh, direction, origin)
        self.block_order = normalize(0.6 * pos + 0.4 * self.rng.random((self.gh, self.gw)))
        self.block_drift = drift
        self.prev_level = 0.0

    def _transition(self, frame, person, bg, now, t, p, vanishing):
        q = p if vanishing else 1 - p
        size = int(1 + (BLOCK - 1) * smoothstep(0, SPLIT, q))
        pix, mask = _pixelate(frame, size), _pixelate(person, size) > 0.3

        # Bloques que ya se soltaron.
        level = float(smoothstep(SPLIT, 1.0, q))
        loose = self.block_order < level
        if vanishing and size == BLOCK:
            self._drop(frame, person, loose & (self.block_order >= self.prev_level))
        self.prev_level = level
        loose = cv2.resize(loose.astype(np.uint8), (self.gw * BLOCK, self.gh * BLOCK),
                           interpolation=cv2.INTER_NEAREST)[: self.h, : self.w]
        mask &= loose == 0

        out = self._background(frame, bg, person, np.ones_like(person), p, vanishing)
        np.copyto(out, pix, where=mask[..., None])
        return out

    def _drop(self, frame, person, new):
        """Los bloques que se acaban de soltar salen despedidos y caen."""
        colors = cv2.resize(frame, (self.gw, self.gh), interpolation=cv2.INTER_AREA)
        body = cv2.resize(person, (self.gw, self.gh), interpolation=cv2.INTER_AREA) > 0.3
        gy, gx = np.nonzero(new & body)
        k = len(gx)
        if not k:
            return
        vx = self.block_drift[gy, gx] * self.rng.uniform(20, 160, k) + self.rng.normal(0, 40, k)
        vy = self.rng.uniform(-320, -60, k)
        self._emit(gx * BLOCK + 1, gy * BLOCK + 1, vx, vy, np.full(k, 2.0), np.full(k, BLOCK - 2), colors[gy, gx])


def _pixelate(img, size):
    if size <= 1:
        return img
    h, w = img.shape[:2]
    gw, gh = -(-w // size), -(-h // size)
    small = cv2.resize(img, (gw, gh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (gw * size, gh * size), interpolation=cv2.INTER_NEAREST)[:h, :w]
