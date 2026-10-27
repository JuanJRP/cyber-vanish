"""Teletransporte: el cuerpo se llena de luz y destellos y se va por un haz vertical."""

import math

import cv2
import numpy as np

from .base import Vanish, glow, register, smoothstep

LIGHT = (255, 232, 195)    # BGR: blanco azulado


@register
class Teleport(Vanish):
    name = "teleport"
    key = "4"
    summary = "te llenas de luz y destellos y te vas por un haz de energía"

    GRAVITY = -40.0
    MAX_PARTICLES = 1200

    def __init__(self, width, height, **options):
        super().__init__(width, height, **options)
        self.light = np.full((height, width, 3), LIGHT, np.uint8)
        self.xs = np.arange(width, dtype=np.float32)
        self.cx, self.half = width / 2, width / 6

    def _plan(self, person, direction, origin):
        # No hay frente que avance: todo el cuerpo se va a la vez. Solo hace falta dónde está.
        cols = np.flatnonzero(person.max(axis=0) > 0.5)
        if len(cols):
            self.cx, self.half = (cols[0] + cols[-1]) / 2, max((cols[-1] - cols[0]) / 2, 20)

    def _transition(self, frame, person, bg, now, t, p, vanishing):
        q = p if vanishing else 1 - p          # 0 = estás, 1 = te fuiste
        env = math.sin(math.pi * p)
        presence = 1 - smoothstep(0.45, 0.85, q)
        shine = smoothstep(0.0, 0.45, q)

        gone = np.full_like(person, 1 - presence)
        out = self._background(frame, bg, person, gone, p, vanishing)

        # Luz que recorre el cuerpo en columnas que ondulan.
        cols = 0.55 + 0.35 * np.cos(self.xs * 0.12 + np.float32(t * 9))
        a = person * (cols * np.float32(shine * presence * 0.9))
        out = cv2.blendLinear(out, self.light, 1 - a, a)

        out = self._beam(out, q, env)
        self._sparkle(person, int(260 * env))
        return out

    def _beam(self, out, q, env):
        """Haz vertical sobre el cuerpo que se estrecha hasta una línea al final."""
        width = self.half * (1 - smoothstep(0.5, 1.0, q)) + 4
        profile = np.exp(-(((self.xs - self.cx) / width) ** 2)) * (0.65 * env)
        row = (profile[:, None] * np.array(LIGHT, np.float32)).astype(np.uint8)[None]
        return cv2.add(out, cv2.resize(row, (self.w, self.h), interpolation=cv2.INTER_NEAREST))

    def _sparkle(self, person, n):
        xs, ys = self._sample(person, n, stride=2)
        k = len(xs)
        if not k:
            return
        colors = np.where(self.rng.random((k, 1)) < 0.7, [255, 255, 255], [255, 220, 120])
        vy = self.rng.uniform(-80, -10, k)
        self._emit(xs, ys, self.rng.normal(0, 10, k), vy, self.rng.uniform(0.15, 0.4, k), np.full(k, 2), colors)

    def _draw_particles(self, out, p):
        layer = self._splat(np.zeros_like(out), p)
        return cv2.add(cv2.add(out, layer), glow(layer, 3.0, 2))
