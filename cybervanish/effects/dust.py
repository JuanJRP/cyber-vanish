"""Polvo: el cuerpo se vuelve ceniza y se deshace en granos que se lleva el viento."""

import cv2
import numpy as np

from .base import Vanish, below, register

EDGE = 0.035   # banda (en progreso 0..1) que se está deshaciendo en granos
ASH = 0.12     # banda por delante donde el color se apaga a gris ceniza


@register
class Dust(Vanish):
    name = "polvo"
    key = "2"
    summary = "te vuelves ceniza y te deshaces en polvo que se lleva el viento"

    GRID = 3
    MIX = (0.55, 0.3, 0.15)
    BLOB_SIGMA = 2
    GRAVITY = -30.0          # el viento levanta los granos poco a poco
    DRAG = 0.985
    MAX_PARTICLES = 7000

    def _transition(self, frame, person, bg, now, t, p, vanishing):
        d = self._front(p, vanishing, ASH)
        gone = below(d, 0)
        out = self._background(frame, bg, person, gone, p, vanishing)

        # Justo antes de irse, la piel se apaga a gris ceniza.
        ashen = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        ashen = cv2.add(cv2.convertScaleAbs(ashen, alpha=0.75), (28, 36, 44, 0))
        a = cv2.multiply(np.clip(1 - d / ASH, 0, 1) * (1 - gone), person) * np.float32(0.85)
        out = cv2.blendLinear(out, ashen, 1 - a, a)

        edge = cv2.multiply(below(d, EDGE) - gone, person)
        self._spawn(edge, frame, ashen, 450 if vanishing else 160, vanishing)
        return out

    def _spawn(self, edge, frame, ashen, n, vanishing):
        xs, ys = self._sample(edge, n)
        k = len(xs)
        if not k:
            return
        vx = self.drift[ys, xs] * self.rng.uniform(60, 280, k) + self.rng.normal(0, 30, k)
        vy = self.rng.uniform(-110, 25, k)
        life = self.rng.uniform(0.8, 2.0, k)
        # Mezcla del color real y de la ceniza: los granos siguen recordando de qué parte salen.
        mix = self.rng.uniform(0.3, 0.9, (k, 1))
        colors = frame[ys, xs] * (1 - mix) + ashen[ys, xs] * mix
        if not vanishing:
            # Al volver, los granos llegan desde fuera hacia su sitio.
            life = life * 0.5
            xs, ys = xs - vx * life, ys - vy * life
        self._emit(xs, ys, vx, vy, life, self.rng.choice([1, 2, 2, 3], k), colors)

    def _draw_particles(self, out, p):
        # Los granos no se oscurecen al morir: se encogen.
        q = p.copy()
        q[:, 6] *= np.sqrt(q[:, 4] / q[:, 5])
        return self._splat(out.copy(), q, fade=False)
