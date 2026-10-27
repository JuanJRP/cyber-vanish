"""Humo: ¡puf! Una nube de humo blanco estalla de golpe, te tapa y, cuando se disipa, ya no estás.

Inspirado en el «¡puf!» de los ninjas del anime al hacer clones o sustituciones. El humo va con
sombreado de dibujo animado: dos tonos planos y contorno, más líneas de impacto al estallar.
"""

import math

import cv2
import numpy as np

from .base import LIFE, MAX_LIFE, SIZE, Vanish, X, Y, body_box, register

POOF = 0.1        # momento (en progreso 0..1) en que estalla la nube
SWAP = 0.16       # momento en que desapareces (o apareces) dentro de la nube, ya tapado
LINES = 0.09      # duración (en progreso) de las líneas de impacto
LIGHT = (246, 247, 250)    # BGR
SHADE = (196, 200, 210)
INK = (110, 112, 125)


@register
class Smoke(Vanish):
    name = "humo"
    key = "9"
    summary = "¡puf! una nube de humo ninja estalla, te tapa y al disiparse ya no estás"

    GRAVITY = -12.0          # el humo sube despacio
    DRAG = 0.9               # sale disparado y frena enseguida: la nube se queda flotando
    MAX_PARTICLES = 600

    def __init__(self, width, height, **options):
        super().__init__(width, height, **options)
        self.fw, self.fh = width // 4, height // 4
        self.light = np.full((height, width, 3), LIGHT, np.uint8)
        self.shade = np.full((height, width, 3), SHADE, np.uint8)
        self.box = body_box(np.zeros((height, width), np.float32), width, height)
        self.burst = False

    def _plan(self, person, direction, origin):
        self.box = body_box(person, self.w, self.h)
        self.burst = False

    def _transition(self, frame, person, bg, now, t, p, vanishing):
        # Al irte, desapareces dentro de la nube; al volver, apareces dentro de ella. Siempre tapado.
        present = p < SWAP if vanishing else p >= SWAP
        gone = np.full_like(person, 0.0 if present else 1.0)
        out = self._background(frame, bg, person, gone, p, vanishing)
        if not self.burst and p >= POOF:
            self.burst = True
            self._poof(person)
        if POOF <= p < POOF + LINES:
            out = self._impact_lines(out, (p - POOF) / LINES)
        return out

    def _poof(self, person):
        """La nube: bolas de humo repartidas por todo el cuerpo que salen disparadas hacia fuera."""
        cx, cy, half, top, bottom = self.box
        xs, ys = self._sample(person, 120, stride=6)
        # Unas cuantas más en el centro para que no quede ningún hueco.
        xs = np.concatenate([xs, cx + self.rng.normal(0, half * 0.4, 30)])
        ys = np.concatenate([ys, cy + self.rng.normal(0, (bottom - top) * 0.2, 30)])
        k = len(xs)
        dx, dy = xs - cx, ys - cy
        dist = np.hypot(dx, dy) + 1
        # Sale disparada pero se queda compacta: una nube algo mayor que tú, no toda la pantalla.
        speed = self.rng.uniform(120, 380, k)
        vx, vy = dx / dist * speed, dy / dist * speed * 0.7 - 40
        size = self.rng.uniform(0.1, 0.17, k) * (bottom - top)
        life = np.minimum(self.rng.uniform(0.8, 1.5, k), 0.6 * self.duration)
        self._emit(xs, ys, vx, vy, life, size, np.zeros((k, 3)))

    def _impact_lines(self, out, k):
        """Líneas que salen del centro al estallar, como en un manga."""
        cx, cy, half, top, bottom = self.box
        layer = out.copy()
        reach = max(half, (bottom - top) / 2)
        for i in range(18):
            a = i / 18 * 2 * math.pi + 0.3 * math.sin(i * 7.1)
            r0, r1 = reach * (0.9 + 0.9 * k), reach * (1.2 + 1.6 * k)
            p0 = (int(cx + math.cos(a) * r0), int(cy + math.sin(a) * r0))
            p1 = (int(cx + math.cos(a) * r1), int(cy + math.sin(a) * r1))
            cv2.line(layer, p0, p1, (255, 255, 255), max(1, int(7 * (1 - k))), cv2.LINE_AA)
        return cv2.addWeighted(out, 0.25, layer, 0.75, 0)

    def _draw_particles(self, out, p):
        # Bolas de humo a 1/4 de resolución. Como en los dibujos animados, no se vuelven transparentes:
        # crecen un poco y, al final de su vida, se encogen hasta desaparecer.
        dens = np.zeros((self.fh, self.fw), np.float32)
        for row in p.tolist():
            f = max(0.0, row[LIFE] / row[MAX_LIFE])
            radius = row[SIZE] * (1 + 0.6 * (1 - f)) * min(1.0, f / 0.4) / 4
            if radius >= 1:
                cv2.circle(dens, (int(row[X] / 4), int(row[Y] / 4)), int(radius), 1.0, -1)
        if not dens.any():
            return out
        dens = cv2.GaussianBlur(dens, (0, 0), 1.5)
        # Dibujo animado: borde nítido, dos tonos (luz arriba a la izquierda) y contorno.
        solid = cv2.GaussianBlur(np.clip((dens - 0.3) * 6, 0, 1), (0, 0), 0.8)
        lit = np.clip(dens - np.roll(dens, (3, 3), axis=(0, 1)), 0, 1) > 0.02
        outline = cv2.morphologyEx((solid > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((2, 2), np.uint8))
        a = cv2.resize(solid, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        tone = cv2.resize(lit.astype(np.float32), (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        smoke = cv2.blendLinear(self.shade, self.light, 1 - tone, tone)
        out = cv2.blendLinear(out, smoke, 1 - a, a)
        line = cv2.resize(outline, (self.w, self.h), interpolation=cv2.INTER_NEAREST) > 0
        out[line & (a > 0.3)] = INK
        return out
