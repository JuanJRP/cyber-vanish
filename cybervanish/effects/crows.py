"""Cuervos: el cuerpo se oscurece como tinta y se deshace en una bandada de cuervos que sale volando
entre plumas negras. Al volver, los cuervos llegan volando y se funden en tu silueta.

Inspirado en los ninjas del anime que se disuelven en cuervos.
"""

import math

import cv2
import numpy as np

from .base import KIND, LIFE, MAX_LIFE, SIZE, TX, VX, VY, Vanish, X, Y, band, below, body_box, register

EDGE = 0.04       # banda (en progreso) que se está deshaciendo en cuervos
INK_BAND = 0.13   # banda por delante donde el cuerpo se vuelve tinta
INK = (22, 12, 20)          # BGR: negro con un punto morado
CROW = (18, 14, 16)
SHEEN = (90, 60, 70)        # reflejo azulado de las plumas
FEATHER = -1.0              # KIND de las plumas; en los cuervos KIND guarda la frecuencia del aleteo
LEAD = 0.6                  # segundos que tardan los cuervos en volver a su sitio
# Contorno del ala (adelante, hacia fuera) en tamaños de cuervo: borde de ataque, plumas primarias
# separadas en la punta y borde de salida. Se escala hacia fuera según el aleteo.
WING = [(0.2, 0.08), (0.12, 0.5), (0.0, 0.85), (-0.1, 1.0), (-0.18, 0.9), (-0.22, 0.98), (-0.28, 0.86),
        (-0.32, 0.9), (-0.36, 0.76), (-0.33, 0.5), (-0.22, 0.08)]


@register
class Crows(Vanish):
    name = "cuervos"
    key = "c"
    summary = "te vuelves tinta y te deshaces en una bandada de cuervos entre plumas negras"

    MIX = (0.55, 0.3, 0.15)
    BLOB_SIGMA = 4
    DRAG = 1.0
    MAX_PARTICLES = 1500

    def __init__(self, width, height, **options):
        super().__init__(width, height, **options)
        self.ink = np.full((height, width, 3), INK, np.uint8)
        self.center = (width / 2, height / 2)
        self.scale = height / 720

    def _plan(self, person, direction, origin):
        super()._plan(person, direction, origin)
        cx, cy, *_ = body_box(person, self.w, self.h)
        self.center = (cx, cy)

    def _transition(self, frame, person, bg, now, t, p, vanishing):
        d = self._front(p, vanishing, INK_BAND)
        gone = below(d, 0)
        out = self._background(frame, bg, person, gone, p, vanishing)
        a = cv2.multiply(np.clip(1 - d / INK_BAND, 0, 1) * (1 - gone), person) * np.float32(0.92)
        out = cv2.blendLinear(out, self.ink, 1 - a, a)
        if vanishing:
            edge = cv2.multiply(band(d, 0, EDGE), person)
            self._take_off(edge, 2, 4)
        else:
            self._fly_in(d, person, 3)
        return out

    def _crow_sizes(self, k):
        return self.rng.uniform(26, 48, k) * self.scale

    def _take_off(self, edge, crows, feathers):
        xs, ys = self._sample(edge, crows, stride=3)
        k = len(xs)
        if k:
            cx, cy = self.center
            dx, dy = xs - cx, ys - cy
            dist = np.hypot(dx, dy) + 1
            speed = self.rng.uniform(220, 420, k) * self.scale
            vx = dx / dist * speed + self.rng.normal(0, 60, k)
            vy = dy / dist * speed * 0.5 - self.rng.uniform(120, 260, k) * self.scale
            self._emit(xs, ys, vx, vy, np.full(k, 3.0), self._crow_sizes(k), np.zeros((k, 3)),
                       self.rng.uniform(6, 10, k))
        xs, ys = self._sample(edge, feathers, stride=2)
        k = len(xs)
        if k:
            self._emit(xs, ys, self.rng.normal(0, 50, k), self.rng.uniform(-80, 10, k), self.rng.uniform(1.2, 2.4, k),
                       self.rng.uniform(5, 9, k) * self.scale, np.zeros((k, 3)), np.full(k, FEATHER))

    def _fly_in(self, d, person, n):
        arriving, speed = self._arriving(d, INK_BAND, LEAD)
        xs, ys = self._sample(cv2.multiply(arriving, person), n, stride=3)
        k = len(xs)
        if not k:
            return
        life = np.maximum(-d[ys, xs] / speed, 0.05)
        angle = self.rng.uniform(-math.pi, 0, k)                 # llegan desde arriba y los lados
        dist = self.rng.uniform(300, 700, k) * self.scale
        self._emit_homing(xs + np.cos(angle) * dist, ys + np.sin(angle) * dist, xs, ys, life,
                          self._crow_sizes(k), np.zeros((k, 3)), self.rng.uniform(6, 10, k))

    def _forces(self, p, free, dt):
        crow = free & (p[:, KIND] > 0)
        feather = free & (p[:, KIND] == FEATHER)
        p[crow, VY] -= 60 * dt                                     # suben al alejarse
        # Las plumas caen despacio meciéndose de un lado a otro.
        p[feather, VY] += (55 - p[feather, VY]) * 2 * dt
        p[feather, VX] = np.sin(self.now * 3 + p[feather, Y] / 25) * 70

    def _draw_particles(self, out, p):
        out = out.copy()
        for row in p.tolist():
            if row[KIND] == FEATHER:
                self._feather(out, row)
            else:
                self._crow(out, row)
        return out

    def _feather(self, img, row):
        f = max(0.0, row[LIFE] / row[MAX_LIFE])
        if f < 0.05:
            return
        angle = math.degrees(math.sin(self.now * 3 + row[Y] / 25)) * 0.8 + 90
        s = row[SIZE]
        cv2.ellipse(img, (int(row[X]), int(row[Y])), (max(2, int(s)), max(1, int(s * 0.28))), angle, 0, 360,
                    CROW, -1, cv2.LINE_AA)

    def _crow(self, img, row):
        x, y, s = row[X], row[Y], row[SIZE]
        # Hacia dónde vuela: su velocidad; en los que vuelven, hacia su destino (VX/VY = distancia que falta).
        heading = math.atan2(-row[VY], -row[VX]) if row[TX] >= 0 else math.atan2(row[VY], row[VX])
        age = row[MAX_LIFE] - row[LIFE]
        flap = math.sin(age * row[KIND] * 2 * math.pi + row[MAX_LIFE] * 1000)
        fx, fy = math.cos(heading), math.sin(heading)       # adelante
        sx, sy = -fy, fx                                     # a un lado

        def at(forward, side):
            return int(x + fx * forward * s + sx * side * s), int(y + fy * forward * s + sy * side * s)

        span = 0.35 + 0.65 * (0.5 + 0.5 * flap)            # alas abiertas ↔ recogidas
        for side in (1, -1):
            # Ala ancha de cuervo, con las plumas primarias abiertas como dedos en la punta.
            wing = [at(f, side * w * span) for f, w in WING]
            cv2.fillPoly(img, [np.array(wing, np.int32)], CROW, cv2.LINE_AA)
            cv2.line(img, at(0.12, side * 0.1 * span), at(-0.08, side * 0.95 * span), SHEEN, 1, cv2.LINE_AA)
        angle = math.degrees(heading)
        cv2.ellipse(img, (int(x), int(y)), (max(2, int(s * 0.42)), max(1, int(s * 0.15))), angle, 0, 360, CROW, -1,
                    cv2.LINE_AA)
        cv2.circle(img, at(0.45, 0), max(1, int(s * 0.12)), CROW, -1, cv2.LINE_AA)               # cabeza
        cv2.fillConvexPoly(img, np.array([at(0.55, 0.05), at(0.72, 0), at(0.55, -0.05)], np.int32), CROW)  # pico
        tail = np.array([at(-0.3, 0.06), at(-0.62, 0.17), at(-0.68, 0.06), at(-0.7, 0), at(-0.68, -0.06),
                         at(-0.62, -0.17), at(-0.3, -0.06)], np.int32)
        cv2.fillPoly(img, [tail], CROW, cv2.LINE_AA)                                                # cola en abanico
