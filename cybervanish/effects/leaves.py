"""Hojas: un remolino de hojas te envuelve cada vez más rápido, te tapa y se dispersa con el viento.

Inspirado en el desplazamiento instantáneo de los ninjas del anime, que se van en un torbellino de
hojas. Las posiciones son función del progreso, así que al volver el remolino ocurre al revés.
"""

import math

import cv2
import numpy as np

from .base import Vanish, body_box, register, smoothstep

N_LEAVES = 240
# BGR: verdes de distintos tonos y alguna hoja seca.
PALETTE = np.array([(40, 150, 60), (30, 120, 45), (60, 180, 90), (40, 170, 140), (30, 110, 170), (50, 140, 200)],
                   np.float32)
GATHER = (0.0, 0.35)      # las hojas llegan y el remolino se cierra
HIDE = (0.36, 0.5)        # te desvaneces tapado por las hojas
SCATTER = (0.55, 1.0)     # el viento se las lleva


@register
class Leaves(Vanish):
    name = "hojas"
    key = "0"
    summary = "un remolino de hojas te envuelve, te tapa y se dispersa con el viento"

    def __init__(self, width, height, **options):
        super().__init__(width, height, **options)
        self.box = body_box(np.zeros((height, width), np.float32), width, height)
        self._seed_leaves()

    def _seed_leaves(self):
        n = N_LEAVES
        self.theta0 = self.rng.uniform(0, 2 * math.pi, n)
        self.height = self.rng.uniform(-0.05, 1.05, n)        # dónde del cuerpo giran (0 = arriba)
        self.radius = self.rng.uniform(0.8, 1.3, n)
        self.spin = self.rng.uniform(-9, 9, n)
        self.size = self.rng.uniform(10, 18, n)
        self.color = PALETTE[self.rng.integers(0, len(PALETTE), n)] * self.rng.uniform(0.8, 1.15, (n, 1))
        self.wind = np.stack([self.rng.uniform(0.6, 1.6, n), self.rng.uniform(-1.2, 0.3, n)], axis=1)

    def _plan(self, person, direction, origin):
        self.box = body_box(person, self.w, self.h)
        self._seed_leaves()
        # El viento se las lleva hacia el lado contrario a la mano (o hacia donde se barrió).
        side = direction
        if origin is not None:
            side = -1 if origin[0] * self.w > self.box[0] else 1
        self.wind[:, 0] = np.abs(self.wind[:, 0]) * side

    def _transition(self, frame, person, bg, now, t, p, vanishing):
        q = p if vanishing else 1 - p
        presence = 1 - float(smoothstep(*HIDE, q))
        out = self._background(frame, bg, person, np.full_like(person, 1 - presence), p, vanishing)

        cx, cy, half, top, bottom = self.box
        gather = float(smoothstep(*GATHER, q))
        scatter = float(smoothstep(*SCATTER, q))
        # Gira cada vez más rápido; se cierra sobre el cuerpo y luego se abre al dispersarse.
        theta = self.theta0 + 16 * q + 14 * q * q
        r = half * self.radius * (2.4 - 1.25 * gather) * (1 + 0.4 * scatter)
        x = cx + r * np.cos(theta) + self.wind[:, 0] * self.w * 0.9 * scatter**1.5
        y = (top + self.height * (bottom - top) + np.sin(theta * 1.7) * 12
             + self.wind[:, 1] * self.h * 0.6 * scatter**1.5)
        depth = np.sin(theta)                           # < 0: la hoja pasa por detrás de ti
        visible = np.clip(gather * 3, 0, 1) * (1 - smoothstep(0.85, 1.0, q))

        out = self._wind_arcs(out, q, gather, scatter)
        if visible < 0.02:
            return out
        # Las hojas van llegando poco a poco (y se van al final): solo se dibuja una parte.
        active = np.arange(N_LEAVES) < visible * N_LEAVES
        back, front = active & (depth < 0), active & (depth >= 0)
        behind = self._draw(np.zeros_like(out), x[back], y[back], depth[back], theta[back], q, back)
        mask = cv2.cvtColor(behind, cv2.COLOR_BGR2GRAY) > 0
        # Las de detrás solo se ven donde no tapa tu cuerpo.
        mask &= (person * presence) < 0.5
        out[mask] = behind[mask]
        return self._draw(out, x[front], y[front], depth[front], theta[front], q, front)

    def _draw(self, img, x, y, depth, theta, q, sel):
        size = self.size[sel] * (0.8 + 0.3 * depth)
        color = self.color[sel] * (0.75 + 0.25 * depth)[:, None]
        angle = np.degrees(theta + self.spin[sel] * q * 3)
        for xi, yi, si, ci, ai in zip(x, y, size, color.tolist(), angle, strict=True):
            c = (int(xi), int(yi))
            axes = (max(2, int(si)), max(1, int(si * 0.42)))
            cv2.ellipse(img, c, axes, float(ai), 0, 360, ci, -1, cv2.LINE_AA)
            # Nervio central, más oscuro.
            dx, dy = math.cos(math.radians(ai)) * si * 0.9, math.sin(math.radians(ai)) * si * 0.9
            cv2.line(img, (int(xi - dx), int(yi - dy)), (int(xi + dx), int(yi + dy)),
                     tuple(v * 0.6 for v in ci), 1, cv2.LINE_AA)
        return img

    def _wind_arcs(self, out, q, gather, scatter):
        """Trazos de viento blancos que giran alrededor del cuerpo mientras dura el remolino."""
        strength = gather * (1 - scatter) * min(1.0, q * 4)
        if strength < 0.05:
            return out
        cx, cy, half, top, bottom = self.box
        layer = np.zeros_like(out)
        for i in range(6):
            yc = int(top + (bottom - top) * (0.15 + 0.14 * i))
            start = (q * 2200 + i * 61) % 360
            cv2.ellipse(layer, (int(cx), yc), (int(half * 1.3), int(half * 0.28)), 0, start, start + 110,
                        (255, 255, 255), 2, cv2.LINE_AA)
        return cv2.addWeighted(out, 1, layer, 0.55 * strength, 0)
