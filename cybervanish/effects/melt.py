"""Derretir: el cuerpo se escurre hacia abajo como cera caliente, en chorretones brillantes que
gotean y forman un charco que acaba evaporándose."""

import math

import cv2
import numpy as np

from .base import COLOR, SIZE, Vanish, X, Y, register, smoothstep


@register
class Melt(Vanish):
    name = "derretir"
    key = "5"
    summary = "te derrites como cera: chorretones brillantes, gotas y un charco"

    GRAVITY = 1400.0
    DRAG = 0.995
    MAX_PARTICLES = 600

    def __init__(self, width, height, **options):
        super().__init__(width, height, **options)
        self.xs = np.arange(width, dtype=np.float32).reshape(1, -1)
        self.ys = np.arange(height, dtype=np.float32).reshape(-1, 1)
        self.speed = np.ones((1, width), np.float32)
        self.cx, self.half = width / 2, width / 6
        self.tint = np.zeros((height, width, 3), np.uint8)

    def _plan(self, person, direction, origin):
        # Velocidad de caída por columna: ondas suaves más algunos chorretones rápidos y estrechos.
        n = self.w // 6
        speed = cv2.GaussianBlur(self.rng.random((1, n)).astype(np.float32), (0, 0), 4)
        speed = (speed - speed.min()) / (np.ptp(speed) + 1e-6)
        drips = cv2.GaussianBlur((self.rng.random((1, n)) < 0.08).astype(np.float32), (0, 0), 1) * 1.5
        self.speed = cv2.resize(0.5 + speed + drips, (self.w, 1), interpolation=cv2.INTER_LINEAR)
        body = person > 0.5
        cols = np.flatnonzero(body.any(axis=0))
        if len(cols):
            self.cx, self.half = (cols[0] + cols[-1]) / 2, max((cols[-1] - cols[0]) / 2, 20)

    def _transition(self, frame, person, bg, now, t, p, vanishing):
        q = p if vanishing else 1 - p
        k = np.float32(q * q)
        # Cada punto baja más cuanto más abajo está: el cuerpo se estira en chorretones.
        dy = k * self.h * self.speed * (0.25 + 0.75 * self.ys / self.h)
        map_x = self.xs + np.sin(self.ys * 0.05 + t * 3) * (6 * k)
        map_x, map_y = (m.astype(np.float32) for m in np.broadcast_arrays(map_x, self.ys - dy))
        body = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        mask = cv2.remap(person, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        fade = np.float32(1 - smoothstep(0.6, 1.0, q))

        # Lo derretido se oscurece y se satura un poco, como cera.
        body = cv2.convertScaleAbs(body, alpha=1 - 0.28 * float(k))
        out = self._background(frame, bg, person, np.ones_like(person), p, vanishing)
        a = mask * fade
        out = cv2.blendLinear(out, body, 1 - a, a)
        out = self._sheen(out, mask, float(k) * float(fade))
        out = self._puddle(out, body, mask, q, t)
        if vanishing and 0.12 < q < 0.9:
            self._drip(mask, body, 6)
        return out

    def _sheen(self, out, mask, k):
        """Brillo de líquido: la parte de arriba de cada chorretón refleja la luz."""
        if k < 0.02:
            return out
        small = cv2.GaussianBlur(cv2.resize(mask, (self.w // 2, self.h // 2), interpolation=cv2.INTER_AREA), (0, 0), 2)
        up = -cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)       # > 0 donde la superficie mira hacia arriba
        side = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
        spec = np.clip(up * 1.4 - np.abs(side) * 0.4, 0, 1) ** 2 * (small > 0.3)
        spec = cv2.resize(spec, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        s = cv2.convertScaleAbs(spec, alpha=230 * k)
        return cv2.add(out, cv2.merge([s, s, s]))

    def _puddle(self, out, body, mask, q, t):
        """Charco que crece debajo del cuerpo y se evapora al final."""
        a = float(smoothstep(0.15, 0.45, q) * (1 - smoothstep(0.75, 1.0, q)))
        if a < 0.01:
            return out
        # Todo lo derretido escurre hasta el borde de abajo: ahí se forma el charco.
        fw, fh = self.w // 4, self.h // 4
        rx = self.half * (0.7 + 0.9 * q) / 4
        ry = (14 + 50 * q) / 4
        cy = fh - 1
        layer = np.zeros((fh, fw), np.float32)
        cv2.ellipse(layer, (int(self.cx / 4), cy), (int(rx), max(1, int(ry))), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        layer = cv2.GaussianBlur(layer, (0, 0), 1.5)
        # Reflejo en el borde de arriba del charco y ondas suaves donde caen las gotas.
        rim = np.clip(layer - np.roll(layer, 2, axis=0), 0, 1)
        yy, xx = np.ogrid[:fh, :fw]
        r = np.hypot((xx - self.cx / 4) / max(rx, 1), (yy - cy) / max(ry, 1))
        ripple = (0.5 + 0.5 * np.sin(r * 7 - t * 6)).astype(np.float32) * layer * 0.1
        m = cv2.resize(layer * np.float32(a * 0.92), (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        hi = cv2.resize((ripple + rim * 0.6) * np.float32(a), (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        # Color del charco: el color medio de lo que se está derritiendo.
        sel = mask[::4, ::4] > 0.5
        color = body[::4, ::4][sel].mean(axis=0) * 0.7 if sel.any() else np.array([60, 60, 60])
        self.tint[:] = color.astype(np.uint8)
        out = cv2.blendLinear(out, self.tint, 1 - m, m)
        h8 = cv2.convertScaleAbs(hi, alpha=255)
        return cv2.add(out, cv2.merge([h8, h8, h8]))

    def _drip(self, mask, body, n):
        """Gotas que se sueltan de las puntas de los chorretones más rápidos."""
        cols = np.flatnonzero(mask.max(axis=0) > 0.5)
        if not len(cols):
            return
        w = self.speed[0, cols] ** 4
        xs = self.rng.choice(cols, min(n, len(cols)), p=w / w.sum())
        hit = mask[:, xs] > 0.5
        ok = hit.any(axis=0)
        xs = xs[ok]
        ys = self.h - 1 - np.argmax(hit[::-1, ok], axis=0)
        ys = np.clip(ys - 2, 0, self.h - 1)
        k = len(xs)
        if not k:
            return
        self._emit(xs, ys, self.rng.normal(0, 8, k), self.rng.uniform(20, 120, k), np.full(k, 1.5),
                   self.rng.uniform(2.5, 4.5, k), body[ys, xs])

    def _draw_particles(self, out, p):
        out = out.copy()
        for row in p.tolist():
            x, y, r = int(row[X]), int(row[Y]), max(1, int(row[SIZE] / 2 + 0.5))
            color = tuple(row[COLOR])
            cv2.ellipse(out, (x, y), (r, r + 1 + int(math.sqrt(r))), 0, 0, 360, color, -1, cv2.LINE_AA)
            cv2.circle(out, (x - r // 2, y - r // 2), max(1, r // 2), (235, 235, 235), -1, cv2.LINE_AA)
        return out
