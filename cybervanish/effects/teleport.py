"""Teletransporte: un escáner te recorre y te llenas de energía, te vuelves luz, te comprimes en una
línea, la línea en un punto, y el punto estalla en un destello. Al volver, todo ocurre al revés."""

import math

import cv2
import numpy as np

from .base import Vanish, glow, register, smoothstep

LIGHT = (255, 236, 205)    # BGR: blanco azulado
SPARK = (255, 215, 140)    # destellos cian
# Fases, en q (0 = estás, 1 = te fuiste).
ENERGIZE = (0.0, 0.45)     # luz que ondula y destellos sobre tu cuerpo real
SOLID = (0.3, 0.5)         # tu cuerpo real se apaga y queda una silueta de luz
SQUEEZE = (0.5, 0.8)       # la silueta se comprime en una línea vertical
COLLAPSE = (0.8, 0.9)      # la línea se encoge a un punto
FLASH = 0.9                # el punto estalla


@register
class Teleport(Vanish):
    name = "teleport"
    key = "4"
    summary = "un escáner te llena de energía, te comprimes en un punto de luz y estallas"

    MAX_PARTICLES = 2500

    def __init__(self, width, height, **options):
        super().__init__(width, height, **options)
        self.light = np.full((height, width, 3), LIGHT, np.uint8)
        self.xs = np.arange(width, dtype=np.float32).reshape(1, -1)
        self.ys = np.arange(height, dtype=np.float32).reshape(-1, 1)
        self.cx, self.cy, self.half = width / 2, height / 2, width / 6
        self.top, self.bottom = 0, height
        self.burst = False

    def _plan(self, person, direction, origin):
        # No hay frente que avance: todo el cuerpo se va a la vez. Solo hace falta dónde está.
        body = person > 0.5
        cols, rows = np.flatnonzero(body.any(axis=0)), np.flatnonzero(body.any(axis=1))
        if len(cols):
            self.cx, self.half = (cols[0] + cols[-1]) / 2, max((cols[-1] - cols[0]) / 2, 20)
            self.top, self.bottom = rows[0], rows[-1]
            self.cy = (self.top + self.bottom) / 2
        self.burst = False

    def _transition(self, frame, person, bg, now, t, p, vanishing):
        q = p if vanishing else 1 - p
        presence = 1 - smoothstep(*SOLID, q)
        out = self._background(frame, bg, person, np.full_like(person, 1 - presence), p, vanishing)

        # 1) Energía sobre tu cuerpo real: columnas de luz que ondulan y un escáner que sube y baja.
        energy = float(math.sin(math.pi * smoothstep(*ENERGIZE, q)))
        shine = float(smoothstep(0.05, SOLID[1], q))
        cols = 0.6 + 0.4 * np.cos(self.xs * 0.13 + np.float32(t * 10))
        scan_y = np.float32(self.top + (self.bottom - self.top) * (0.5 + 0.5 * math.sin(t * 4.5)))
        scan = np.exp(-(((self.ys - scan_y) / 16) ** 2))
        a = np.clip(person * (cols * np.float32(shine * 0.9) + scan * np.float32(0.7 * energy)), 0, 1)
        a *= np.float32(presence)
        out = cv2.blendLinear(out, self.light, 1 - a, a)

        # 2) Silueta de luz que se comprime en una línea y la línea en un punto.
        sx = 1 - 0.985 * float(smoothstep(*SQUEEZE, q))
        sy = 1 - 0.99 * float(smoothstep(*COLLAPSE, q))
        lum = float(smoothstep(SOLID[0], SOLID[1], q) * (1 - smoothstep(FLASH - 0.03, FLASH, q)))
        if lum > 0.01:
            lit = cv2.convertScaleAbs(person * (0.85 + 0.15 * cols), alpha=255 * lum * min(3.0, 1 / math.sqrt(sx * sy)))
            layer = cv2.merge([lit, cv2.convertScaleAbs(lit, alpha=0.93), cv2.convertScaleAbs(lit, alpha=0.8)])
            m = np.float32([[sx, 0, self.cx * (1 - sx)], [0, sy, self.cy * (1 - sy)]])
            layer = cv2.warpAffine(layer, m, (self.w, self.h), flags=cv2.INTER_LINEAR)
            out = cv2.add(cv2.add(out, layer), glow(layer, 2.2, 3))

        out = self._beam(out, q, sx)
        out = self._flash(out, q)
        self._sparkle(person, q, sx, sy, int(240 * energy + 120 * (sx < 0.95) * lum), vanishing)
        return out

    def _beam(self, out, q, sx):
        """Haz vertical sobre el cuerpo que se estrecha con la silueta."""
        strength = 0.55 * math.sin(math.pi * min(1.0, q / FLASH))
        if strength < 0.01:
            return out
        width = self.half * sx * 0.9 + 3
        profile = np.exp(-(((self.xs[0] - self.cx) / width) ** 2)) * strength
        row = (profile[:, None] * np.array(LIGHT, np.float32)).astype(np.uint8)[None]
        return cv2.add(out, cv2.resize(row, (self.w, self.h), interpolation=cv2.INTER_NEAREST))

    def _flash(self, out, q):
        """Destello al colapsar: núcleo, estela horizontal anamórfica y estela vertical."""
        f = float(smoothstep(FLASH - 0.05, FLASH, q) * (1 - smoothstep(FLASH, 1.0, q)))
        if f < 0.01:
            return out
        fw, fh = self.w // 4, self.h // 4
        x = (np.arange(fw, dtype=np.float32) - self.cx / 4).reshape(1, -1)
        y = (np.arange(fh, dtype=np.float32) - self.cy / 4).reshape(-1, 1)
        core = np.exp(-(x**2 + y**2) / (2 * (3 + 10 * f) ** 2))
        streak_h = np.exp(-((x / (fw * 0.35)) ** 2)) * np.exp(-((y / 1.2) ** 2))
        streak_v = np.exp(-((x / 1.0) ** 2)) * np.exp(-((y / (fh * 0.25)) ** 2)) * 0.6
        light = np.clip((core + streak_h + streak_v) * f * 1.4, 0, 1)
        lit = cv2.convertScaleAbs(light, alpha=255)
        layer = cv2.resize(cv2.merge([lit, cv2.convertScaleAbs(lit, alpha=0.9), cv2.convertScaleAbs(lit, alpha=0.75)]),
                           (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        out = cv2.add(out, (70 * f, 65 * f, 55 * f, 0))   # toda la imagen se ilumina un instante
        return cv2.add(cv2.add(out, layer), glow(layer, 2.0, 4))

    def _sparkle(self, person, q, sx, sy, n, vanishing):
        xs, ys = self._sample(person, n, stride=2)
        k = len(xs)
        if k:
            # Siguen a la silueta mientras se comprime.
            xs = self.cx + (xs - self.cx) * sx
            ys = self.cy + (ys - self.cy) * sy
            colors = np.where(self.rng.random((k, 1)) < 0.65, [255, 255, 255], SPARK)
            self._emit(xs, ys, self.rng.normal(0, 12, k), self.rng.uniform(-90, -10, k),
                       self.rng.uniform(0.15, 0.45, k), np.full(k, 2), colors)
        # El estallido: al irte, chispas que salen del punto; al volver, chispas que caen hacia él.
        if not self.burst and (q >= FLASH or not vanishing):
            self.burst = True
            k = 400
            angle = self.rng.uniform(0, 2 * math.pi, k)
            dist = self.rng.uniform(80, 520, k) * np.where(self.rng.random(k) < 0.5, 1, 0.4)
            colors = np.where(self.rng.random((k, 1)) < 0.5, [255, 255, 255], SPARK)
            sizes = self.rng.choice([1, 2, 2, 3], k)
            if vanishing:
                life = self.rng.uniform(0.4, 0.9, k)
                self._emit(np.full(k, self.cx), np.full(k, self.cy), np.cos(angle) * dist / life,
                           np.sin(angle) * dist / life * 0.6, life, sizes, colors)
            else:
                # Aterrizan en el punto justo cuando estalla (q = FLASH, al principio de la aparición).
                life = np.full(k, (1 - FLASH) * self.duration)
                self._emit_homing(self.cx + np.cos(angle) * dist, self.cy + np.sin(angle) * dist * 0.6,
                                  np.full(k, self.cx), np.full(k, self.cy), life, sizes, colors)

    def _draw_particles(self, out, p):
        layer = self._splat(np.zeros_like(out), p)
        return cv2.add(cv2.add(out, layer), glow(layer, 3.0, 2))
