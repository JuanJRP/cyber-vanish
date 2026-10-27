"""Remolino: un vórtice espacio-temporal se abre en tu cara, te retuerce y te absorbe hacia un punto,
y luego se cierra. Al volver, el vórtice te expulsa de vuelta.

Inspirado en las técnicas de teletransporte por vórtice del anime ninja.
"""

import math

import cv2
import numpy as np

from .base import Vanish, body_box, register, smoothstep

TWIST = 9.0        # vueltas (en radianes) que da el centro del remolino al final
ARMS = 3           # brazos de la espiral


@register
class Vortex(Vanish):
    name = "remolino"
    key = "r"
    summary = "un vórtice se abre en tu cara, te retuerce y te absorbe hacia un punto"

    def __init__(self, width, height, **options):
        super().__init__(width, height, **options)
        self._polar(width / 2, height / 3, height / 2)

    def _plan(self, person, direction, origin):
        cx, cy, half, top, bottom = body_box(person, self.w, self.h)
        # Centrado en la cara, más o menos: arriba del cuerpo. Abarca cabeza y torso, no toda la habitación.
        self._polar(cx, top + 0.22 * (bottom - top), max(half * 1.1, (bottom - top) * 0.45))

    def _polar(self, cx, cy, radius):
        """Coordenadas polares respecto al centro del remolino, a media resolución (en px de la imagen
        completa): las deformaciones son suaves, así que los mapas se calculan aquí y se amplían."""
        self.c = (np.float32(cx), np.float32(cy))
        self.R = float(radius)
        hw, hh = self.w // 2, self.h // 2
        gx, gy = np.meshgrid(np.arange(hw, dtype=np.float32) * 2 + 0.5 - np.float32(cx),
                             np.arange(hh, dtype=np.float32) * 2 + 0.5 - np.float32(cy))
        self.r2, self.phi2 = cv2.cartToPolar(gx, gy)
        self.fall2 = np.exp(-((self.r2 / np.float32(self.R)) ** 2))   # el remolino manda cerca del centro

    def _warp(self, img, radius, angle, border):
        mx, my = cv2.polarToCart(radius, angle)
        mx = cv2.resize(mx + self.c[0], (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        my = cv2.resize(my + self.c[1], (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=border)

    def _transition(self, frame, person, bg, now, t, p, vanishing):
        q = p if vanishing else 1 - p
        env = math.sin(math.pi * min(1.0, q / 0.95))
        twist = TWIST * float(smoothstep(0.05, 0.9, q))
        shrink = max(1 - float(smoothstep(0.2, 0.92, q)), 0.015)

        # Tú: cada punto muestra lo que estaba más lejos del centro y menos girado → te encoges girando.
        angle = self.phi2 - np.float32(twist) * (0.35 + self.fall2)
        radius = self.r2 / np.float32(shrink)
        body = self._warp(frame, radius, angle, cv2.BORDER_CONSTANT)
        mask = self._warp(person, radius, angle, cv2.BORDER_CONSTANT)
        mask *= np.float32(1 - smoothstep(0.85, 0.97, q))

        # La habitación también se retuerce un poco, y vuelve a su sitio cuando se cierra el remolino.
        base = self._background(frame, bg, person, np.ones_like(person), p, vanishing)
        if env > 0.01:
            base = self._warp(base, self.r2 * (1 - np.float32(0.12 * env) * self.fall2),
                              self.phi2 - np.float32(1.4 * env) * self.fall2, cv2.BORDER_REPLICATE)
        out = cv2.blendLinear(base, body, 1 - mask, mask)
        return self._darken(out, env, t)

    def _darken(self, out, env, t):
        """Brazos oscuros de la espiral y un núcleo negro que se abre y se cierra."""
        if env < 0.01:
            return out
        arms = 0.5 + 0.5 * np.sin(ARMS * self.phi2 + 5 * np.log1p(self.r2 / 25) - np.float32(t * 9))
        core = np.exp(-((self.r2 / (self.R * (0.06 + 0.16 * env))) ** 2))
        dark = np.clip(self.fall2 * env * (0.25 + 0.45 * arms) + core * env, 0, 0.96)
        keep = cv2.resize(1 - dark, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        out = cv2.multiply(out, cv2.merge([keep, keep, keep]), dtype=cv2.CV_8U)
        # Un borde azulado muy leve, como luz que se tuerce.
        rim = cv2.resize(self.fall2 * (1 - self.fall2) * 4 * env * arms, (self.w, self.h))
        r8 = cv2.convertScaleAbs(rim, alpha=60)
        return cv2.add(out, cv2.merge([r8, cv2.convertScaleAbs(rim, alpha=25), cv2.convertScaleAbs(rim, alpha=10)]))
