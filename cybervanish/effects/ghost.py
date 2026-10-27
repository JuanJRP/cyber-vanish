"""Fantasma: el cuerpo se vuelve translúcido, se separa en colores como una señal espectral, ondula en
jirones que suben y deja ecos que flotan hacia arriba hasta desvanecerse."""

import math

import cv2
import numpy as np

from .base import Vanish, glow, gray3, register, smoothstep

RISE = 3.0       # px por frame que suben los ecos
DECAY = 0.86     # cuánto queda de los ecos de un frame al siguiente
PHASES = (0.0, 1.0, 2.0)   # desfase de la onda en cada canal (B, G, R): separación espectral


@register
class Ghost(Vanish):
    name = "fantasma"
    key = "7"
    summary = "te vuelves un espectro translúcido que ondula y deja ecos flotando"

    def __init__(self, width, height, **options):
        super().__init__(width, height, **options)
        self.xs = np.arange(width, dtype=np.float32).reshape(1, -1)
        self.ys = np.arange(height, dtype=np.float32).reshape(-1, 1)
        # Los jirones ondulan más cuanto más arriba: como humo que se eleva.
        self.lift = 0.6 + 0.8 * (1 - self.ys / height)
        self.echo = np.zeros((height, width, 3), np.uint8)
        self.echo_energy = 0.0
        self.rise = np.float32([[1, 0, 0], [0, 1, -RISE]])

    def _plan(self, person, direction, origin):
        self.echo[:] = 0     # se desvanece todo a la vez; solo se limpian los ecos anteriores
        self.echo_energy = 0.0

    def _transition(self, frame, person, bg, now, t, p, vanishing):
        q = p if vanishing else 1 - p
        amp = 22 * float(smoothstep(0, 0.7, q))
        split = 1.3 * q
        planes, channels, mask = cv2.split(frame), [], None
        for c, phase in enumerate(PHASES):
            ph = np.float32(phase * split)
            map_x, map_y = np.broadcast_arrays(
                self.xs + (amp * self.lift) * np.sin(self.ys / 18 + np.float32(t * 8) + ph),
                self.ys + (0.4 * amp) * np.sin(self.xs / 25 + np.float32(t * 6) + ph),
            )
            map_x, map_y = map_x.astype(np.float32), map_y.astype(np.float32)
            channels.append(cv2.remap(planes[c], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
            if c == 1:
                mask = cv2.remap(person, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        body = cv2.merge(channels)
        # Pierde el color y se vuelve frío y luminoso a medida que se va.
        body = cv2.addWeighted(body, 1 - 0.55 * q, gray3(body), 0.55 * q, 0)
        body = cv2.add(body, (50 * q, 28 * q, 0, 0))

        # Opacidad que baja con un leve parpadeo, como una aparición inestable.
        flicker = 1 - 0.18 * q * (0.5 + 0.5 * math.sin(t * 23)) * self.rng.uniform(0.6, 1.0)
        a = mask * np.float32((1 - smoothstep(0.15, 1.0, q)) * flicker)

        out = self._background(frame, bg, person, np.ones_like(person), p, vanishing)
        out = self._echoes(out)
        out = cv2.blendLinear(out, body, 1 - a, a)
        # Halo etéreo alrededor de la silueta.
        aura = cv2.multiply(body, cv2.merge([a, a, a]), dtype=cv2.CV_8U)
        out = cv2.add(out, glow(aura, 0.8 * math.sin(math.pi * q), 6))
        # El eco de este frame empezará a subir en el siguiente.
        if vanishing:
            self.echo = cv2.max(self.echo, cv2.convertScaleAbs(aura, alpha=0.5 * q))
            self.echo_energy = 1.0
        return out

    def _echoes(self, out):
        """Copias de la silueta que suben y se apagan."""
        if self.echo_energy < 0.03:
            return out
        self.echo = cv2.convertScaleAbs(cv2.warpAffine(self.echo, self.rise, (self.w, self.h)), alpha=DECAY)
        self.echo_energy *= DECAY
        return cv2.add(out, cv2.GaussianBlur(self.echo, (0, 0), 1.5))

    def _vanished(self, frame, person, bg, now):
        # Los últimos ecos siguen subiendo un momento después de desaparecer.
        return self._echoes(super()._vanished(frame, person, bg, now))
