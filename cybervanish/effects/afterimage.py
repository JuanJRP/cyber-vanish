"""Velocidad: parpadeas un instante y sales disparado de lado, estirado y con estela, dejando atrás
una imagen residual que parpadea y se apaga, entre líneas de velocidad de manga.

Inspirado en las imágenes residuales de los luchadores del anime que se mueven tan rápido que su
copia se queda quieta un momento. Al volver, llegas a toda velocidad y frenas en seco.
"""

import math

import cv2
import numpy as np

from .base import Vanish, body_box, gray3, register, smoothstep

FLICKER = 0.25            # hasta aquí (en progreso) parpadeas, a punto de salir
ZIP = (0.25, 0.42)        # sales disparado
GHOST = (0.3, 0.95)       # la imagen residual se apaga


@register
class Afterimage(Vanish):
    name = "velocidad"
    key = "v"
    summary = "sales disparado a toda velocidad dejando una imagen residual que parpadea"

    def __init__(self, width, height, **options):
        super().__init__(width, height, **options)
        self.side = 1
        self.box = body_box(np.zeros((height, width), np.float32), width, height)
        self.ghost = None
        self.lines = np.zeros((0, 4), np.float32)

    def _plan(self, person, direction, origin):
        self.box = body_box(person, self.w, self.h)
        self.side = direction
        if origin is not None:   # sales hacia el lado contrario a la mano que chasqueó
            self.side = -1 if origin[0] * self.w > self.box[0] else 1
        self.ghost = None
        # Líneas de velocidad: altura, largo, grosor y desfase de cada una.
        n = 46
        self.lines = np.stack([self.rng.uniform(0, self.h, n), self.rng.uniform(0.15, 0.6, n) * self.w,
                               self.rng.integers(1, 4, n), self.rng.uniform(0, 1, n)], axis=1)

    def _transition(self, frame, person, bg, now, t, p, vanishing):
        q = p if vanishing else 1 - p
        z = float(smoothstep(*ZIP, q))
        out = self._background(frame, bg, person, np.ones_like(person), p, vanishing)

        # La imagen residual: una foto fija de justo antes de salir, que parpadea y se apaga.
        if vanishing and self.ghost is None and q >= ZIP[0]:
            # Algo desteñida y azulada, para que se lea como una copia y no como tú.
            tinted = cv2.add(cv2.addWeighted(frame, 0.55, gray3(frame), 0.45, 0), (55, 22, 0, 0))
            self.ghost = (tinted, person.copy())
        if vanishing and self.ghost is not None:
            strobe = 1.0 if int(t * 24) % 3 else 0.2
            fade = 1 - float(smoothstep(*GHOST, q))
            # Mientras sales disparado, más copias en estela entre tu sitio y donde vas.
            for k in (2, 1):
                if 0 < z < 1:
                    m = np.float32([[1, 0, self.side * self.w * 0.09 * k * min(1.0, z * 3)], [0, 1, 0]])
                    img = cv2.warpAffine(self.ghost[0], m, (self.w, self.h), borderMode=cv2.BORDER_REPLICATE)
                    a = cv2.warpAffine(self.ghost[1], m, (self.w, self.h)) * np.float32(0.4 / k * (1 - z))
                    out = cv2.blendLinear(out, img, 1 - a, a)
            a = self.ghost[1] * np.float32(0.85 * strobe * fade)
            out = cv2.blendLinear(out, self.ghost[0], 1 - a, a)

        out = self._dash(out, frame, person, q, z, t)
        return self._speed_lines(out, z)

    def _dash(self, out, frame, person, q, z, t):
        """Tú: parpadeo y luego salida disparada, estirado en la dirección del movimiento y con estela."""
        if z >= 1:
            return out
        cx = self.box[0]
        shift = self.side * self.w * 1.3 * z * z
        stretch = 1 + 0.7 * math.sin(math.pi * z)
        m = np.float32([[stretch, 0, cx * (1 - stretch) + shift], [0, 1, 0]])
        body = cv2.warpAffine(frame, m, (self.w, self.h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mask = cv2.warpAffine(person, m, (self.w, self.h), flags=cv2.INTER_LINEAR)
        blur = int(1 + 140 * math.sin(math.pi * z))
        if blur > 2:
            body = cv2.blur(body, (blur, 1))
            mask = cv2.blur(mask, (blur, 1))
        # Antes de salir, parpadeas como una señal inestable.
        strobe = 1.0 if q >= FLICKER or int(t * 30) % 2 else 0.55
        a = mask * np.float32(strobe)
        return cv2.blendLinear(out, body, 1 - a, a)

    def _speed_lines(self, out, z):
        """Rayas horizontales que cruzan la pantalla en la dirección de la salida."""
        strength = math.sin(math.pi * z)
        if strength < 0.05:
            return out
        layer = out.copy()
        for y, length, thick, phase in self.lines.tolist():
            x0 = ((phase + z * 2.5) % 1.4 - 0.2) * self.w
            x0 = x0 if self.side > 0 else self.w - x0
            x1 = x0 - self.side * length
            cv2.line(layer, (int(x0), int(y)), (int(x1), int(y)), (255, 255, 255), int(thick), cv2.LINE_AA)
        return cv2.addWeighted(out, 1 - 0.7 * strength, layer, 0.7 * strength, 0)
