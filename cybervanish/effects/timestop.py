"""Tiempo: el tiempo se detiene. Una esfera de colores invertidos se expande desde ti y la imagen se
congela; mientras todo está parado desapareces, y cuando el tiempo vuelve a correr ya no estás.

Inspirado en las escenas de tiempo detenido del anime de los noventa (JoJo). Al volver ocurre lo
mismo: el tiempo se para, y al reanudarse ya estás ahí.
"""

import cv2
import numpy as np

from .base import Vanish, body_box, gray3, register, smoothstep

EXPAND = (0.0, 0.22)      # la esfera crece hasta cubrirlo todo
SWAP = 0.5                # con el tiempo parado, desapareces (o apareces)
CONTRACT = (0.7, 0.93)    # la esfera vuelve hacia ti y el tiempo se reanuda
EDGE = 10                 # px de borde suave de la esfera
TINT = (40, 0, 30)        # BGR: el mundo detenido tira a morado


@register
class TimeStop(Vanish):
    name = "tiempo"
    key = "t"
    summary = "el tiempo se detiene en una esfera de colores invertidos y, al volver a correr, ya no estás"

    def __init__(self, width, height, **options):
        super().__init__(width, height, **options)
        self._center(width / 2, height / 2)
        self.frozen = None

    def _plan(self, person, direction, origin):
        if origin is not None:
            self._center(origin[0] * self.w, origin[1] * self.h)
        else:
            cx, cy, *_ = body_box(person, self.w, self.h)
            self._center(cx, cy)
        self.frozen = None

    def _center(self, cx, cy):
        gx, gy = np.meshgrid(np.arange(self.w, dtype=np.float32) - cx, np.arange(self.h, dtype=np.float32) - cy)
        self.dist = cv2.magnitude(gx, gy)
        self.reach = float(self.dist.max()) + EDGE

    def _transition(self, frame, person, bg, now, t, p, vanishing):
        present = p < SWAP if vanishing else p >= SWAP
        if p >= CONTRACT[1]:
            # El tiempo vuelve a correr: imagen en vivo, ya sin ti (o contigo).
            gone = np.full_like(person, 0.0 if present else 1.0)
            return self._background(frame, bg, person, gone, 1.0, vanishing)

        # El tiempo se paró en el primer frame: la imagen queda congelada (con grano, para que respire).
        if self.frozen is None:
            self.frozen = (frame.copy(), np.clip(person * 2, 0, 1))
        still, still_person = self.frozen
        scene = self._with_grain(still)
        if not present:
            scene = cv2.blendLinear(scene, self._with_grain(bg), 1 - still_person, still_person)

        # La esfera: crece desde ti, se queda cubriéndolo todo y vuelve a encogerse hacia ti.
        if p < EXPAND[1]:
            radius = self.reach * float(smoothstep(*EXPAND, p))
        else:
            radius = self.reach * (1 - float(smoothstep(*CONTRACT, p)))
        inside = np.clip((np.float32(radius) - self.dist) / EDGE, 0, 1)
        stopped = cv2.add(cv2.addWeighted(cv2.bitwise_not(scene), 0.85, gray3(scene), 0.15, 0), (*TINT, 0))
        out = cv2.blendLinear(scene, stopped, 1 - inside, inside)

        # El frente de la esfera: un anillo de luz.
        ring = np.exp(-(((self.dist - np.float32(radius)) / 7) ** 2))
        r8 = cv2.convertScaleAbs(ring, alpha=230 if 0 < radius < self.reach - EDGE else 0)
        return cv2.add(out, cv2.merge([r8, r8, r8]))
