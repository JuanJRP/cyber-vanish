"""Matrix: el cuerpo se convierte en lluvia de código verde que se apaga."""

import cv2
import numpy as np

from .base import Vanish, below, register

CELL_W, CELL_H = 14, 18
GLYPHS = "0123456789ABCDEFXZ:.=*+-<>|"
TINT = 0.08    # banda por delante del frente donde el cuerpo se tiñe de verde
CODE = 0.18    # banda por detrás del frente donde sigue el código antes de apagarse


@register
class Matrix(Vanish):
    name = "matrix"
    key = "8"
    summary = "te conviertes en lluvia de código verde que se apaga"

    MIX = (0.6, 0.25, 0.15)

    def __init__(self, width, height, **options):
        super().__init__(width, height, **options)
        self.rain = self._make_rain()
        self.speeds = self.rng.uniform(120, 420, -(-width // CELL_W))
        self.code = np.zeros((height, width, 3), np.uint8)

    def _make_rain(self):
        """Textura de código de dos alturas de pantalla (para que el bucle de caída no tenga costura)."""
        tex = np.zeros((self.h, self.w, 3), np.uint8)
        for x in range(0, self.w, CELL_W):
            for y in range(CELL_H, self.h + CELL_H, CELL_H):
                v = int(self.rng.integers(90, 256))
                color = (v // 3, v, v // 4) if self.rng.random() > 0.05 else (220, 255, 220)
                cv2.putText(tex, GLYPHS[int(self.rng.integers(len(GLYPHS)))], (x + 1, y - 3),
                            cv2.FONT_HERSHEY_PLAIN, 1.1, color, 2, cv2.LINE_AA)
        return np.concatenate([tex, tex])

    def _code_frame(self, t):
        for i, speed in enumerate(self.speeds):
            x = i * CELL_W
            start = (self.h - int(t * speed) % self.h) % self.h
            self.code[:, x : x + CELL_W] = self.rain[start : start + self.h, x : x + CELL_W]
        return self.code

    def _transition(self, frame, person, bg, now, t, p, vanishing):
        d = self._front(p, vanishing, CODE)
        gone = below(d, 0)
        out = self._background(frame, bg, person, gone, p, vanishing)

        # Por delante del frente: tinte verde.
        green = cv2.transform(frame, np.array([[0.2, 0.3, 0.1], [0.3, 0.9, 0.3], [0.1, 0.2, 0.15]], np.float32))
        a = cv2.multiply(np.clip(1 - d / TINT, 0, 1) * (1 - gone), person)
        out = cv2.blendLinear(out, green, 1 - a, a)

        # Por detrás: tu silueta en verde oscuro con el código cayendo encima, y se va apagando.
        code = cv2.add(cv2.convertScaleAbs(green, alpha=0.35), self._code_frame(t))
        c = cv2.multiply(np.clip(1 + d / CODE, 0, 1) * gone, person)
        return cv2.blendLinear(out, code, 1 - c, c)
