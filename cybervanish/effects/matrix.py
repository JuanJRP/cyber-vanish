"""Matrix: el cuerpo se digitaliza en verde y se convierte en lluvia de código —columnas con cabeza
blanca y estela que se apaga, glifos que cambian— que dibuja tu cara antes de apagarse."""

import cv2
import numpy as np

from .base import KIND, LIFE, MAX_LIFE, Vanish, X, Y, below, glow, register

CELL_W, CELL_H = 14, 18
GLYPHS = "0123456789ABCDEFXZ:.=*+-<>|"
TINT = 0.08    # banda por delante del frente donde el cuerpo se digitaliza en verde
CODE = 0.2     # banda por detrás del frente donde sigue el código antes de apagarse
MUTATE = 0.04  # parte de los glifos que cambian en cada frame
GREEN = np.array([[0.15, 0.35, 0.1], [0.25, 0.95, 0.3], [0.05, 0.25, 0.1]], np.float32)


def _atlas():
    """Glifos en blanco sobre negro, normales y en espejo (como en las pantallas de la película)."""
    tiles = []
    for ch in GLYPHS:
        tile = np.zeros((CELL_H, CELL_W), np.uint8)
        cv2.putText(tile, ch, (1, CELL_H - 4), cv2.FONT_HERSHEY_PLAIN, 1.1, 255, 2, cv2.LINE_AA)
        tiles += [tile, np.fliplr(tile)]
    return np.stack(tiles)


@register
class Matrix(Vanish):
    name = "matrix"
    key = "8"
    summary = "te digitalizas y te conviertes en lluvia de código verde que dibuja tu cara"

    MAX_PARTICLES = 400
    GRAVITY = 380.0
    DRAG = 1.0

    def __init__(self, width, height, **options):
        super().__init__(width, height, **options)
        self.atlas = _atlas().astype(np.float32) / 255
        self.cw, self.ch = -(-width // CELL_W), -(-height // CELL_H)
        self.glyphs = self.rng.integers(0, len(self.atlas), (self.ch, self.cw))
        self.heads = self.rng.uniform(-self.ch, self.ch, self.cw)
        self.speeds = self.rng.uniform(8, 26, self.cw)        # celdas por segundo
        self.trails = self.rng.uniform(6, 22, self.cw)        # largo de la estela, en celdas
        self.rows = np.arange(self.ch).reshape(-1, 1)
        self.last_t = 0.0

    def _plan(self, person, direction, origin):
        # Como la lluvia cae, el frente baja: recorrido + de arriba abajo + un desfase por columna de código.
        gw, gh = -(-self.w // self.GRID), -(-self.h // self.GRID)
        pos, drift = self._travel(person, gw, gh, direction, origin)
        down = np.linspace(0, 1, gh, dtype=np.float32).reshape(-1, 1)
        cols = self.rng.random(self.cw)[(np.arange(gw) * self.GRID) // CELL_W].reshape(1, -1)
        order = 0.45 * pos + 0.35 * down + 0.2 * cols
        order = (order - order.min()) / (np.ptp(order) + 1e-6)
        self.order = cv2.resize(order.astype(np.float32), (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        self.drift = cv2.resize(drift.astype(np.float32), (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        self.last_t = 0.0

    def _rain(self, dt, luma):
        """Un frame de lluvia de código. luma: brillo de la imagen por celda (0..1), para dibujar tu cara."""
        mutate = self.rng.random((self.ch, self.cw)) < MUTATE
        self.glyphs[mutate] = self.rng.integers(0, len(self.atlas), mutate.sum())
        self.heads += self.speeds * dt
        done = self.heads - self.trails > self.ch
        self.heads[done] = -self.rng.uniform(0, 8, done.sum())
        self.speeds[done] = self.rng.uniform(8, 26, done.sum())

        dist = self.heads - self.rows                         # celdas por encima de la cabeza de su columna
        trail = np.clip(1 - dist / self.trails, 0, 1) * (dist >= 0)
        trail = trail**1.5 * (0.3 + 0.9 * luma)
        head = ((dist >= 0) & (dist < 1)).astype(np.float32)

        # Todos los glifos de una vez: (filas, columnas, alto, ancho) → imagen.
        img = self.atlas[self.glyphs] * np.maximum(trail, head).astype(np.float32)[..., None, None]
        g = img.transpose(0, 2, 1, 3).reshape(self.ch * CELL_H, self.cw * CELL_W)[: self.h, : self.w]
        heads = cv2.resize(head, (self.cw * CELL_W, self.ch * CELL_H), interpolation=cv2.INTER_NEAREST)
        g8 = cv2.convertScaleAbs(g, alpha=255)
        w8 = cv2.convertScaleAbs(cv2.multiply(g, heads[: self.h, : self.w]), alpha=255)
        return cv2.merge([cv2.add(cv2.convertScaleAbs(g, alpha=70), w8), cv2.add(g8, w8),
                          cv2.add(cv2.convertScaleAbs(g, alpha=40), w8)])

    def _transition(self, frame, person, bg, now, t, p, vanishing):
        dt, self.last_t = t - self.last_t, t
        d = self._front(p, vanishing, CODE)
        gone = below(d, 0)
        out = self._background(frame, bg, person, gone, p, vanishing)

        # Por delante del frente: se digitaliza en verde, con líneas de barrido.
        green = cv2.transform(frame, GREEN)
        green[::3] = cv2.convertScaleAbs(green[::3], alpha=0.6)
        a = cv2.multiply(np.clip(1 - d / TINT, 0, 1) * (1 - gone), person)
        out = cv2.blendLinear(out, green, 1 - a, a)

        # Por detrás: tu silueta en verde oscuro con la lluvia de código encima, y se va apagando.
        luma = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (self.cw, self.ch),
                          interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        code = cv2.add(cv2.convertScaleAbs(green, alpha=0.22), self._rain(max(dt, 0.0), luma))
        c = cv2.multiply(np.clip(1 + d / CODE, 0, 1) * gone, person)
        lit = cv2.multiply(code, cv2.merge([c, c, c]), dtype=cv2.CV_8U)
        out = cv2.blendLinear(out, code, 1 - c, c)
        out = cv2.add(out, glow(lit, 0.9, 2))

        if vanishing:
            self._shed(cv2.multiply(below(d, 0) - below(d, -0.03), person), 14)
        return out

    def _shed(self, edge, n):
        """Glifos sueltos que se desprenden del borde y caen."""
        xs, ys = self._sample(edge, n, stride=2)
        k = len(xs)
        if not k:
            return
        colors = np.tile([90, 255, 120], (k, 1))
        glyph = self.rng.integers(0, len(GLYPHS), k)
        self._emit(xs, ys, self.rng.normal(0, 10, k), self.rng.uniform(20, 120, k), self.rng.uniform(0.6, 1.4, k),
                   np.ones(k), colors, glyph)

    def _draw_particles(self, out, p):
        layer = np.zeros_like(out)
        for row in p.tolist():
            f = row[LIFE] / row[MAX_LIFE]
            cv2.putText(layer, GLYPHS[int(row[KIND])], (int(row[X]), int(row[Y])), cv2.FONT_HERSHEY_PLAIN, 1.1,
                        (90 * f, 255 * f, 120 * f), 1, cv2.LINE_AA)
        return cv2.add(cv2.add(out, layer), glow(layer, 1.2, 2))
