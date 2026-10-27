"""Fuego: un frente de brasa avanza por el cuerpo con llamas que lamen hacia arriba, calor que
deforma el aire, grietas incandescentes, chispas y humo."""

import cv2
import numpy as np

from .base import COLOR, KIND, LIFE, MAX_LIFE, SIZE, VX, VY, Vanish, Y, band, below, glow, register

EDGE = 0.03     # anchura de la brasa (en progreso 0..1)
CHAR = 0.13     # banda por delante de la brasa que se chamusca
FLAME_H = 36    # alto máximo de las llamas, en px a 1/4 de resolución
SOOT = (12, 20, 34)
SMOKE = (70, 72, 76)
SPARK, PUFF = 0, 1   # tipos de partícula (columna KIND)


def _fire_palette():
    """Negro → rojo oscuro → naranja → amarillo → blanco cálido (BGR)."""
    stops = [(0, (0, 0, 0)), (60, (8, 14, 120)), (120, (10, 70, 230)), (180, (30, 165, 255)),
             (225, (110, 230, 255)), (255, (215, 250, 255))]
    x = np.arange(256)
    lut = [np.interp(x, [s[0] for s in stops], [s[1][c] for s in stops]) for c in range(3)]
    return np.stack(lut, axis=-1).astype(np.uint8).reshape(256, 1, 3)


@register
class Fire(Vanish):
    name = "fuego"
    key = "3"
    summary = "un frente de brasa con llamas te consume dejando chispas y humo"

    MIX = (0.35, 0.5, 0.15)
    BLOB_SIGMA = 5
    MAX_PARTICLES = 2500

    def __init__(self, width, height, **options):
        super().__init__(width, height, **options)
        self.fw, self.fh = width // 4, height // 4
        self.soot = np.full((height, width, 3), SOOT, np.uint8)
        self.smoke = np.full((height, width, 3), SMOKE, np.uint8)
        self.lut = _fire_palette()
        # Las llamas: cada punto de brasa ilumina lo que tiene encima, cada vez menos.
        self.kernel = (np.linspace(1, 0, FLAME_H, dtype=np.float32) ** 1.5).reshape(-1, 1)
        # Ruido alargado en vertical: al subir, rompe las llamas en lenguas.
        self.tongues = self._streaks(sx=1.6, sy=5)
        self.sway = self._streaks(sx=6, sy=10)
        self.gx4, self.gy4 = (a.astype(np.float32) for a in np.meshgrid(np.arange(self.fw), np.arange(self.fh)))
        self.gx, self.gy = (a.astype(np.float32) for a in np.meshgrid(np.arange(width), np.arange(height)))
        self.cracks = np.zeros((height, width), np.float32)

    def _streaks(self, sx, sy):
        noise = cv2.GaussianBlur(self.rng.random((self.fh, self.fw)).astype(np.float32), (0, 0), sx, sigmaY=sy)
        return cv2.normalize(noise, None, 0, 1, cv2.NORM_MINMAX)

    def _plan(self, person, direction, origin):
        super()._plan(person, direction, origin)
        # Grietas: crestas finas de un ruido suave, que brillarán en la zona chamuscada.
        n = cv2.GaussianBlur(self.rng.random((self.fh, self.fw)).astype(np.float32), (0, 0), 2.5)
        n = cv2.normalize(n, None, 0, 1, cv2.NORM_MINMAX)
        ridges = np.clip((1 - np.abs(2 * n - 1)) * 12 - 10.5, 0, 1)
        self.cracks = cv2.resize(ridges, (self.w, self.h), interpolation=cv2.INTER_LINEAR)

    def _transition(self, frame, person, bg, now, t, p, vanishing):
        d = self._front(p, vanishing, CHAR)
        gone = below(d, 0)
        out = self._background(frame, bg, person, gone, p, vanishing)

        # Chamuscado: oscuro y marrón justo detrás de la brasa, normal más lejos; con grietas al rojo.
        heat = cv2.multiply(np.clip(1 - (d - EDGE) / (CHAR - EDGE), 0, 1) * (1 - gone), person)
        c = heat * np.float32(0.88)
        out = cv2.blendLinear(out, self.soot, 1 - c, c)
        g = cv2.convertScaleAbs(cv2.multiply(self.cracks, heat), alpha=255)
        out = cv2.add(out, cv2.merge([cv2.convertScaleAbs(g, alpha=0.05), cv2.convertScaleAbs(g, alpha=0.4), g]))

        edge = cv2.multiply(band(d, 0, EDGE), person)
        flames, intensity = self._flames(edge, t)
        out = self._heat_haze(out, intensity, t)
        out = cv2.add(cv2.add(out, flames), glow(flames, 1.3, 4))

        self._spawn(edge, 34 if vanishing else 18, 3)
        return out

    def _flames(self, edge, t):
        """Llamas a 1/4 de resolución: la brasa se estira hacia arriba, se rompe en lenguas y ondea."""
        src = cv2.resize(edge, (self.fw, self.fh), interpolation=cv2.INTER_AREA)
        src *= self.rng.uniform(0.75, 1.15, (self.fh, self.fw)).astype(np.float32)
        rise = cv2.filter2D(src, -1, self.kernel, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
        n = np.roll(self.tongues, -int(t * 70) % self.fh, axis=0)
        flame = np.maximum(rise * (0.2 + 1.4 * n) * 0.55 - 0.12, 0) + src * 0.9
        sway = (np.roll(self.sway, -int(t * 45) % self.fh, axis=0) - 0.5) * 7
        flame = cv2.remap(flame, self.gx4 + sway, self.gy4 + 1.0, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        intensity = np.clip(flame, 0, 1)
        idx = cv2.convertScaleAbs(intensity, alpha=255)
        colored = cv2.LUT(cv2.merge([idx, idx, idx]), self.lut)
        return cv2.resize(colored, (self.w, self.h), interpolation=cv2.INTER_LINEAR), intensity

    def _heat_haze(self, out, intensity, t):
        """El aire caliente encima del fuego tiembla."""
        hot = cv2.GaussianBlur(intensity, (0, 0), 5)
        if hot.max() < 0.02:
            return out
        wobble = np.roll(self.sway, -int(t * 120) % self.fh, axis=0) - 0.5
        dx = cv2.resize(hot * wobble * 14, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        dy = cv2.resize(hot * 4, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return cv2.remap(out, self.gx + dx, self.gy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def _spawn(self, edge, sparks, puffs):
        xs, ys = self._sample(edge, sparks, stride=2)
        k = len(xs)
        if k:
            vx = self.rng.normal(0, 45, k) + self.drift[ys, xs] * 15
            hot = self.rng.random((k, 1))
            colors = (1 - hot) * np.array([0, 110, 255]) + hot * np.array([150, 235, 255])
            self._emit(xs, ys - 6, vx, self.rng.uniform(-260, -80, k), self.rng.uniform(0.5, 1.4, k),
                       self.rng.choice([1, 2, 2, 3], k), colors, np.full(k, SPARK))
        xs, ys = self._sample(edge, puffs, stride=4)
        k = len(xs)
        if k:
            self._emit(xs, ys - 30, self.rng.normal(0, 20, k), self.rng.uniform(-90, -40, k),
                       self.rng.uniform(1.2, 2.2, k), self.rng.uniform(10, 18, k), np.zeros((k, 3)), np.full(k, PUFF))

    def _forces(self, p, free, dt):
        spark = free & (p[:, KIND] == SPARK)
        puff = free & (p[:, KIND] == PUFF)
        p[spark, VY] -= 160 * dt                          # el aire caliente las acelera hacia arriba
        p[spark, VX] += np.sin(p[spark, Y] / 23 + self.now * 5) * 120 * dt
        p[puff, VY] *= 0.99
        p[puff, VX] += np.sin(p[puff, Y] / 60 + self.now) * 30 * dt

    def _draw_particles(self, out, p):
        puffs = p[p[:, KIND] == PUFF]
        if len(puffs):
            # Humo: manchas que crecen y se aclaran, difuminadas a 1/4 de resolución.
            q = puffs.copy()
            f = np.clip(q[:, LIFE] / q[:, MAX_LIFE], 0, 1)
            q[:, SIZE] *= 1 + 2.5 * (1 - f)
            q[:, COLOR] = (4 * f * (1 - f))[:, None]
            dens = self._splat(np.zeros((self.fh, self.fw), np.float32), q, fade=False, max_size=8, scale=0.25)
            dens = cv2.resize(cv2.GaussianBlur(dens, (0, 0), 5), (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            a = np.clip(dens * 0.8, 0, 0.32)
            out = cv2.blendLinear(out, self.smoke, 1 - a, a)
        sparks = p[p[:, KIND] != PUFF]
        if len(sparks):
            layer = self._splat(np.zeros_like(out), sparks)
            out = cv2.add(cv2.add(out, layer), glow(layer, 1.6, 2))
        return out
