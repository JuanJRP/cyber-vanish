"""Relleno de huecos copiando parches de la propia imagen (completado por ejemplos, Wexler et al. 2007).

Para la zona que tapa la persona cuando no se capturó el fondo: en vez de estirar los bordes hacia
dentro (las rayas de cv2.inpaint), cada trozo del hueco copia textura real de la habitación —pared,
estantería, sofá— que encaja con lo que tiene alrededor. Va de grueso a fino: decide qué va dónde a
baja resolución, afina en escalas mayores y al final copia los píxeles a resolución completa.
"""

import cv2
import numpy as np

PATCH = 7                  # lado de los parches que se comparan (en px de cada escala)
R = PATCH // 2
WORK_WIDTH = 320           # ancho de la escala más fina en la que se buscan parches
LEVELS = 3                 # escalas: WORK_WIDTH / 4, / 2 y / 1
ITERATIONS = (6, 4, 3)     # vueltas de búsqueda + votación en cada escala, de gruesa a fina
PCA_DIMS = 24              # los parches se comparan en este número de dimensiones
COHERENCE = 0.8            # preferencia por seguir copiando de donde copia el vecino (Ashikhmin 2001)
SPREAD = 6                 # pasadas de propagación de esa coherencia por iteración
# Dureza de la votación: con 1 (Wexler) cada píxel es la media de muchos parches y la textura se
# emborrona; más alto, manda el parche que mejor encaja y se conserva la textura copiada.
VOTE_SHARPNESS = 6.0
FINAL_VOTE = 2             # radio (px a resolución completa) del suavizado de costuras al final
TONE_SIGMA = 30            # px (a 1280 de ancho): por debajo de esta escala, el tono sale del entorno
NEIGHBOURS = ((0, 1), (0, -1), (1, 0), (-1, 0))


def complete(image, hole):
    """Devuelve image (BGR uint8) con hole (bool HxW, True = desconocido) relleno."""
    if not hole.any():
        return image.copy()
    h, w = hole.shape
    scale = min(1.0, WORK_WIDTH / w)
    sizes = [(max(8, round(w * scale / 2**k)), max(8, round(h * scale / 2**k))) for k in reversed(range(LEVELS))]

    cur = None
    for (lw, lh), iterations in zip(sizes, ITERATIONS, strict=True):
        known = cv2.resize(image, (lw, lh), interpolation=cv2.INTER_AREA).astype(np.float32)
        # Cualquier píxel que toque el hueco cuenta como hueco: así las fuentes son 100 % conocidas.
        lhole = cv2.resize(hole.astype(np.float32), (lw, lh), interpolation=cv2.INTER_AREA) > 0.01
        if cur is None:
            cur = _smooth_fill(known, lhole)
        else:
            up = cv2.resize(cur, (lw, lh), interpolation=cv2.INTER_LINEAR)
            cur = np.where(lhole[..., None], up, known)
        level = _Level(cur, lhole)
        if level.index is None:
            return _fallback(image, hole)
        for _ in range(iterations):
            level.search(cur)
            cur = level.vote(cur)
    return _match_tone(_full_resolution(image, hole, level, cur), image, hole)


class _Level:
    """Búsqueda de parches en una escala: fuentes (parches 100 % conocidos) e índice para buscarlos."""

    def __init__(self, cur, hole):
        self.h, self.w = hole.shape
        self.hole = hole
        full = cv2.erode((~hole).astype(np.uint8), np.ones((PATCH, PATCH), np.uint8)) > 0
        # Las fuentes están lejos del borde para que sus parches no se salgan de la imagen.
        full[:R], full[-R:], full[:, :R], full[:, -R:] = False, False, False, False
        self.source = full
        self.src_y, self.src_x = np.nonzero(full)
        # Centros de los parches que tocan el hueco: todos votan por sus píxeles.
        self.target = cv2.dilate(hole.astype(np.uint8), np.ones((PATCH, PATCH), np.uint8)) > 0
        self.tgt_y, self.tgt_x = np.nonzero(self.target)
        self.hole_y, self.hole_x = np.nonzero(hole)
        self.off_y = np.zeros((self.h, self.w), np.int32)
        self.off_x = np.zeros((self.h, self.w), np.int32)
        self.weight = np.zeros((self.h, self.w), np.float32)
        self.index = None
        if len(self.src_y) < 16:
            return
        feats = _features(cur)
        src = feats[self.src_y, self.src_x]
        rng = np.random.default_rng(0)
        fit = src[rng.choice(len(src), min(len(src), 6000), replace=False)]
        self.mean, self.basis = cv2.PCACompute(fit, mean=None, maxComponents=min(PCA_DIMS, fit.shape[1]))
        self.index = cv2.flann_Index(self._project(src), dict(algorithm=1, trees=4))

    def _project(self, feats):
        return np.ascontiguousarray((feats - self.mean) @ self.basis.T, dtype=np.float32)

    def search(self, cur):
        """Para cada parche que toca el hueco, el parche conocido más parecido."""
        feats = _features(cur)
        proj = self._project(feats.reshape(-1, feats.shape[2])).reshape(self.h, self.w, -1)
        ty, tx = self.tgt_y, self.tgt_x
        query = proj[ty, tx]

        # Candidato 1: el más parecido de toda la imagen (búsqueda aproximada).
        ids, _ = self.index.knnSearch(query, 1, params=dict(checks=48))
        best_y, best_x = self.src_y[ids[:, 0]], self.src_x[ids[:, 0]]
        best = ((query - proj[best_y, best_x]) ** 2).sum(axis=1)
        # Candidato 2: el de la vuelta anterior, que ya era coherente con sus vecinos.
        best_y, best_x, best = self._try(proj, query, ty + self.off_y[ty, tx], tx + self.off_x[ty, tx],
                                         best_y, best_x, best)
        # Candidatos 3+: seguir copiando de donde copia cada vecino. Así se copian trozos grandes y
        # contiguos de textura real (nítidos) en vez de una media de trocitos sueltos (borrosa).
        # En tablero de ajedrez: si todos cambiaran a la vez, cada píxel copiaría a un vecino que
        # también está cambiando y la coherencia nunca cuajaría.
        parity = (ty + tx) & 1
        for _ in range(SPREAD):
            for color in (0, 1):
                self.off_y[ty, tx], self.off_x[ty, tx] = best_y - ty, best_x - tx
                sel = parity == color
                sy, sx, q = ty[sel], tx[sel], query[sel]
                by, bx, bd = best_y[sel], best_x[sel], best[sel]
                for ey, ex in NEIGHBOURS:
                    ny, nx = np.clip(sy + ey, 0, self.h - 1), np.clip(sx + ex, 0, self.w - 1)
                    by, bx, bd = self._try(proj, q, sy + self.off_y[ny, nx], sx + self.off_x[ny, nx], by, bx, bd)
                best_y[sel], best_x[sel], best[sel] = by, bx, bd
        self.off_y[ty, tx], self.off_x[ty, tx] = best_y - ty, best_x - tx
        # Los parches que encajan mejor pesan más en la votación (Wexler: e^(-d²/2σ²), endurecido).
        sigma2 = max(float(np.percentile(best, 75)), 1e-3)
        self.weight[ty, tx] = np.exp(-best * VOTE_SHARPNESS / (2 * sigma2)) + 1e-12

    def _try(self, proj, query, cy, cx, best_y, best_x, best):
        """Se queda con el candidato (cy, cx) donde sea una fuente válida y encaje mejor (con ventaja)."""
        inside = (cy >= 0) & (cy < self.h) & (cx >= 0) & (cx < self.w)
        cy, cx = np.where(inside, cy, 0), np.where(inside, cx, 0)
        ok = inside & self.source[cy, cx]
        d = ((query - proj[cy, cx]) ** 2).sum(axis=1) * (1 - COHERENCE)
        better = ok & (d < best)
        return np.where(better, cy, best_y), np.where(better, cx, best_x), np.where(better, d, best)

    def vote(self, cur):
        """Cada píxel del hueco = media ponderada de lo que le proponen los parches que lo cubren."""
        num = np.zeros((len(self.hole_y), 3), np.float32)
        den = np.zeros(len(self.hole_y), np.float32)
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                py = np.clip(self.hole_y - dy, 0, self.h - 1)
                px = np.clip(self.hole_x - dx, 0, self.w - 1)
                wgt = self.weight[py, px] * self.target[py, px]
                sy = np.clip(self.hole_y + self.off_y[py, px], 0, self.h - 1)
                sx = np.clip(self.hole_x + self.off_x[py, px], 0, self.w - 1)
                num += cur[sy, sx] * wgt[:, None]
                den += wgt
        out = cur.copy()
        ok = den > 0
        out[self.hole_y[ok], self.hole_x[ok]] = num[ok] / den[ok, None]
        return out


def _features(img):
    """Cada píxel → su parche PATCH×PATCH aplanado (bordes reflejados)."""
    p = np.pad(img, ((R, R), (R, R), (0, 0)), mode="reflect")
    h, w = img.shape[:2]
    return np.concatenate([p[dy : dy + h, dx : dx + w] for dy in range(PATCH) for dx in range(PATCH)], axis=2)


def _full_resolution(image, hole, level, cur):
    """Copia los píxeles reales a resolución completa siguiendo los parches elegidos en la escala fina."""
    h, w = hole.shape
    fy, fx = h / level.h, w / level.w
    # Desplazamiento de cada píxel a resolución completa (el de su píxel en la escala fina, escalado).
    off_y = cv2.resize(level.off_y.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) * fy
    off_x = cv2.resize(level.off_x.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) * fx
    valid = cv2.resize(level.target.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    weight = cv2.resize(level.weight, (w, h), interpolation=cv2.INTER_NEAREST) * valid
    ys, xs = np.nonzero(hole)
    num = np.zeros((len(ys), 3), np.float32)
    den = np.zeros(len(ys), np.float32)
    # Pequeña votación a resolución completa: suaviza las costuras entre trozos copiados de sitios distintos.
    for dy in range(-FINAL_VOTE, FINAL_VOTE + 1):
        for dx in range(-FINAL_VOTE, FINAL_VOTE + 1):
            py, px = np.clip(ys - dy, 0, h - 1), np.clip(xs - dx, 0, w - 1)
            wgt = weight[py, px]
            sy = np.clip(np.rint(ys + off_y[py, px]).astype(np.int32), 0, h - 1)
            sx = np.clip(np.rint(xs + off_x[py, px]).astype(np.int32), 0, w - 1)
            wgt = wgt * ~hole[sy, sx]
            num += image[sy, sx].astype(np.float32) * wgt[:, None]
            den += wgt
    # Donde no hubo votos (raro), queda el resultado de la escala fina ampliado.
    coarse = cv2.resize(cur, (w, h), interpolation=cv2.INTER_LINEAR)[ys, xs]
    ok = den > 0
    fill = np.where(ok[:, None], num / np.maximum(den, 1e-30)[:, None], coarse)
    out = image.copy()
    out[ys, xs] = np.clip(fill, 0, 255).astype(np.uint8)
    return out


def _match_tone(filled, image, hole):
    """La textura viene de los parches copiados, pero el tono general, del entorno del hueco.

    Si alrededor del hueco hay sobre todo un sofá blanco, los parches copiados serán de sofá y
    quedaría una silueta blanca que delata dónde estabas. Se sustituyen las frecuencias bajas del
    relleno por una interpolación suave desde el borde (sin las rayas de cv2.inpaint).
    """
    h, w = hole.shape
    small = (max(8, w // 4), max(8, h // 4))
    sigma = max(1.0, TONE_SIGMA * w / 1280 / 4)
    base = cv2.resize(image, small, interpolation=cv2.INTER_AREA).astype(np.float32)
    shole = cv2.resize(hole.astype(np.float32), small, interpolation=cv2.INTER_AREA) > 0.01
    target = cv2.GaussianBlur(_membrane(base, shole), (0, 0), sigma)
    current = cv2.resize(filled, small, interpolation=cv2.INTER_AREA).astype(np.float32)
    current = cv2.GaussianBlur(current, (0, 0), sigma)
    shift = cv2.resize(target - current, (w, h), interpolation=cv2.INTER_LINEAR)
    out = filled.copy()
    out[hole] = np.clip(filled[hole] + shift[hole], 0, 255).astype(np.uint8)
    return out


def _membrane(img, hole):
    """Interpolación suave desde el borde del hueco (push-pull piramidal)."""
    w = (~hole).astype(np.float32)
    c = img * w[..., None]
    cs, ws = [c], [w]
    while min(ws[-1].shape) > 4:
        cs.append(cv2.pyrDown(cs[-1]))
        ws.append(cv2.pyrDown(ws[-1]))
    r = cs[-1] / np.maximum(ws[-1], 1e-6)[..., None]
    for c_l, w_l in zip(reversed(cs[:-1]), reversed(ws[:-1]), strict=True):
        up = cv2.pyrUp(r, dstsize=(w_l.shape[1], w_l.shape[0]))
        a = np.clip(w_l * 4, 0, 1)[..., None]
        r = a * (c_l / np.maximum(w_l, 1e-6)[..., None]) + (1 - a) * up
    return r


def _smooth_fill(img, hole):
    """Punto de partida en la escala más gruesa: relleno suave (aquí las rayas no importan, se sustituyen)."""
    u8 = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.inpaint(u8, hole.astype(np.uint8), 3, cv2.INPAINT_TELEA).astype(np.float32)


def _fallback(image, hole):
    """Sin parches de los que copiar (casi toda la imagen es hueco): relleno suave."""
    w, h = hole.shape[1], hole.shape[0]
    small = (max(8, w // 4), max(8, h // 4))
    img = cv2.resize(image, small, interpolation=cv2.INTER_AREA)
    hs = cv2.resize(hole.astype(np.uint8), small, interpolation=cv2.INTER_NEAREST)
    fixed = cv2.resize(cv2.inpaint(img, hs, 5, cv2.INPAINT_TELEA), (w, h))
    out = image.copy()
    np.copyto(out, fixed, where=hole[..., None])
    return out
