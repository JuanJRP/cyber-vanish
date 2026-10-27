"""Efecto cyberpunk: holograma → bloques glitch de neón → nada, con onda de choque, HUD y lluvia de datos."""

import math

import cv2
import numpy as np

CYAN = (255, 255, 0)      # BGR
MAGENTA = (255, 0, 255)
YELLOW = (0, 230, 255)
BLOCK = 10                # tamaño (px) de cada bloque de la desintegración
EDGE = 0.07               # cuánto antes de irse empieza a brillar un bloque (en progreso 0..1)
HOLO = 0.16               # cuánto antes de brillar se vuelve holograma
SHOCK = 0.7               # duración (s) de la onda de choque del chasquido
GLYPHS = "01ABCDEF#$%&<>/\\|=+*"


def _grade_lut():
    """Gradación cyberpunk: sombras verde azulado, luces cálidas tirando a magenta."""
    x = np.arange(256, dtype=np.float32) / 255
    b = x * 0.85 + 0.17 * (1 - x) + 0.04
    g = x * 0.92 + 0.10 * (1 - x)
    r = x * 1.08 + 0.12 * x**2 - 0.05
    lut = np.stack([b, g, r], axis=-1) * 255
    return np.clip(lut, 0, 255).astype(np.uint8).reshape(256, 1, 3)


class CyberVanish:
    def __init__(self, width, height, duration=3.0, hud=True, ghost=False):
        self.w, self.h = width, height
        self.duration = duration
        self.hud = hud
        self.ghost = ghost
        self.state = "visible"   # visible | vanishing | vanished | appearing
        self.t0 = 0.0
        self.last_render = None
        self.next_ghost = 0.0
        self.ghost_until = 0.0
        self.rng = np.random.default_rng()
        self.order = None        # HxW 0..1: en qué momento del efecto se va cada píxel
        self.neon = None         # HxW x3: color de neón de cada bloque
        self.drift = None        # HxW: hacia qué lado salen volando los fragmentos (-1 / +1)
        self.wave = None         # HxW 0..1: orden suave (sin bloques) para la onda de holograma
        self.origin_px = None    # punto del chasquido, centro de la onda de choque
        self.subject_id = 0
        # x, y, vx, vy, vida, vida_max, tamaño, b, g, r, glifo (-1 = cuadrado)
        self.particles = np.zeros((0, 11), np.float32)
        # Grano de cámara precalculado: la foto del fondo sola parecería congelada.
        self.grain = [self.rng.integers(0, 5, (height, width, 3), dtype=np.uint8) for _ in range(6)]
        self.lut = _grade_lut()
        self.identity = np.arange(256, dtype=np.float32).reshape(256, 1, 1).repeat(3, axis=2)

    @property
    def busy(self):
        return self.state in ("vanishing", "appearing")

    def trigger(self, now, person, direction=1, origin=None):
        """origin: (x, y) normalizado desde donde se expande (chasquido). Sin origin, avanza de lado según direction."""
        if self.busy:
            return
        self._plan(person, direction, origin)
        self.origin_px = None if origin is None else (int(origin[0] * self.w), int(origin[1] * self.h))
        self.subject_id = int(self.rng.integers(0x1000, 0xFFFF))
        self.state = "vanishing" if self.state == "visible" else "appearing"
        self.t0 = now

    def _plan(self, person, direction, origin):
        gw, gh = -(-self.w // BLOCK), -(-self.h // BLOCK)
        gx, gy = np.meshgrid(np.linspace(0, 1, gw), np.linspace(0, 1, gh))
        # El recorrido se reparte sobre la silueta, no sobre toda la imagen,
        # para que la desintegración ocupe toda la duración.
        body = cv2.resize(person, (gw, gh), interpolation=cv2.INTER_AREA) > 0.3
        if not body.any():
            body[:] = True
        if origin is None:
            x0, x1 = gx[body].min(), gx[body].max()
            pos = np.clip((gx - x0) / (x1 - x0 + 1e-6), 0, 1)
            pos = pos if direction > 0 else 1 - pos
            drift = np.full_like(gx, direction)
        else:
            ox, oy = origin
            pos = np.hypot((gx - ox) * self.w / self.h, gy - oy)
            pos = np.clip(pos / pos[body].max(), 0, 1)
            drift = np.where(gx >= ox, 1.0, -1.0)
        # Manchas: zonas vecinas que se van juntas, para que no parezca un simple fundido.
        blobs = cv2.GaussianBlur(self.rng.random((gh, gw)).astype(np.float32), (0, 0), 3)
        blobs = (blobs - blobs.min()) / (np.ptp(blobs) + 1e-6)
        order = 0.45 * pos + 0.3 * blobs + 0.25 * self.rng.random((gh, gw))
        order = ((order - order.min()) / (np.ptp(order) + 1e-6)).astype(np.float32)
        # El holograma ocupa el principio: el orden real de desintegración empieza después.
        order = order * (1 - HOLO) + HOLO
        # Onda de holograma: sigue la distancia a la mano pero sin el ruido de bloques,
        # así se ve como un frente continuo que recorre el cuerpo por delante de los bloques.
        wave = 0.8 * pos + 0.2 * blobs
        wave = ((wave - wave.min()) / (np.ptp(wave) + 1e-6) * (1 - HOLO)).astype(np.float32)
        neon = np.where((self.rng.random((gh, gw)) < 0.5)[..., None], CYAN, MAGENTA).astype(np.uint8)

        def up(a):
            return cv2.resize(a, (gw * BLOCK, gh * BLOCK), interpolation=cv2.INTER_NEAREST)[: self.h, : self.w]

        self.order, self.neon, self.drift = up(order), up(neon), up(drift.astype(np.float32))
        self.wave = cv2.resize(wave, (self.w, self.h), interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------------------------ render

    def render(self, frame, person, plate, now):
        dt = 0.0 if self.last_render is None else min(now - self.last_render, 0.1)
        self.last_render = now

        if self.state == "visible":
            out = frame
        else:
            # Cobertura total: incluso el borde suave de la máscara cuenta como persona,
            # para que no quede ni un contorno translúcido.
            person = np.clip(person * 2.0, 0, 1)
            bg = plate.get(now, frame, person)
            if self.state == "vanished":
                out = self._vanished(frame, person, bg, now)
            else:
                out = self._transition(frame, person, bg, now)

        if len(self.particles):
            out = self._particles(out, dt)
        return out

    def _with_grain(self, img):
        i, j = self.rng.choice(len(self.grain), 2, replace=False)
        return cv2.subtract(cv2.add(img, self.grain[i]), self.grain[j])

    def _vanished(self, frame, person, bg, now):
        # La salida es solo la foto de la habitación: sin costuras entre lo vivo y lo guardado.
        out = self._with_grain(bg)
        if not self.ghost:
            return out
        # De vez en cuando se "cuela" la señal: un parpadeo holográfico de la silueta.
        if now >= self.next_ghost:
            self.ghost_until = now + 0.18
            self.next_ghost = now + self.rng.uniform(5, 11)
        if now < self.ghost_until:
            a = person * 0.35
            out = cv2.blendLinear(out, self._hologram(frame, now), 1 - a, a)
        return out

    def _transition(self, frame, person, bg, now):
        t = now - self.t0
        p = min(1.0, t / self.duration)
        env = math.sin(math.pi * p)   # intensidad de los efectos globales: 0 → 1 → 0
        vanishing = self.state == "vanishing"

        # Cada píxel pasa por: normal → holograma → bloque glitch de neón → se va (o al revés).
        # (Máscaras con cv2.threshold/multiply: multihilo, mucho más rápido que numpy a 720p.)
        level = p * (1 + 2 * EDGE) - EDGE
        passed = _below(self.order, level)
        before_edge = _below(self.order, level + EDGE)
        gone = passed if vanishing else 1 - passed
        edge = cv2.multiply(before_edge - passed, person)
        # Holograma: lo que la onda ya alcanzó pero aún no se ha deshecho en bloques.
        holo = cv2.multiply(cv2.multiply(_below(self.wave, level + HOLO), 1 - before_edge), person)

        # Cada bloque es opaco hasta que se va: nada de transparencias. Mientras tanto,
        # el resto de la imagen pasa poco a poco a la foto del fondo, que queda debajo de todo.
        s = p * p * (3 - 2 * p)
        s = s if vanishing else 1 - s
        a = cv2.multiply(person, gone - s) + s          # = persona·gone + (1 − persona)·s
        out = cv2.blendLinear(frame, self._with_grain(bg), 1 - a, a)

        # Holograma: al irse, tu cuerpo se vuelve cian con contorno de neón; al volver, se materializa.
        wh = holo * (self.rng.uniform(0.75, 0.95) if vanishing else self.rng.uniform(0.45, 0.7))
        out = cv2.blendLinear(out, self._hologram(frame, now), 1 - wh, wh)

        # Los bloques a punto de irse/volver se corrompen, dejan estelas y brillan en neón.
        g = edge * 0.85
        out = cv2.blendLinear(out, self._glitch(frame), 1 - g, g)
        self._streaks(out, edge, frame, 18 if vanishing else 8)
        out = self._neon_glow(out, edge)
        self._spawn(edge, frame, 70 if vanishing else 30)

        out = self._grade(out, 0.65 * env)
        out = self._shockwave(out, t)
        burst = 16 * max(0.0, 1 - t / 0.25) if self.origin_px else 0
        out = _aberration(out, int(2 + 5 * env + burst))
        out = _scanlines(out, 0.2 * env)
        out = self._tears(out, env)
        if self.hud:
            self._draw_reticle(out, person, p, vanishing)
            self._draw_hud(out, p, vanishing)

        if p >= 1:
            if vanishing:
                self.state = "vanished"
                self.next_ghost = now + self.rng.uniform(4, 8)
            else:
                self.state = "visible"
        return out

    # ----------------------------------------------------------------- piezas

    def _hologram(self, frame, now):
        """Versión holográfica de la imagen: monocromo cian, contornos de neón y barrido de escaneo."""
        h, w = self.h // 2, self.w // 2
        gray = cv2.cvtColor(cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        edges = cv2.dilate(cv2.Canny(gray, 50, 130), np.ones((2, 2), np.uint8))
        holo = cv2.multiply(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), (1.4, 1.2, 0.3, 0))
        holo = cv2.add(holo, (80, 60, 0, 0))
        holo[edges > 0] = (255, 255, 170)
        holo[::2] = holo[::2] // 2
        # Barra de escaneo que recorre la imagen de arriba abajo.
        y = int((now * 260) % h)
        holo[y : y + 6] = cv2.add(holo[y : y + 6], (90, 90, 40, 0))
        return cv2.resize(holo, (self.w, self.h), interpolation=cv2.INTER_LINEAR)

    def _glitch(self, frame):
        # Se trabaja a media resolución: el resultado es pixelado de todas formas y cuesta 4 veces menos.
        h, w = self.h // 2, self.w // 2
        src = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        g = src.copy()
        g[..., 2] = np.roll(src[..., 2], 5, axis=1)
        g[..., 0] = np.roll(src[..., 0], -5, axis=1)
        for _ in range(8):
            y = int(self.rng.integers(0, h))
            y2 = y + int(self.rng.integers(2, 15))
            g[y:y2] = np.roll(g[y:y2], int(self.rng.integers(-25, 25)), axis=1)
        # Bloques pixelados al azar.
        small = cv2.resize(g, (w // 7, h // 7), interpolation=cv2.INTER_AREA)
        pix = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        blocks = cv2.resize(
            (self.rng.random((h // 14, w // 14)) < 0.35).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        )
        np.copyto(g, pix, where=blocks[..., None].astype(bool))
        # Tinte cian: 80 % imagen + 20 % cian.
        g = cv2.add(cv2.convertScaleAbs(g, alpha=0.8), (51, 51, 0, 0))
        return cv2.resize(g, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

    def _streaks(self, out, edge, frame, n):
        """Estelas de píxeles arrastrados desde los bloques que se desintegran (pixel stretch)."""
        idx = np.flatnonzero(edge[::4, ::4].ravel() > 0.5)
        if not len(idx):
            return
        idx = self.rng.choice(idx, min(n, len(idx)), replace=False)
        ys, xs = np.divmod(idx, edge[::4, ::4].shape[1])
        for y, x in zip(ys * 4, xs * 4):
            length = int(self.rng.uniform(40, 240) * self.drift[y, x])
            color = tuple(int(c) for c in frame[y, x])
            cv2.rectangle(out, (int(x), int(y)), (int(x) + length, int(y) + int(self.rng.integers(2, 6))), color, -1)

    def _neon_glow(self, out, edge):
        e = cv2.convertScaleAbs(edge, alpha=180)
        layer = cv2.multiply(self.neon, cv2.merge([e, e, e]), scale=1 / 255)
        small = cv2.resize(layer, (self.w // 4, self.h // 4), interpolation=cv2.INTER_AREA)
        glow = cv2.resize(cv2.GaussianBlur(small, (0, 0), 3), (self.w, self.h))
        return cv2.add(cv2.add(out, layer), cv2.convertScaleAbs(glow, alpha=2.0))

    def _grade(self, out, k):
        if k < 0.02:
            return out
        # Mezclar las tablas (256 valores) es mucho más barato que mezclar imágenes.
        lut = ((1 - k) * self.identity + k * self.lut).astype(np.uint8)
        return cv2.LUT(out, lut)

    def _shockwave(self, out, t):
        """Anillo de neón que sale de la mano al chasquear, con destello inicial."""
        if self.origin_px is None or t > SHOCK:
            return out
        k = t / SHOCK
        r = int((1 - (1 - k) ** 3) * max(self.w, self.h) * 0.9)
        layer = np.zeros_like(out)
        cv2.circle(layer, self.origin_px, r, CYAN, 8, cv2.LINE_AA)
        cv2.circle(layer, self.origin_px, int(r * 0.93), MAGENTA, 3, cv2.LINE_AA)
        cv2.circle(layer, self.origin_px, int(r * 0.86), YELLOW, 1, cv2.LINE_AA)
        small = cv2.resize(layer, (self.w // 4, self.h // 4), interpolation=cv2.INTER_AREA)
        glow = cv2.resize(cv2.GaussianBlur(small, (0, 0), 4), (self.w, self.h))
        layer = cv2.add(layer, cv2.convertScaleAbs(glow, alpha=2.5))
        out = cv2.add(out, cv2.convertScaleAbs(layer, alpha=1 - k))
        if t < 0.1:
            v = 140 * (1 - t / 0.1)
            out = cv2.add(out, (v, v, v * 0.8, 0))
        return out

    def _spawn(self, edge, frame, n):
        idx = np.flatnonzero(edge.ravel() > 0.5)
        if not len(idx):
            return
        idx = self.rng.choice(idx, min(n, len(idx)), replace=False)
        ys, xs = np.divmod(idx, self.w)
        k = len(idx)
        # Fragmentos que suben y se alejan, como ceniza digital; un tercio son caracteres de datos.
        vx = self.drift[ys, xs] * self.rng.uniform(30, 160, k) + self.rng.normal(0, 40, k)
        vy = self.rng.uniform(-150, -20, k)
        life = self.rng.uniform(0.4, 1.2, k)
        colors = frame[ys, xs].astype(np.float32)
        kind = self.rng.random(k)
        colors[kind < 0.35] = CYAN
        colors[(kind >= 0.35) & (kind < 0.6)] = MAGENTA
        glyph = np.where(self.rng.random(k) < 0.33, self.rng.integers(0, len(GLYPHS), k), -1)
        is_glyph = glyph >= 0
        colors[is_glyph] = np.array([CYAN, MAGENTA, YELLOW], np.float32)[self.rng.integers(0, 3, is_glyph.sum())]
        vy[is_glyph] *= 1.5
        new = np.column_stack(
            [xs, ys, vx, vy, life, life, self.rng.uniform(2, 6, k), colors, glyph]
        ).astype(np.float32)
        self.particles = np.concatenate([self.particles, new])[-1500:]

    def _particles(self, out, dt):
        p = self.particles
        p[:, 0] += p[:, 2] * dt
        p[:, 1] += p[:, 3] * dt
        p[:, 2] *= 0.97
        p[:, 4] -= dt
        p = p[(p[:, 4] > 0) & (p[:, 0] > -10) & (p[:, 0] < self.w + 10) & (p[:, 1] > -10)]
        self.particles = p

        layer = np.zeros_like(out)
        for x, y, _, _, life, max_life, size, b, g, r, glyph in p.tolist():
            f = life / max_life
            color = (b * f, g * f, r * f)
            if glyph >= 0:
                cv2.putText(layer, GLYPHS[int(glyph)], (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN, 1.1, color, 1)
            else:
                s = int(size)
                cv2.rectangle(layer, (int(x), int(y)), (int(x) + s, int(y) + s), color, -1)
        return cv2.add(out, layer)

    def _tears(self, out, env):
        """Franjas horizontales desplazadas, como señal corrupta."""
        if self.rng.random() > 0.1 + 0.4 * env:
            return out
        out = out.copy()
        for _ in range(int(self.rng.integers(1, 4))):
            y = int(self.rng.integers(0, self.h - 1))
            hgt = int(self.rng.integers(3, 25))
            out[y : y + hgt] = np.roll(out[y : y + hgt], int(self.rng.integers(-60, 60)), axis=1)
        return out

    def _draw_reticle(self, img, person, p, vanishing):
        """Mira de escaneo con esquinas que encuadra a la persona, con etiqueta y columna de datos."""
        small = person[::8, ::8] > 0.5
        if not small.any():
            return
        ys, xs = np.nonzero(small)
        j = lambda: int(self.rng.integers(-2, 3))   # temblor de señal
        x0, y0 = int(xs.min() * 8) - 14 + j(), int(ys.min() * 8) - 14 + j()
        x1, y1 = int(xs.max() * 8) + 14 + j(), int(ys.max() * 8) + 14 + j()
        x0, y0, x1, y1 = max(x0, 4), max(y0, 60), min(x1, self.w - 4), min(y1, self.h - 4)
        arm = 34
        for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            cv2.line(img, (cx, cy), (cx + dx * arm, cy), CYAN, 2, cv2.LINE_AA)
            cv2.line(img, (cx, cy), (cx, cy + dy * arm), CYAN, 2, cv2.LINE_AA)
        label = f"SUBJECT_{self.subject_id:04X} // {'DEREZZING' if vanishing else 'REZZING'} {int(p * 100):3d}%"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1.1, 1)
        cv2.rectangle(img, (x0, y0 - 24), (x0 + tw + 8, y0 - 4), (40, 0, 40), -1)
        cv2.putText(img, label, (x0 + 4, y0 - 9), cv2.FONT_HERSHEY_PLAIN, 1.1, YELLOW, 1, cv2.LINE_AA)
        if x1 + 120 < self.w:
            for i in range(8):
                hexline = "".join(self.rng.choice(list("0123456789ABCDEF"), 6))
                color = MAGENTA if self.rng.random() < 0.2 else CYAN
                cv2.putText(img, hexline, (x1 + 12, y0 + 20 + i * 18), cv2.FONT_HERSHEY_PLAIN, 0.9, color, 1)

    def _draw_hud(self, img, p, vanishing):
        title = "SIGNAL_LOST // CLOAK.EXE" if vanishing else "REBOOT // REBUILDING_SUBJECT"
        filled = int(p * 20)
        noise = "".join(self.rng.choice(list("0123456789ABCDEF"), 8))
        lines = [title, f"[{'#' * filled}{'.' * (20 - filled)}] {int(p * 100):3d}%", f"0x{noise}"]
        for i, text in enumerate(lines):
            y = 34 + i * 24
            cv2.putText(img, text, (22, y), cv2.FONT_HERSHEY_PLAIN, 1.4, MAGENTA, 2, cv2.LINE_AA)
            cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_PLAIN, 1.4, CYAN, 2, cv2.LINE_AA)


def _below(values, limit):
    """1.0 donde values < limit, 0.0 en el resto (float32)."""
    return cv2.threshold(values, limit - 1e-6, 1.0, cv2.THRESH_BINARY_INV)[1]


def _aberration(img, shift):
    out = img.copy()
    out[..., 2] = np.roll(img[..., 2], shift, axis=1)
    out[..., 0] = np.roll(img[..., 0], -shift, axis=1)
    return out


def _scanlines(img, strength):
    if strength <= 0.01:
        return img
    out = img.copy()
    out[::3] = cv2.convertScaleAbs(out[::3], alpha=1 - strength)
    return out
