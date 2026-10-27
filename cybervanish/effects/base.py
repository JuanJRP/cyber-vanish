"""Base común de los efectos de desaparición: estados, frente de avance, fondo y partículas.

Para crear un efecto nuevo: hereda de Vanish, ponle `name`, `key` y `summary`,
decóralo con @register e implementa _transition(). Ver README > "Crear tu propio efecto".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import cv2
import numpy as np

if TYPE_CHECKING:
    from ..background import BackgroundPlate

_REGISTRY: dict[str, type[Vanish]] = {}


def register(cls: type[Vanish]) -> type[Vanish]:
    """Decorador: deja el efecto disponible en --effect y en su tecla."""
    if not cls.name or not cls.key:
        raise ValueError(f"{cls.__name__} necesita name y key")
    if cls.name in _REGISTRY:
        raise ValueError(f"Ya hay un efecto llamado {cls.name!r}")
    taken = {e.key: e.name for e in _REGISTRY.values()}
    if cls.key in taken:
        raise ValueError(f"La tecla {cls.key!r} de {cls.name!r} ya la usa {taken[cls.key]!r}")
    _REGISTRY[cls.name] = cls
    return cls


def available() -> dict[str, type[Vanish]]:
    """Efectos registrados, por nombre, en orden de tecla."""
    return dict(sorted(_REGISTRY.items(), key=lambda item: item[1].key))


def create(name: str, width: int, height: int, **options) -> Vanish:
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Efecto desconocido {name!r}. Disponibles: {', '.join(available())}") from None
    return cls(width, height, **options)


class Vanish:
    name: ClassVar[str] = ""       # nombre para --effect
    key: ClassVar[str] = ""        # tecla para elegirlo en vivo
    summary: ClassVar[str] = ""    # una línea para --list-effects

    GRID = 4                 # resolución (px) del mapa de orden
    MIX = (0.5, 0.3, 0.2)    # peso de recorrido, manchas y ruido en el orden en que se va cada píxel
    BLOB_SIGMA = 3           # tamaño de las manchas (en celdas de GRID)
    GRAVITY = 0.0            # px/s² hacia abajo para las partículas
    DRAG = 0.97              # frenado horizontal de las partículas por frame
    MAX_PARTICLES = 1500

    def __init__(self, width: int, height: int, duration: float = 3.0, hud: bool = True, ghost: bool = False):
        self.w, self.h = width, height
        self.duration = duration
        self.hud = hud
        self.ghost = ghost
        self.state = "visible"   # visible | vanishing | vanished | appearing
        self.t0 = 0.0
        self.last_render: float | None = None
        self.rng = np.random.default_rng()
        self.order = np.zeros((height, width), np.float32)   # 0..1: en qué momento del efecto se va cada píxel
        self.drift = np.ones((height, width), np.float32)    # hacia qué lado salen volando los trozos (-1 / +1)
        self.origin_px: tuple[int, int] | None = None        # punto del chasquido
        # x, y, vx, vy, vida, vida_max, tamaño, b, g, r, glifo (-1 = cuadrado)
        self.particles = np.zeros((0, 11), np.float32)
        # Grano de cámara precalculado: la foto del fondo sola parecería congelada.
        self.grain = [self.rng.integers(0, 5, (height, width, 3), dtype=np.uint8) for _ in range(6)]

    @property
    def busy(self) -> bool:
        return self.state in ("vanishing", "appearing")

    def take_over(self, old: Vanish, now: float) -> None:
        """Sigue donde lo dejó otro efecto (cambio de efecto en vivo, siempre fuera de una transición)."""
        self.state = old.state
        if self.state == "vanished":
            self._finish(True, now)

    def trigger(self, now: float, person: np.ndarray, direction: int = 1, origin: tuple[float, float] | None = None):
        """origin: (x, y) normalizado desde donde se expande (chasquido). Sin origin, avanza de lado según direction."""
        if self.busy:
            return
        self._plan(person, direction, origin)
        self.origin_px = None if origin is None else (int(origin[0] * self.w), int(origin[1] * self.h))
        self.state = "vanishing" if self.state == "visible" else "appearing"
        self.t0 = now

    # ------------------------------------------------------------------ plan

    def _travel(self, person, gw, gh, direction, origin):
        """Recorrido 0..1 sobre la silueta (de un lado al otro, o desde la mano) y hacia dónde vuelan los trozos."""
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
        return pos, drift

    def _blobs(self, gw, gh, sigma):
        """Manchas: zonas vecinas que se van juntas, para que no parezca un simple fundido."""
        return normalize(cv2.GaussianBlur(self.rng.random((gh, gw)).astype(np.float32), (0, 0), sigma))

    def _plan(self, person, direction, origin):
        """Calcula self.order y self.drift al empezar cada transición."""
        gw, gh = -(-self.w // self.GRID), -(-self.h // self.GRID)
        pos, drift = self._travel(person, gw, gh, direction, origin)
        kp, kb, kn = self.MIX
        order = kp * pos + kb * self._blobs(gw, gh, self.BLOB_SIGMA) + kn * self.rng.random((gh, gw))
        self.order = cv2.resize(normalize(order), (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        self.drift = cv2.resize(drift.astype(np.float32), (self.w, self.h), interpolation=cv2.INTER_NEAREST)

    # ---------------------------------------------------------------- render

    def render(self, frame: np.ndarray, person: np.ndarray, plate: BackgroundPlate, now: float) -> np.ndarray:
        """frame: BGR uint8 HxWx3. person: máscara float32 HxW 0..1. Devuelve el frame de salida."""
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
                t = now - self.t0
                p = min(1.0, t / self.duration)
                vanishing = self.state == "vanishing"
                out = self._transition(frame, person, bg, now, t, p, vanishing)
                if p >= 1:
                    self.state = "vanished" if vanishing else "visible"
                    self._finish(vanishing, now)

        if len(self.particles):
            out = self._particles(out, dt)
        return out

    def _transition(self, frame, person, bg, now, t, p, vanishing):
        """Un frame del efecto. t: segundos desde el gesto; p: progreso 0..1 (también al aparecer)."""
        raise NotImplementedError

    def _finish(self, vanishing, now):
        """Se llama al terminar de desaparecer o de aparecer."""

    def _vanished(self, frame, person, bg, now):
        # La salida es solo la foto de la habitación: sin costuras entre lo vivo y lo guardado.
        return self._with_grain(bg)

    # ---------------------------------------------------------------- piezas

    def _with_grain(self, img):
        i, j = self.rng.choice(len(self.grain), 2, replace=False)
        return cv2.subtract(cv2.add(img, self.grain[i]), self.grain[j])

    def _front(self, p, vanishing, margin=0.0):
        """Distancia de cada píxel al frente de avance: > 0 sigue ahí, <= 0 ya se fue.

        margin: cuánto antes de p = 0 empieza el frente y cuánto después de p = 1 acaba,
        para que la banda del borde también recorra los extremos.
        """
        level = np.float32(p * (1 + 2 * margin) - margin)
        return self.order - level if vanishing else level - self.order

    def _background(self, frame, bg, person, gone, p, vanishing):
        """La persona se sustituye por el fondo donde ya se fue (gone = 1); el resto de la imagen
        pasa poco a poco a la foto del fondo, que queda debajo de todo."""
        s = np.float32(smoothstep(0, 1, p if vanishing else 1 - p))
        a = cv2.multiply(person, gone - s) + s   # = persona·gone + (1 − persona)·s
        return cv2.blendLinear(frame, self._with_grain(bg), 1 - a, a)

    def _emit(self, xs, ys, vx, vy, life, size, colors, glyph=None):
        k = len(xs)
        if not k:
            return
        glyph = np.full(k, -1) if glyph is None else glyph
        new = np.column_stack([xs, ys, vx, vy, life, life, size, colors, glyph]).astype(np.float32)
        self.particles = np.concatenate([self.particles, new])[-self.MAX_PARTICLES :]

    def _sample(self, mask, n, stride=1):
        """Hasta n posiciones (xs, ys) al azar donde mask > 0.5, mirando uno de cada `stride` píxeles."""
        sub = mask[::stride, ::stride]
        idx = np.flatnonzero(sub.ravel() > 0.5)
        if not len(idx):
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        idx = self.rng.choice(idx, min(n, len(idx)), replace=False)
        ys, xs = np.divmod(idx, sub.shape[1])
        return xs * stride, ys * stride

    def _particles(self, out, dt):
        p = self.particles
        p[:, 0] += p[:, 2] * dt
        p[:, 1] += p[:, 3] * dt
        p[:, 3] += self.GRAVITY * dt
        p[:, 2] *= self.DRAG
        p[:, 4] -= dt
        inside = (p[:, 0] > -50) & (p[:, 0] < self.w + 50) & (p[:, 1] > -50) & (p[:, 1] < self.h + 50)
        p = p[(p[:, 4] > 0) & inside]
        self.particles = p
        return self._draw_particles(out, p)

    def _draw_particles(self, out, p):
        """Por defecto: cuadrados opacos del color de cada partícula."""
        out = out.copy()
        for x, y, _, _, _, _, size, b, g, r, _ in p.tolist():
            s = max(1, int(size))
            cv2.rectangle(out, (int(x), int(y)), (int(x) + s - 1, int(y) + s - 1), (b, g, r), -1)
        return out

    def _splat(self, img, p, fade=True):
        """Dibuja en img muchas partículas pequeñas (hasta 3 px) de golpe, sin bucle de Python."""
        xs, ys = p[:, 0].astype(np.int32), p[:, 1].astype(np.int32)
        size = np.clip(p[:, 6], 1, 3).astype(np.int32)
        colors = p[:, 7:10] * (p[:, 4:5] / p[:, 5:6]) if fade else p[:, 7:10]
        colors = np.clip(colors, 0, 255).astype(np.uint8)
        for dy in range(3):
            for dx in range(3):
                sel = size > max(dx, dy)
                x, y = xs[sel] + dx, ys[sel] + dy
                ok = (x >= 0) & (x < self.w) & (y >= 0) & (y < self.h)
                img[y[ok], x[ok]] = colors[sel][ok]
        return img


def smoothstep(a, b, x):
    x = np.clip((x - a) / (b - a), 0, 1)
    return x * x * (3 - 2 * x)


def normalize(a):
    return ((a - a.min()) / (np.ptp(a) + 1e-6)).astype(np.float32)


def below(values, limit):
    """1.0 donde values < limit, 0.0 en el resto (float32)."""
    return cv2.threshold(values, limit - 1e-6, 1.0, cv2.THRESH_BINARY_INV)[1]


def glow(layer, alpha=2.0, sigma=3):
    """Halo de luz alrededor de una capa brillante (calculado a 1/4 de resolución, que es lo caro)."""
    h, w = layer.shape[:2]
    small = cv2.resize(layer, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    halo = cv2.resize(cv2.GaussianBlur(small, (0, 0), sigma), (w, h))
    return cv2.convertScaleAbs(halo, alpha=alpha)
