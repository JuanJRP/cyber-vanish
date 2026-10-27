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

# Columnas de Vanish.particles.
X, Y, VX, VY, LIFE, MAX_LIFE, SIZE = range(7)
COLOR = slice(7, 10)       # b, g, r
KIND = 10                  # uso libre de cada efecto (glifo, giro, tipo de partícula...)
TX, TY = 11, 12            # destino de las partículas que vuelan a su sitio (-1 = partícula libre)
COLUMNS = 13

KEY_ORDER = "1234567890abcdefghijklmnopqrstuvwxyz"   # orden del teclado: así se listan los efectos
RESERVED_KEYS = {"b": "capturar el fondo", "q": "salir", "e": "siguiente efecto", " ": "activar"}

_REGISTRY: dict[str, type[Vanish]] = {}


def register(cls: type[Vanish]) -> type[Vanish]:
    """Decorador: deja el efecto disponible en --effect y en su tecla."""
    if not cls.name or not cls.key:
        raise ValueError(f"{cls.__name__} necesita name y key")
    if cls.key not in KEY_ORDER:
        raise ValueError(f"La tecla de {cls.name!r} debe ser un número o una letra minúscula, no {cls.key!r}")
    if cls.key in RESERVED_KEYS:
        raise ValueError(f"La tecla {cls.key!r} de {cls.name!r} está reservada para {RESERVED_KEYS[cls.key]}")
    if cls.name in _REGISTRY:
        raise ValueError(f"Ya hay un efecto llamado {cls.name!r}")
    taken = {e.key: e.name for e in _REGISTRY.values()}
    if cls.key in taken:
        raise ValueError(f"La tecla {cls.key!r} de {cls.name!r} ya la usa {taken[cls.key]!r}")
    _REGISTRY[cls.name] = cls
    return cls


def available() -> dict[str, type[Vanish]]:
    """Efectos registrados, por nombre, en orden de tecla (1-9, 0 y luego letras)."""
    return dict(sorted(_REGISTRY.items(), key=lambda item: KEY_ORDER.index(item[1].key)))


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
    GRAVITY = 0.0            # px/s² hacia abajo para las partículas libres
    DRAG = 0.97              # frenado horizontal de las partículas libres por frame
    MAX_PARTICLES = 1500

    def __init__(self, width: int, height: int, duration: float = 3.0, hud: bool = True, ghost: bool = False):
        self.w, self.h = width, height
        self.duration = duration
        self.hud = hud
        self.ghost = ghost
        self.state = "visible"   # visible | vanishing | vanished | appearing
        self.t0 = 0.0
        self.now = 0.0
        self.last_render: float | None = None
        self.rng = np.random.default_rng()
        self.order = np.zeros((height, width), np.float32)   # 0..1: en qué momento del efecto se va cada píxel
        self.drift = np.ones((height, width), np.float32)    # hacia qué lado salen volando los trozos (-1 / +1)
        self.origin_px: tuple[int, int] | None = None        # punto del chasquido
        self.particles = np.zeros((0, COLUMNS), np.float32)
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
        self.last_render = self.now = now

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
        """Distancia de cada píxel al frente de avance: > 0 sigue ahí, <= 0 ya se fue (o aún no llegó).

        margin: cuánto antes de p = 0 empieza el frente y cuánto después de p = 1 acaba,
        para que la banda del borde también recorra los extremos.
        """
        level = np.float32(p * (1 + 2 * margin) - margin)
        return self.order - level if vanishing else level - self.order

    def _front_speed(self, margin=0.0):
        """Cuánto avanza el frente (en unidades de _front) por segundo."""
        return (1 + 2 * margin) / self.duration

    def _arriving(self, d, margin, lead):
        """Al aparecer (d < 0: aún no está), los píxeles que llegarán dentro de ~lead segundos.

        Devuelve (máscara, velocidad del frente). El tiempo de llegada de cada píxel es -d / velocidad:
        sirve para lanzar partículas que vuelen a su sitio y aterricen justo cuando ese píxel aparece.
        """
        speed = self._front_speed(margin)
        far = -lead * speed
        # La banda es lo que avanza el frente en ~2 frames: así no se salta ningún píxel.
        return band(d, np.float32(far - speed / 15), np.float32(far)), speed

    def _background(self, frame, bg, person, gone, p, vanishing):
        """La persona se sustituye por el fondo donde ya se fue (gone = 1); el resto de la imagen
        pasa poco a poco a la foto del fondo, que queda debajo de todo."""
        s = np.float32(smoothstep(0, 1, p if vanishing else 1 - p))
        a = cv2.multiply(person, gone - s) + s   # = persona·gone + (1 − persona)·s
        return cv2.blendLinear(frame, self._with_grain(bg), 1 - a, a)

    def _emit(self, xs, ys, vx, vy, life, size, colors, kind=None):
        """Partículas libres: salen de (xs, ys) con velocidad (vx, vy) y las mueve la física."""
        k = len(xs)
        if not k:
            return
        kind = np.full(k, -1) if kind is None else kind
        none = np.full(k, -1)
        self._add(np.column_stack([xs, ys, vx, vy, life, life, size, colors, kind, none, none]))

    def _emit_homing(self, xs, ys, tx, ty, life, size, colors, kind=None):
        """Partículas que vuelan de (xs, ys) a (tx, ty) frenando, y aterrizan justo al acabar su vida."""
        k = len(xs)
        if not k:
            return
        kind = np.full(k, -1) if kind is None else kind
        # En las que vuelan a su sitio, VX/VY guardan la distancia inicial al destino.
        self._add(np.column_stack([xs, ys, xs - tx, ys - ty, life, life, size, colors, kind, tx, ty]))

    def _add(self, rows):
        self.particles = np.concatenate([self.particles, rows.astype(np.float32)])[-self.MAX_PARTICLES :]

    def _sample(self, mask, n, stride=1):
        """Hasta n posiciones (xs, ys) al azar donde mask > 0.5, mirando uno de cada `stride` píxeles."""
        sub = mask[::stride, ::stride]
        idx = np.flatnonzero(sub.ravel() > 0.5)
        if not len(idx):
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        idx = self.rng.choice(idx, min(n, len(idx)), replace=False)
        ys, xs = np.divmod(idx, sub.shape[1])
        return xs * stride, ys * stride

    def _forces(self, p, free, dt):
        """Fuerzas propias del efecto sobre las partículas libres (viento, flotación, rebotes...)."""

    def _particles(self, out, dt):
        p = self.particles
        p[:, LIFE] -= dt
        homing = p[:, TX] >= 0
        free = ~homing
        self._forces(p, free, dt)
        p[free, X] += p[free, VX] * dt
        p[free, Y] += p[free, VY] * dt
        p[free, VY] += self.GRAVITY * dt
        p[free, VX] *= self.DRAG
        e = np.clip(p[homing, LIFE] / p[homing, MAX_LIFE], 0, 1) ** 2
        p[homing, X] = p[homing, TX] + p[homing, VX] * e
        p[homing, Y] = p[homing, TY] + p[homing, VY] * e
        inside = (p[:, X] > -60) & (p[:, X] < self.w + 60) & (p[:, Y] > -60) & (p[:, Y] < self.h + 60)
        self.particles = p = p[(p[:, LIFE] > 0) & (inside | homing)]
        return self._draw_particles(out, p)

    def _draw_particles(self, out, p):
        """Por defecto: cuadrados opacos del color de cada partícula."""
        out = out.copy()
        for row in p.tolist():
            x, y, s = int(row[X]), int(row[Y]), max(1, int(row[SIZE]))
            cv2.rectangle(out, (x, y), (x + s - 1, y + s - 1), tuple(row[COLOR]), -1)
        return out

    def _splat(self, img, p, fade=True, max_size=3, scale=1.0):
        """Dibuja en img muchas partículas pequeñas (hasta max_size px) de golpe, sin bucle de Python.

        scale: para dibujar en una capa a otra resolución (p. ej. 0.25 para una capa a 1/4).
        """
        h, w = img.shape[:2]
        xs, ys = (p[:, X] * scale).astype(np.int32), (p[:, Y] * scale).astype(np.int32)
        size = np.clip(p[:, SIZE] * scale, 1, max_size).astype(np.int32)
        colors = p[:, COLOR] * (p[:, LIFE : LIFE + 1] / p[:, MAX_LIFE : MAX_LIFE + 1]) if fade else p[:, COLOR]
        colors = np.clip(colors, 0, 255).astype(img.dtype)
        for dy in range(max_size):
            for dx in range(max_size):
                sel = size > max(dx, dy)
                x, y = xs[sel] + dx, ys[sel] + dy
                ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
                img[y[ok], x[ok]] = colors[sel][ok] if img.ndim == 3 else colors[sel][ok, 0]
        return img


def smoothstep(a, b, x):
    x = np.clip((x - a) / (b - a), 0, 1)
    return x * x * (3 - 2 * x)


def normalize(a):
    return ((a - a.min()) / (np.ptp(a) + 1e-6)).astype(np.float32)


def below(values, limit):
    """1.0 donde values < limit, 0.0 en el resto (float32)."""
    return cv2.threshold(values, limit - 1e-6, 1.0, cv2.THRESH_BINARY_INV)[1]


def band(values, lo, hi):
    """1.0 donde lo <= values < hi (float32)."""
    return below(values, hi) - below(values, lo)


def glow(layer, alpha=2.0, sigma=3):
    """Halo de luz alrededor de una capa brillante (calculado a 1/4 de resolución, que es lo caro)."""
    h, w = layer.shape[:2]
    small = cv2.resize(layer, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    halo = cv2.resize(cv2.GaussianBlur(small, (0, 0), sigma), (w, h))
    return cv2.convertScaleAbs(halo, alpha=alpha)


def gray3(img):
    """Imagen en escala de grises, en 3 canales."""
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def body_box(person, width, height):
    """(cx, cy, media anchura, arriba, abajo) de la silueta; el centro de la imagen si no hay nadie."""
    body = person > 0.5
    cols, rows = np.flatnonzero(body.any(axis=0)), np.flatnonzero(body.any(axis=1))
    if not len(cols):
        return width / 2, height / 2, width / 6, height / 4, height - 1
    # Floats de Python: un escalar de numpy float64 convertiría en float64 los mapas float32 con los que se opera.
    left, right, top, bottom = int(cols[0]), int(cols[-1]), int(rows[0]), int(rows[-1])
    return (left + right) / 2, (top + bottom) / 2, max((right - left) / 2, 20.0), top, bottom
