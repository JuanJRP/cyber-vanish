"""Imagen de la habitación vacía que se muestra donde "desapareces"."""

import logging
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .fill import complete

DIFF_THRESHOLD = 28       # diferencia de color (0-255) para considerar que algo cambió respecto al fondo
MAX_ATTACHED = 0.6        # si "lo cambiado" ocupa más de esto, es un cambio de luz, no un objeto
REFILL_EVERY = 6.0        # s: como mucho cada cuánto se rehace el relleno de lo que nunca se vio
REFILL_GAIN = 0.01        # parte de la imagen recién vista que justifica rehacerlo
REFILL_EXPOSURE = 0.06    # ...o cambio relativo de brillo del fondo (la webcam reajustó la exposición)
MIN_SEEN_TO_FILL = 0.15   # sin haber visto al menos esto del fondo no hay de dónde copiar textura
STABLE_UPDATES = 3        # un píxel cuenta como fondo tras verse libre en tantas actualizaciones seguidas
# Arranque de la cámara: los primeros frames suelen salir negros o subiendo de brillo, y en ellos
# MediaPipe no ve a nadie. Si se aprendieran, quedaría un parche negro donde estás.
MIN_BRIGHTNESS = 12       # brillo medio (0-255) por debajo del cual el frame no vale
EXPOSURE_STABLE = 0.04    # cambio relativo de brillo entre actualizaciones que se considera estable
WARMUP_STABLE = 8         # actualizaciones seguidas con brillo estable para empezar a aprender
WARMUP_MAX = 90           # ...o, como mucho, tras tantas actualizaciones (~6 s), por si la luz parpadea

log = logging.getLogger(__name__)


class BackgroundPlate:
    def __init__(self, width, height, rate=0.15):
        self.size = (width, height)
        self.small_size = (width // 4, height // 4)
        self.rate = rate
        self.plate = None
        self.seen = np.zeros((height, width), bool)
        # Tras capturar con B el fondo ya no se reescribe: si no, acabaría absorbiendo
        # la silla y todo lo que no sea la persona.
        self.locked = False
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        self.small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        self.prev_attached = None
        self.gain = np.ones(3, np.float32)
        # Relleno rápido (cv2.inpaint, con rayas) mientras se calcula el bueno por primera vez.
        self.quick = None
        self.quick_at = -1.0
        # Relleno bueno (copiando textura real de la habitación, ver fill.py): tarda un par de
        # segundos, así que se calcula en segundo plano desde el principio y se rehace al ver más fondo.
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="relleno-fondo")
        self.job = None
        self.job_generation = 0
        self.generation = 0      # cambia al capturar con B: los rellenos de antes ya no valen
        self.filled = None
        self.filled_at = -1.0
        self.filled_seen = None  # lo que se había visto cuando se lanzó el relleno
        self.filled_level = 0.0  # brillo del fondo cuando se lanzó el relleno
        self.free_run = np.zeros((height, width), np.uint8)   # actualizaciones seguidas libre de persona
        self.ready = False       # la cámara ya terminó de arrancar
        self.updates = 0
        self.stable = 0
        self.prev_level = None

    @property
    def coverage(self):
        return float(self.seen.mean())

    def _free(self, person):
        """Zonas sin persona, con margen de seguridad (erosionando a baja resolución)."""
        small = cv2.resize((person < 0.08).astype(np.uint8), self.small_size, interpolation=cv2.INTER_AREA)
        return cv2.resize(cv2.erode(small, self.kernel), self.size, interpolation=cv2.INTER_NEAREST).astype(bool)

    def capture(self, frame, person):
        """Fija el fondo con el frame actual. Si la persona sigue en cuadro, esa zona queda pendiente."""
        free = self._free(person)
        if self.plate is None:
            self.plate = frame.copy()
        np.copyto(self.plate, frame, where=free[..., None])
        self.seen = free
        self.locked = True
        self.prev_attached = None
        self.quick = self.filled = None
        self.generation += 1
        return self.coverage

    def update(self, frame, person, now=None):
        """Aprende el fondo en las zonas donde no hay persona.

        Con now, además prepara en segundo plano el relleno de lo no visto. Pásalo solo cuando sobre
        tiempo de CPU (p. ej. mientras no hay un efecto en marcha): el relleno tarda un par de segundos.
        """
        if not self._usable(frame):
            return
        # Solo se aprende lo que lleva un rato libre: si MediaPipe te pierde un frame, no se te graba.
        self.free_run = np.where(self._free(person), np.minimum(self.free_run, 254) + 1, 0).astype(np.uint8)
        free = self.free_run >= STABLE_UPDATES
        if self.plate is None:
            self.plate = frame.copy()
        if self.locked:
            # Fondo fijado: solo se completan huecos que nunca se vieron.
            np.copyto(self.plate, frame, where=(free & ~self.seen)[..., None])
        else:
            blended = cv2.addWeighted(self.plate, 1 - self.rate, frame, self.rate, 0)
            np.copyto(self.plate, blended, where=(free & self.seen)[..., None])
            np.copyto(self.plate, frame, where=(free & ~self.seen)[..., None])
        self.seen |= free
        if now is not None:
            self._refill(now)

    def _usable(self, frame):
        """¿Sirve este frame para aprender? No mientras la cámara arranca ni si sale casi negro."""
        level = _level(frame)
        if level < MIN_BRIGHTNESS:
            self.stable = 0
            self.prev_level = level
            return False
        if not self.ready:
            self.updates += 1
            steady = self.prev_level is not None and abs(level - self.prev_level) <= EXPOSURE_STABLE * self.prev_level
            self.stable = self.stable + 1 if steady else 0
            self.ready = self.stable >= WARMUP_STABLE or self.updates >= WARMUP_MAX
        self.prev_level = level
        return self.ready

    def attached(self, frame, person):
        """Persona + todo lo que cambió respecto al fondo y está pegado a ella (silla, sombra, objetos)."""
        if not self.locked:
            return person
        live = cv2.GaussianBlur(cv2.resize(frame, self.small_size, interpolation=cv2.INTER_AREA), (5, 5), 0)
        bg = cv2.GaussianBlur(cv2.resize(self.plate, self.small_size, interpolation=cv2.INTER_AREA), (5, 5), 0)
        seen = cv2.resize(self.seen.astype(np.uint8), self.small_size, interpolation=cv2.INTER_NEAREST).astype(bool)
        body = cv2.resize(person, self.small_size, interpolation=cv2.INTER_AREA) > 0.5
        if not body.any():
            return person

        # Compensar el autoexposición de la webcam: igualar el brillo medio de las zonas libres.
        ref = seen & ~body
        if ref.any():
            gain = live[ref].mean() / max(bg[ref].mean(), 1.0)
            bg = cv2.convertScaleAbs(bg, alpha=gain)

        changed = (cv2.absdiff(live, bg).max(axis=2) > DIFF_THRESHOLD) & seen
        changed = cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_OPEN, self.small_kernel)
        changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, self.small_kernel)

        # Quedarse solo con las regiones conectadas al cuerpo.
        _, labels, stats, _ = cv2.connectedComponentsWithStats(changed | body.astype(np.uint8))
        ids = np.unique(labels[body])
        ids = ids[ids > 0]
        if not len(ids) or stats[ids, cv2.CC_STAT_AREA].max() > MAX_ATTACHED * body.size:
            return person
        mask = np.isin(labels, ids).astype(np.float32)

        if self.prev_attached is not None:
            mask = np.maximum(mask, 0.6 * self.prev_attached)   # evita parpadeos en los bordes
        self.prev_attached = mask
        mask = cv2.GaussianBlur(cv2.dilate(mask, self.small_kernel), (0, 0), 1.5)
        return np.maximum(person, cv2.resize(mask, self.size, interpolation=cv2.INTER_LINEAR))

    def get(self, now, frame=None, body=None):
        """Fondo listo para usar, con el color igualado a la imagen en vivo alrededor del cuerpo."""
        bg = self._complete(now)
        if frame is None or body is None:
            return bg
        # La webcam ajusta exposición y balance de blancos sola: si el fondo guardado queda
        # más claro u oscuro que lo de alrededor, se notaría la silueta. Se iguala por canal
        # comparando un anillo de pared justo alrededor del cuerpo.
        live = cv2.resize(frame, self.small_size, interpolation=cv2.INTER_AREA)
        ref = cv2.resize(bg, self.small_size, interpolation=cv2.INTER_AREA)
        inside = cv2.resize(body, self.small_size, interpolation=cv2.INTER_AREA) > 0.05
        ring = cv2.dilate(inside.astype(np.uint8), self.ring_kernel).astype(bool) & ~inside
        if ring.sum() > 50:
            gain = live[ring].mean(axis=0) / np.maximum(ref[ring].mean(axis=0), 1.0)
            self.gain = 0.8 * self.gain + 0.2 * np.clip(gain, 0.7, 1.4).astype(np.float32)
        if np.abs(self.gain - 1).max() < 0.01:
            return bg
        return cv2.transform(bg, np.diag(self.gain))

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)

    def _complete(self, now):
        """El fondo con los huecos nunca vistos rellenados."""
        if self.seen.all():
            return self.plate
        self._refill(now, start=False)   # durante un efecto solo se recoge: no se lanza trabajo nuevo
        if self.filled is None:
            return self._quick_fill(now)
        # Lo que se ha visto desde que se lanzó el relleno usa el fondo real; el resto, el relleno.
        return np.where(self.seen[..., None], self.plate, self.filled)

    def _refill(self, now, start=True):
        """Recoge el relleno en segundo plano si ya terminó y (con start) lanza otro si se ha visto bastante más."""
        if self.plate is None or self.seen.all():
            return
        if self.job is not None:
            if not self.job.done():
                return
            try:
                result = self.job.result()
            except Exception:   # un fallo del relleno no debe tumbar la llamada: queda el rápido
                log.exception("Falló el relleno del fondo")
                result = None
            if result is not None and self.job_generation == self.generation:
                self.filled = result
            self.job = None
        if not start or self.coverage < MIN_SEEN_TO_FILL:
            return
        level = _level(self.plate)
        stale = self.filled is None or (
            now - self.filled_at > REFILL_EVERY
            and ((self.seen & ~self.filled_seen).mean() > REFILL_GAIN
                 or abs(level - self.filled_level) > REFILL_EXPOSURE * max(self.filled_level, 1.0))
        )
        if stale:
            self.filled_level = level
            self.filled_seen = self.seen.copy()
            self.filled_at = now
            self.job_generation = self.generation
            self.job = self.pool.submit(complete, self.plate.copy(), ~self.filled_seen)

    def _quick_fill(self, now):
        """cv2.inpaint a baja resolución: inmediato pero con rayas. Solo hasta que llega el relleno bueno."""
        if self.quick is None or now - self.quick_at > 0.5:
            small = cv2.resize(self.plate, self.small_size, interpolation=cv2.INTER_AREA)
            holes = cv2.resize((~self.seen).astype(np.uint8), self.small_size, interpolation=cv2.INTER_NEAREST)
            fixed = cv2.resize(cv2.inpaint(small, holes, 5, cv2.INPAINT_TELEA), self.size)
            self.quick = self.plate.copy()
            np.copyto(self.quick, fixed, where=~self.seen[..., None])
            self.quick_at = now
        return self.quick


def _level(img):
    """Brillo medio (0-255), calculado sobre una miniatura."""
    return float(cv2.resize(img, (32, 18), interpolation=cv2.INTER_AREA).mean())
