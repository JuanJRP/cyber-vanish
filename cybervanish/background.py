"""Imagen de la habitación vacía que se muestra donde "desapareces"."""

import cv2
import numpy as np

DIFF_THRESHOLD = 28       # diferencia de color (0-255) para considerar que algo cambió respecto al fondo
MAX_ATTACHED = 0.6        # si "lo cambiado" ocupa más de esto, es un cambio de luz, no un objeto


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
        self.filled = None
        self.filled_at = -1.0

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
        self.filled = None
        return self.coverage

    def update(self, frame, person):
        """Aprende el fondo en las zonas donde no hay persona."""
        if self.plate is None:
            self.plate = frame.copy()
        free = self._free(person)
        if self.locked:
            # Fondo fijado: solo se completan huecos que nunca se vieron.
            np.copyto(self.plate, frame, where=(free & ~self.seen)[..., None])
        else:
            blended = cv2.addWeighted(self.plate, 1 - self.rate, frame, self.rate, 0)
            np.copyto(self.plate, blended, where=(free & self.seen)[..., None])
            np.copyto(self.plate, frame, where=(free & ~self.seen)[..., None])
        self.seen |= free

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

    def _complete(self, now):
        """El fondo con los huecos nunca vistos rellenados por inpainting."""
        if self.seen.all():
            return self.plate
        if self.filled is None or now - self.filled_at > 0.5:
            w, h = self.size
            small = cv2.resize(self.plate, self.small_size, interpolation=cv2.INTER_AREA)
            holes = cv2.resize((~self.seen).astype(np.uint8), self.small_size, interpolation=cv2.INTER_NEAREST)
            fixed = cv2.resize(cv2.inpaint(small, holes, 5, cv2.INPAINT_TELEA), self.size)
            self.filled = self.plate.copy()
            np.copyto(self.filled, fixed, where=~self.seen[..., None])
            self.filled_at = now
        return self.filled
