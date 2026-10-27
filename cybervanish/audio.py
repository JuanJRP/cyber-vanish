"""Escucha el micrófono y marca el instante de los chasquidos de dedos."""

import time
from collections import deque

import numpy as np

RATE = 48000
BLOCK = 512                    # ~10 ms
HISTORY = 4 * BLOCK            # se analiza con un bloque de retraso para ver cómo se apaga el sonido
EARLY = int(0.005 * RATE)      # primeros 5 ms tras el pico
LATE = (int(0.010 * RATE), int(0.020 * RATE))   # de 10 a 20 ms tras el pico
FREQS = np.fft.rfftfreq(EARLY * 2, 1 / RATE)
WINDOW = np.hanning(EARLY * 2).astype(np.float32)


class SnapListener:
    def __init__(self, device=None, sensitivity=1.0):
        self.sound_at = None
        self.background = 2e-5
        self.sensitivity = sensitivity
        self.buf = np.zeros(HISTORY, np.float32)
        # Sonidos candidatos (para --debug): (instante, aceptado, agudos/graves, centroide Hz, eco).
        self.events = deque(maxlen=50)
        self.stream = None
        try:
            import sounddevice as sd

            self.stream = sd.InputStream(
                samplerate=RATE, blocksize=BLOCK, channels=1, dtype="float32",
                device=device, callback=self._callback,
            )
            self.stream.start()
        except Exception as e:  # sin PortAudio o sin micrófono: el chasquido queda solo visual
            print(f"[!] Micrófono no disponible ({e}). El chasquido se detectará solo con la cámara.")

    @property
    def active(self):
        return self.stream is not None

    def _callback(self, data, frames, time_info, status):
        self.buf = np.concatenate([self.buf[len(data):], data[:, 0]])
        result = self._is_snap(self.buf)
        if result is None:
            return
        # El bloque analizado es el de hace ~2 bloques.
        at = time.monotonic() - 2 * BLOCK / RATE
        self.events.append((at, *result))
        if result[0]:
            self.sound_at = at

    def _is_snap(self, buf):
        """None si no hay golpe de sonido; si lo hay, (es_chasquido, agudos/graves, centroide, eco)."""
        block = buf[BLOCK : 2 * BLOCK + 1]
        # Energía aguda (derivada) en ventanas de ~1,3 ms: un chasquido dura 2-3 ms y
        # promediado en todo el bloque se diluiría.
        hf = (np.diff(block) ** 2).reshape(-1, 64).mean(axis=1)
        onset = hf.max() > self.background * 20 / self.sensitivity and np.abs(block).max() > 0.03 / self.sensitivity
        if not onset:
            # Nivel de fondo que se adapta lento (solo con bloques sin golpe).
            self.background = 0.98 * self.background + 0.02 * max(float(hf.mean()), 1e-7)
            return None

        peak = BLOCK + int(np.argmax(np.abs(block)))
        # Espectro alrededor del pico: el chasquido es casi todo agudo; un golpe en la mesa,
        # una palmada o la voz tienen mucha energía grave.
        spec = np.abs(np.fft.rfft(buf[peak - EARLY : peak + EARLY] * WINDOW)) ** 2
        high = spec[FREQS > 1800].sum()
        low = spec[FREQS < 1200].sum() + 1e-12
        centroid = float((spec * FREQS).sum() / (spec.sum() + 1e-12))
        # Duración: el chasquido se apaga en pocos ms; un golpe resuena más.
        # (Margen amplio: el eco de la habitación alarga un poco cualquier sonido.)
        early = float(np.mean(buf[peak : peak + EARLY] ** 2)) + 1e-12
        late = float(np.mean(buf[peak + LATE[0] : peak + LATE[1]] ** 2))
        ratio, echo = high / low, late / early
        return ratio > 2.5 and centroid > 1800 and echo < 0.3, ratio, centroid, echo

    def close(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
