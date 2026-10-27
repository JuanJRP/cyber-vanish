"""Gesto de chasquido de dedos."""

import numpy as np

# Índices de puntos de MediaPipe Hands.
WRIST, THUMB_TIP, MIDDLE_MCP, MIDDLE_TIP = 0, 4, 9, 12
# Muñeca + nudillos: su media es un centro de palma estable.
PALM_IDS = [0, 5, 9, 13, 17]


def palm_center(hand):
    return hand[PALM_IDS].mean(axis=0)


class SnapDetector:
    """Chasquido = postura de chasquido (pulgar contra el corazón estirado) y, justo después,
    el sonido del chasquido. Sin micrófono, en vez del sonido vale ver el corazón caer a la palma."""

    def __init__(self, use_audio=True, window=1.0, cooldown=1.5):
        self.use_audio = use_audio
        self.window = window
        self.cooldown = cooldown
        self.pinch_at = -1.0      # último instante en postura de chasquido
        self.pinch_pos = None
        self.blocked_until = 0.0
        self.metrics = None       # (gap, reach) de la última mano vista, para --debug

    def update(self, hands, now, sound_at=None):
        """Devuelve (x, y) normalizados de la mano que chasqueó, o None."""
        released = None
        posed = False
        for hand in hands:
            scale = float(np.linalg.norm(hand[WRIST] - hand[MIDDLE_MCP])) + 1e-6
            gap = float(np.linalg.norm(hand[THUMB_TIP] - hand[MIDDLE_TIP])) / scale
            reach = float(np.linalg.norm(hand[MIDDLE_TIP] - hand[WRIST])) / scale
            self.metrics = (gap, reach)
            # En un puño el pulgar también toca el corazón, pero el corazón está doblado
            # (cerca de la muñeca, reach ~0.65). En la postura de chasquido está estirado
            # contra el pulgar (reach ~0.95-1.25, medido con chasquidos reales).
            if gap < 0.5 and reach > 0.85:
                posed = True
                self.pinch_pos = palm_center(hand)
            # Tras el chasquido el corazón cae contra la palma y se aleja del pulgar.
            elif gap > 0.6 and reach < 0.85 and now - self.pinch_at < 0.3:
                released = palm_center(hand)

        if posed:
            self.pinch_at = now

        if now < self.blocked_until or now - self.pinch_at > self.window:
            return None
        if self.use_audio:
            # Con micrófono hacen falta las dos cosas: postura + sonido de chasquido.
            fired = sound_at is not None and -0.15 < sound_at - self.pinch_at < self.window
        else:
            fired = released is not None
        if fired:
            self.blocked_until = now + self.cooldown
            self.pinch_at = -1.0
            return tuple(released if released is not None else self.pinch_pos)
        return None
