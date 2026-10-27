"""Efectos de desaparición.

Cada módulo registra su efecto con @register al importarse. Para añadir uno,
crea un archivo aquí e impórtalo en la lista de abajo.
"""

from . import cyber, dust, fire, ghost, matrix, melt, pixel, teleport  # noqa: F401  (registran sus efectos)
from .base import Vanish, available, create, register

__all__ = ["Vanish", "available", "create", "register"]
