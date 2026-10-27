"""Efectos de desaparición.

Cada módulo registra su efecto con @register al importarse. Para añadir uno,
crea un archivo aquí e impórtalo en la lista de abajo.

Inspirados en desapariciones de anime: afterimage, crows, leaves, smoke, timestop y vortex.
"""

from . import (  # noqa: F401  (se importan para que registren sus efectos)
    afterimage,
    crows,
    cyber,
    dust,
    fire,
    ghost,
    leaves,
    matrix,
    melt,
    pixel,
    smoke,
    teleport,
    timestop,
    vortex,
)
from .base import Vanish, available, create, register

__all__ = ["Vanish", "available", "create", "register"]
