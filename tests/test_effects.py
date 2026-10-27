"""Cada efecto debe recorrer desaparecer → invisible → aparecer sin errores y dejar ver el fondo al final."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from cybervanish.effects import Vanish, available, create, register
from cybervanish.effects.base import X, Y

W, H = 320, 180
FPS = 30
DURATION = 1.0


class FakePlate:
    """Fondo fijo, con la misma interfaz que BackgroundPlate.get."""

    def __init__(self, bg):
        self.bg = bg

    def get(self, now, frame=None, body=None):
        return self.bg


@pytest.fixture
def scene():
    rng = np.random.default_rng(0)
    bg = np.full((H, W, 3), (40, 90, 60), np.uint8)
    frame = bg.copy()
    person = np.zeros((H, W), np.float32)
    cv2.ellipse(person, (W // 2, H // 2 + 20), (45, 80), 0, 0, 360, 1.0, -1)
    body = rng.integers(120, 255, (H, W, 3), dtype=np.uint8)
    np.copyto(frame, body, where=person[..., None] > 0)
    return frame, person, FakePlate(bg)


def run(fx, frame, person, plate, t0, seconds):
    out = None
    for i in range(int(seconds * FPS)):
        out = fx.render(frame, person, plate, t0 + i / FPS)
        assert out.shape == frame.shape and out.dtype == np.uint8
    return out


@pytest.mark.parametrize("name", list(available()))
@pytest.mark.parametrize("origin", [None, (0.5, 0.4)], ids=["barrido", "chasquido"])
def test_full_cycle(name, origin, scene):
    frame, person, plate = scene
    fx = create(name, W, H, duration=DURATION)
    assert fx.state == "visible"
    assert fx.render(frame, person, plate, 0.0) is frame

    fx.trigger(0.0, person, origin=origin)
    assert fx.state == "vanishing" and fx.busy
    run(fx, frame, person, plate, 0.0, DURATION + 0.1)
    assert fx.state == "vanished"

    # Ya sin partículas en el aire, donde estaba la persona solo queda el fondo (más el grano).
    out = run(fx, frame, person, plate, 2.0, 3.0)
    inside = person > 0.5
    diff = np.abs(out.astype(int) - plate.bg.astype(int))[inside]
    assert diff.mean() < 6, f"{name}: queda rastro de la persona (diferencia media {diff.mean():.1f})"

    fx.trigger(5.0, person, origin=origin)
    assert fx.state == "appearing"
    run(fx, frame, person, plate, 5.0, DURATION + 0.1)
    assert fx.state == "visible"


@pytest.mark.parametrize("name", list(available()))
def test_trigger_ignored_while_busy(name, scene):
    frame, person, plate = scene
    fx = create(name, W, H, duration=DURATION)
    fx.trigger(0.0, person)
    fx.render(frame, person, plate, 0.3)
    fx.trigger(0.3, person)
    assert fx.state == "vanishing"


def test_take_over_keeps_vanished_state(scene):
    frame, person, plate = scene
    old = create("cyber", W, H, duration=DURATION)
    old.trigger(0.0, person)
    run(old, frame, person, plate, 0.0, DURATION + 0.1)

    new = create("polvo", W, H, duration=DURATION)
    new.take_over(old, 2.0)
    assert new.state == "vanished"
    new.trigger(2.0, person)
    assert new.state == "appearing"


def test_registry_has_unique_keys():
    effects = available()
    keys = [cls.key for cls in effects.values()]
    assert len(keys) == len(set(keys))
    assert all(len(k) == 1 for k in keys)
    assert all(cls.summary for cls in effects.values())
    assert list(effects)[0] == "cyber"


def test_unknown_effect():
    with pytest.raises(ValueError, match="Disponibles"):
        create("no-existe", W, H)


def test_register_rejects_duplicates():
    class Copy(Vanish):
        name = "cyber"
        key = "0"

    class SameKey(Vanish):
        name = "otro"
        key = "1"

    with pytest.raises(ValueError, match="Ya hay"):
        register(Copy)
    with pytest.raises(ValueError, match="tecla"):
        register(SameKey)


def test_homing_particles_land_on_target():
    fx = create("polvo", W, H)
    fx.GRAVITY = 0.0
    life = 0.5
    fx._emit_homing(np.array([0.0]), np.array([0.0]), np.array([100.0]), np.array([50.0]),
                    np.array([life]), np.array([2]), np.array([[255, 255, 255]]))
    out = np.zeros((H, W, 3), np.uint8)
    for _ in range(int(life * FPS) - 1):
        fx._particles(out, 1 / FPS)
    assert len(fx.particles) == 1
    assert abs(fx.particles[0, X] - 100) < 2 and abs(fx.particles[0, Y] - 50) < 2


def test_readme_lists_every_effect():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    for name, cls in available().items():
        assert f"| `{cls.key}` | `{name}` |" in readme, f"Falta {name} en la tabla de efectos del README"
