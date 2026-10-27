"""Relleno de lo que tapa la persona cuando no se capturó el fondo."""

import cv2
import numpy as np
import pytest

from cybervanish.background import BackgroundPlate
from cybervanish.fill import complete

W, H = 320, 180


def textured_room(seed=0):
    """Pared con rayas verticales y grano (estructura + textura): lo que peor lleva cv2.inpaint."""
    rng = np.random.default_rng(seed)
    x = np.arange(W)
    stripes = (np.sin(x / 5.0) > 0).astype(np.float32)[None, :, None]
    img = 90 + 80 * stripes * np.array([1.0, 0.8, 0.6]) + rng.normal(0, 10, (H, W, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def person_hole():
    hole = np.zeros((H, W), np.uint8)
    cv2.ellipse(hole, (W // 2, 70), (28, 36), 0, 0, 360, 1, -1)
    cv2.ellipse(hole, (W // 2, 200), (80, 90), 0, 180, 360, 1, -1)
    return hole > 0


def psnr(a, b, mask):
    d = (a.astype(np.float32) - b.astype(np.float32))[mask]
    return 10 * np.log10(255**2 / max(float(np.mean(d**2)), 1e-6))


def test_fill_keeps_known_pixels_and_fills_hole():
    room, hole = textured_room(), person_hole()
    plate = room.copy()
    plate[hole] = 0
    out = complete(plate, hole)
    assert out.shape == room.shape and out.dtype == np.uint8
    assert np.array_equal(out[~hole], plate[~hole])
    assert out[hole].mean() > 60   # nada de negro: el hueco se rellenó


def test_fill_beats_smear_on_texture():
    """Copiando textura real, las rayas siguen; cv2.inpaint las difumina en manchas."""
    room, hole = textured_room(), person_hole()
    plate = room.copy()
    plate[hole] = 0
    ours = complete(plate, hole)
    smear = cv2.inpaint(plate, hole.astype(np.uint8), 5, cv2.INPAINT_TELEA)

    def detail(img):
        return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_32F)[hole].var()

    assert detail(ours) > 0.25 * detail(room)      # conserva buena parte de la textura...
    assert detail(ours) > 10 * detail(smear)       # ...que cv2.inpaint borra casi entera
    assert psnr(ours, room, hole) > psnr(smear, room, hole) - 1.0


def test_fill_edge_cases():
    room = textured_room()
    assert np.array_equal(complete(room, np.zeros((H, W), bool)), room)
    everything = np.ones((H, W), bool)
    everything[:3, :3] = False
    out = complete(room, everything)   # casi sin nada de lo que copiar: no debe fallar
    assert out.shape == room.shape


@pytest.fixture
def plate():
    p = BackgroundPlate(W, H)
    yield p
    p.close()


def test_plate_uses_good_fill_once_ready(plate):
    room, hole = textured_room(), person_hole()
    frame = room.copy()
    frame[hole] = (30, 30, 200)                    # la persona
    person = hole.astype(np.float32)

    plate.update(frame, person, now=0.0)
    assert plate.job is not None                    # el relleno bueno arranca en segundo plano
    early = plate.get(0.0)                          # mientras tanto, el rápido
    assert early.shape == room.shape

    plate.job.result(timeout=60)
    bg = plate.get(0.1)
    unseen = ~plate.seen
    assert unseen.any()
    assert np.array_equal(bg[plate.seen], plate.plate[plate.seen])
    assert not np.any(np.all(bg[unseen] == (30, 30, 200), axis=1)), "se coló la persona en el fondo"


def test_capture_discards_old_fill(plate):
    room, hole = textured_room(), person_hole()
    person = hole.astype(np.float32)
    plate.update(room, person, now=0.0)
    old_job = plate.job
    plate.capture(room, np.zeros((H, W), np.float32))
    old_job.result(timeout=60)
    plate.get(0.1)
    assert plate.filled is None                     # con todo visto ya no hace falta relleno
    assert plate.seen.all()
