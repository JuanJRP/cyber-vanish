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


def learn(plate, frame, person, n=12, now=0.0):
    """Varias actualizaciones con el mismo frame: lo que tarda en arrancar la cámara y en fiarse de un píxel."""
    for i in range(n):
        plate.update(frame, person, now + i / 15)


def wait_fill(plate):
    """Espera al relleno en segundo plano (si sigue en marcha)."""
    if plate.job is not None:
        plate.job.result(timeout=60)


def test_plate_uses_good_fill_once_ready(plate):
    room, hole = textured_room(), person_hole()
    frame = room.copy()
    frame[hole] = (30, 30, 200)                    # la persona
    person = hole.astype(np.float32)

    learn(plate, frame, person)
    assert plate.job is not None or plate.filled is not None   # el relleno bueno arranca solo
    assert plate.get(1.0).shape == room.shape                  # mientras tanto, el rápido

    wait_fill(plate)
    bg = plate.get(1.1)
    assert plate.filled is not None
    unseen = ~plate.seen
    assert unseen.any()
    assert np.array_equal(bg[plate.seen], plate.plate[plate.seen])
    assert not np.any(np.all(bg[unseen] == (30, 30, 200), axis=1)), "se coló la persona en el fondo"


def test_capture_discards_old_fill(plate):
    room, hole = textured_room(), person_hole()
    person = hole.astype(np.float32)
    learn(plate, room, person)
    plate.capture(room, np.zeros((H, W), np.float32))
    wait_fill(plate)
    plate.get(1.0)
    assert plate.filled is None                     # con todo visto ya no hace falta relleno
    assert plate.seen.all()


def test_capture_is_the_photo_as_is(plate):
    """Lo capturado con B es la foto tal cual: aunque MediaPipe marque algo (la silla vacía), no hay
    huecos que rellenar ni restos del fondo aprendido antes. Al desaparecer se ve esa foto."""
    room, hole = textured_room(), person_hole()
    sitting = room.copy()
    sitting[hole] = (30, 30, 200)
    learn(plate, sitting, np.zeros((H, W), np.float32))   # el fondo aprendido te tiene dentro
    plate.capture(room, hole.astype(np.float32) * 0.6)
    assert plate.seen.all()
    assert np.array_equal(plate.get(1.0), room)


def test_dark_camera_start_is_not_learned(plate):
    """Al arrancar, la webcam da frames negros en los que MediaPipe no ve a nadie: no deben quedar en el fondo."""
    room, hole = textured_room(), person_hole()
    frame = room.copy()
    frame[hole] = (30, 30, 200)
    nobody = np.zeros((H, W), np.float32)
    for i in range(10):
        plate.update(np.zeros_like(room), nobody, i / 15)      # cámara aún a oscuras
    for i in range(6):                                         # la exposición va subiendo
        plate.update(cv2.convertScaleAbs(frame, alpha=(i + 1) / 6), nobody, (10 + i) / 15)
    assert not plate.seen.any()
    learn(plate, frame, hole.astype(np.float32), n=16, now=2.0)
    assert not plate.seen[hole].any(), "se aprendió como fondo la zona donde estabas"
    wait_fill(plate)
    bg = plate.get(4.0)
    assert bg[hole].max(axis=1).mean() > 60, "queda un parche negro"


def test_one_frame_mask_dropout_is_not_learned(plate):
    """Si MediaPipe te pierde durante un frame, no te graba en el fondo."""
    room, hole = textured_room(), person_hole()
    frame = room.copy()
    frame[hole] = (30, 30, 200)
    person = hole.astype(np.float32)
    learn(plate, frame, person, n=12)
    plate.update(frame, np.zeros((H, W), np.float32), 1.0)      # un frame sin persona detectada
    plate.update(frame, person, 1.1)
    assert not plate.seen[hole].any()
