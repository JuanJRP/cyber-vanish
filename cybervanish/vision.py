"""Detección de silueta y manos con MediaPipe Tasks."""

import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_URLS = {
    "selfie_segmenter.tflite": "https://storage.googleapis.com/mediapipe-models/"
    "image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
}


def ensure_model(name):
    path = MODELS_DIR / name
    if not path.exists():
        MODELS_DIR.mkdir(exist_ok=True)
        print(f"Descargando modelo {name}...")
        tmp = path.with_name(name + ".part")
        urllib.request.urlretrieve(MODEL_URLS[name], tmp)
        tmp.replace(path)
    return str(path)


class Tracker:
    def __init__(self, width, height, proc_width=640):
        self.size = (width, height)
        self.proc_size = (proc_width, round(height * proc_width / width))
        self.segmenter = vision.ImageSegmenter.create_from_options(
            vision.ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=ensure_model("selfie_segmenter.tflite")),
                running_mode=vision.RunningMode.VIDEO,
                output_confidence_masks=True,
            )
        )
        self.hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=ensure_model("hand_landmarker.task")),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                # Umbrales bajos: con el movimiento rápido la mano sale borrosa.
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.4,
                min_tracking_confidence=0.4,
            )
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        self.prev_mask = None
        self.hand_pool = ThreadPoolExecutor(max_workers=1)
        self.last_ts = -1

    def process(self, frame, now):
        """Devuelve (máscara de persona HxW float32 0..1, manos: lista de arrays (21, 2) normalizados)."""
        # MediaPipe exige timestamps estrictamente crecientes.
        ts = max(self.last_ts + 1, int(now * 1000))
        self.last_ts = ts

        small = cv2.resize(frame, self.proc_size, interpolation=cv2.INTER_AREA)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        # Manos y silueta en paralelo: cada modelo va por su lado.
        hands_job = self.hand_pool.submit(self.hands.detect_for_video, image, ts)
        seg = self.segmenter.segment_for_video(image, ts)
        hands = [
            np.array([(lm.x, lm.y) for lm in hand], np.float32) for hand in hands_job.result().hand_landmarks
        ]

        mask = seg.confidence_masks[0].numpy_view().squeeze().astype(np.float32)
        if self.prev_mask is not None:
            mask = 0.6 * mask + 0.4 * self.prev_mask
        self.prev_mask = mask
        # Para desaparecer del todo: cualquier zona con confianza media (pelo, bordes, manos)
        # cuenta como persona completa, y se agranda la silueta para que no quede halo.
        mask = np.clip((mask - 0.1) / 0.35, 0, 1)
        mask = np.clip(cv2.GaussianBlur(cv2.dilate(mask, self.kernel), (0, 0), 2.5), 0, 1)
        return cv2.resize(mask, self.size, interpolation=cv2.INTER_LINEAR), hands
