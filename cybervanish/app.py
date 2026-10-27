"""Cyber Vanish: desaparece de tu videollamada chasqueando los dedos, estilo cyberpunk.

La salida va a una cámara virtual que Meet / Zoom / Teams / Discord ven como una webcam.
"""

import argparse
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2

from .audio import SnapListener
from .background import BackgroundPlate
from .effects import available, create
from .gesture import SnapDetector
from .vision import Tracker

SYSTEM = platform.system()
VCAM_HELP = {
    "Linux": 'sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="Cyber Cam" exclusive_caps=1',
    "Windows": "Instala OBS Studio (https://obsproject.com): pyvirtualcam usa su cámara virtual.",
    "Darwin": "Instala OBS Studio, ábrelo una vez y pulsa 'Iniciar cámara virtual' para registrarla.",
}
CAPTURE_DELAY = 5.0


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--effect", default="cyber", choices=list(available()),
                    help="efecto de desaparición (cámbialo en vivo con su tecla o con E)")
    ap.add_argument("--list-effects", action="store_true", help="lista los efectos y sus teclas, y sale")
    ap.add_argument("--camera", type=int, default=0, help="índice de la webcam real")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--device", help="dispositivo de cámara virtual (p. ej. /dev/video10 en Linux)")
    ap.add_argument("--duration", type=float, default=3.0, help="segundos que tarda en desaparecer")
    ap.add_argument("--mic", help="micrófono para oír el chasquido (nombre o índice)")
    ap.add_argument("--snap-sensitivity", type=float, default=1.0, help="más alto = chasquidos más suaves")
    ap.add_argument("--no-mic", action="store_true", help="detectar el chasquido solo con la cámara")
    ap.add_argument("--no-virtual", action="store_true", help="solo vista previa, sin cámara virtual")
    ap.add_argument("--no-preview", action="store_true", help="sin ventana de vista previa")
    ap.add_argument("--no-hud", action="store_true", help="sin texto SIGNAL_LOST durante el efecto")
    ap.add_argument("--debug", action="store_true", help="muestra lo que oye y ve el detector de chasquido")
    ap.add_argument("--ghost", action="store_true", help="parpadeos holográficos de vez en cuando estando invisible")
    ap.add_argument("--no-mirror", action="store_true", help="no reflejar la vista previa")
    return ap.parse_args()


def open_camera(args):
    backend = {"Windows": cv2.CAP_DSHOW, "Darwin": cv2.CAP_AVFOUNDATION}.get(SYSTEM, cv2.CAP_ANY)
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        sys.exit(f"No se pudo abrir la cámara {args.camera}. Prueba con --camera 1")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    ok, frame = cap.read()
    if not ok:
        sys.exit("La cámara no devuelve imagen.")
    return cap, frame.shape[1], frame.shape[0]


def open_virtual_cam(width, height, fps, device):
    try:
        import pyvirtualcam

        cam = pyvirtualcam.Camera(width, height, fps, fmt=pyvirtualcam.PixelFormat.BGR, device=device)
    except Exception as e:  # pyvirtualcam lanza distintos errores según el backend
        print(f"\n[!] No se pudo abrir la cámara virtual: {e}")
        print(f"    {VCAM_HELP.get(SYSTEM, '')}")
        print("    Sigo solo con la vista previa.\n")
        return None
    print(f"Cámara virtual activa: {cam.device}. Elígela como cámara en Meet.")
    return cam


def draw_status(view, fx, plate, fps, capture_at, now):
    h = view.shape[0]
    status = f"{fx.name.upper()} [{fx.key}]  |  {fx.state.upper()}  |  fondo {plate.coverage:.0%}  |  {fps:.0f} fps"
    cv2.putText(view, status, (12, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
    keys = "ESPACIO: activar  1-0 C R T V: efecto  E: siguiente  B: capturar fondo  Q: salir"
    cv2.putText(view, keys, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    if capture_at is not None:
        n = max(1, int(capture_at - now + 1))
        cv2.putText(view, "LEVANTATE Y SAL DEL ENCUADRE", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                    (255, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(view, str(n), (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 255, 0), 6, cv2.LINE_AA)
    elif plate.coverage < 0.97:
        cv2.putText(view, "Pulsa B, levantate y sal del encuadre para capturar el fondo", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)


def debug_snap(mic, snap, hands, now):
    """Imprime cada sonido candidato junto con el estado de la mano en ese momento."""
    while mic and mic.events:
        at, ok, ratio, centroid, echo = mic.events.popleft()
        pose = "sin mano" if snap.metrics is None else f"gap {snap.metrics[0]:.2f} reach {snap.metrics[1]:.2f}"
        posed = "sí" if now - snap.pinch_at < snap.window else "no"
        print(f"[sonido] {'CHASQUIDO' if ok else 'descartado':10s} agudos/graves {ratio:6.1f}  "
              f"centroide {centroid:5.0f} Hz  eco {echo:.2f}  | manos {len(hands)} {pose}  postura {posed}")


def print_effects():
    for name, cls in available().items():
        print(f"  [{cls.key}] {name:10s} {cls.summary}")


def switch_effect(fx, name, width, height, args, now):
    """Cambia de efecto en vivo. No se puede a mitad de una transición."""
    if fx.busy or name == fx.name:
        return fx
    new = create(name, width, height, duration=args.duration, hud=not args.no_hud, ghost=args.ghost)
    new.take_over(fx, now)
    print(f"Efecto: {name}")
    return new


def main():
    args = parse_args()
    if args.list_effects:
        print_effects()
        return
    cap, width, height = open_camera(args)
    print(f"Cámara real: {width}x{height}")

    tracker = Tracker(width, height)
    plate = BackgroundPlate(width, height)
    mic = None if args.no_mic else SnapListener(args.mic, args.snap_sensitivity)
    snap = SnapDetector(use_audio=bool(mic and mic.active))
    fx = create(args.effect, width, height, duration=args.duration, hud=not args.no_hud, ghost=args.ghost)
    effect_keys = {ord(cls.key): name for name, cls in available().items()}
    print("Efectos (pulsa su tecla para cambiar):")
    print_effects()
    vcam = None if args.no_virtual else open_virtual_cam(width, height, args.fps, args.device)

    # Pipeline: mientras se dibuja un frame, el tracker ya analiza el siguiente
    # (MediaPipe y OpenCV sueltan el GIL, así que el hilo sí corre en paralelo).
    pool = ThreadPoolExecutor(max_workers=1)
    pending = None
    capture_at = None
    frame_i = 0
    fps, fps_t, fps_n = 0.0, time.monotonic(), 0
    try:
        while True:
            ok, new_frame = cap.read()
            if not ok:
                print("Se perdió la señal de la cámara.")
                break
            t_new = time.monotonic()
            job = (new_frame, t_new, pool.submit(tracker.process, new_frame, t_new))
            if pending is None:
                pending = job
                continue
            frame, now, future = pending
            person, hands = future.result()
            pending = job

            if capture_at is not None and now >= capture_at:
                coverage = plate.capture(frame, person)
                capture_at = None
                print(f"Fondo capturado ({coverage:.0%}).")
                if coverage < 0.97:
                    print("  Seguías en cuadro: esa zona se completará cuando te muevas. "
                          "Pulsa B otra vez si queda mal.")
            elif capture_at is None and frame_i % 2 == 0:
                # El relleno de lo no visto se prepara solo cuando sobra CPU: mientras se te ve.
                plate.update(frame, person, now if fx.state == "visible" else None)

            # Lo que desaparece: tú + lo pegado a ti que no estaba en el fondo (la silla, tu sombra...).
            body = plate.attached(frame, person)

            snap_at = snap.update(hands, now, mic.sound_at if mic else None)
            if args.debug:
                debug_snap(mic, snap, hands, now)
            if snap_at is not None:
                print("Chasquido")
                fx.trigger(now, body, origin=snap_at)

            out = fx.render(frame, body, plate, now)
            if vcam:
                vcam.send(out)

            fps_n += 1
            if now - fps_t >= 1:
                fps, fps_t, fps_n = fps_n / (now - fps_t), now, 0
            frame_i += 1

            if args.no_preview:
                continue
            view = out if args.no_mirror else cv2.flip(out, 1)
            draw_status(view, fx, plate, fps, capture_at, now)
            cv2.imshow("Cyber Vanish", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                fx.trigger(now, body)
            elif key == ord("b"):
                capture_at = now + CAPTURE_DELAY
            elif key in effect_keys:
                fx = switch_effect(fx, effect_keys[key], width, height, args, now)
            elif key == ord("e"):
                names = list(available())
                fx = switch_effect(fx, names[(names.index(fx.name) + 1) % len(names)], width, height, args, now)
    except KeyboardInterrupt:
        pass
    finally:
        pool.shutdown(cancel_futures=True)
        plate.close()
        if mic:
            mic.close()
        cap.release()
        if vcam:
            vcam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
