# Cyber Vanish

Desaparece de tu videollamada barriendo la mano o chasqueando los dedos: te desintegras poco a poco (3 s) en bloques digitales con brillo de neón, ceniza de píxeles, aberración cromática y HUD `SIGNAL_LOST`, hasta no quedar nada. Repite el gesto y vuelves a aparecer.

La salida es una **cámara virtual**, así que funciona en Meet, Zoom, Teams, Discord, OBS...

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

En Linux, el micrófono necesita PortAudio: `sudo apt install libportaudio2`.

Los modelos de MediaPipe se descargan solos la primera vez (carpeta `models/`).

### Cámara virtual según sistema

| Sistema | Qué hace falta |
|---|---|
| **Linux** | `sudo apt install v4l2loopback-dkms` y luego `sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="Cyber Cam" exclusive_caps=1` (`exclusive_caps=1` es necesario para que Chrome la vea). |
| **Windows** | Instalar [OBS Studio](https://obsproject.com). No hace falta tenerlo abierto. |
| **macOS** | Instalar OBS Studio, abrirlo una vez y pulsar **Iniciar cámara virtual** para que macOS la registre. Da permiso de cámara a tu terminal. |

## Uso

```bash
python main.py
```

1. Pulsa **B**, levántate y sal del encuadre: en 5 s captura la habitación. (Si no lo haces, el fondo se va aprendiendo solo mientras te mueves, pero los huecos se rellenan peor.)
   - Todo lo que esté pegado a ti y no estuviera en ese fondo desaparece contigo: silla, sombra, lo que tengas en la mano.
   - **Silla giratoria**: captura el fondo *sin la silla* (apártala) y desaparecerás con silla y todo.
2. En Meet → Configuración → Vídeo, elige **Cyber Cam** / **OBS Virtual Camera**.
3. Gestos (repítelo para reaparecer):
   - **Barrer la mano** de un lado a otro: te desintegras empezando por el lado desde el que barres.
   - **Chasquear los dedos** con la mano visible: te desintegras desde la mano hacia afuera. Usa la cámara (pulgar y corazón juntos) y el micrófono (el sonido seco) a la vez.

| Tecla | Acción |
|---|---|
| `Espacio` | Activar/desactivar el efecto |
| `B` | Capturar el fondo (cuenta atrás de 3 s) |
| `Q` / `Esc` | Salir |

### Opciones

```
--camera N       índice de la webcam real (por defecto 0)
--width/--height resolución (1280x720 por defecto; baja a 960x540 si va lento)
--device PATH    dispositivo de cámara virtual (p. ej. /dev/video10)
--duration S     segundos que tarda en desaparecer (3.0)
--mic NOMBRE     micrófono para oír el chasquido
--snap-sensitivity X  más alto = acepta chasquidos más suaves (1.0)
--no-mic         chasquido solo con la cámara
--no-hud         sin el texto SIGNAL_LOST
--no-ghost       sin parpadeos holográficos mientras eres invisible
--no-virtual     solo vista previa
--no-preview     sin ventana
--no-mirror      vista previa sin reflejar
```

## Consejos

- Cámara fija y luz estable: si la cámara se mueve, el fondo capturado deja de encajar.
- Si al desaparecer queda una sombra, vuelve a pulsar **B**.
- La vista previa sale reflejada como un espejo (el HUD se lee al revés ahí); lo que reciben los demás va sin reflejar.

## Archivos

- `main.py`: cámara, cámara virtual, bucle y teclado.
- `vision.py`: silueta y manos con MediaPipe.
- `gesture.py`: detectores de barrido y chasquido.
- `audio.py`: escucha el micrófono para el chasquido.
- `background.py`: fondo de la habitación vacía.
- `fx.py`: el efecto cyberpunk.
