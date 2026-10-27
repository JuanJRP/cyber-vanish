# Cyber Vanish

Desaparece de tu videollamada chasqueando los dedos: te desintegras poco a poco (3 s) con el efecto que elijas —bloques de neón, polvo, fuego, teletransporte, cera derretida, píxeles, fantasma o lluvia de código— hasta no quedar nada. Repite el gesto y vuelves a aparecer.

La salida es una **cámara virtual**, así que funciona en Meet, Zoom, Teams, Discord, OBS...

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

En Linux, el micrófono necesita PortAudio: `sudo apt install libportaudio2`. En macOS: `brew install portaudio`.

Los modelos de MediaPipe se descargan solos la primera vez (carpeta `cybervanish/models/`).

### Cámara virtual según sistema

| Sistema | Qué hace falta |
|---|---|
| **Linux** | `sudo apt install v4l2loopback-dkms` y luego `sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="Cyber Cam" exclusive_caps=1` (`exclusive_caps=1` es necesario para que Chrome la vea). |
| **Windows** | Instalar [OBS Studio](https://obsproject.com). No hace falta tenerlo abierto. |
| **macOS** | `brew install --cask obs`, abrirlo una vez y pulsar **Iniciar cámara virtual** para que macOS la registre. Da permiso de cámara y micrófono a tu terminal. |

## Uso

```bash
cybervanish                      # o: python main.py  /  python -m cybervanish
cybervanish --effect fuego       # empieza con otro efecto
cybervanish --list-effects       # lista los efectos y sus teclas
```

1. Pulsa **B**, levántate y sal del encuadre: en 5 s captura la habitación. (Si no lo haces, el fondo se va aprendiendo solo mientras te mueves, pero los huecos se rellenan peor.)
   - Todo lo que esté pegado a ti y no estuviera en ese fondo desaparece contigo: silla, sombra, lo que tengas en la mano.
   - **Silla giratoria**: captura el fondo *sin la silla* (apártala) y desaparecerás con silla y todo.
2. En Meet → Configuración → Vídeo, elige **Cyber Cam** / **OBS Virtual Camera**.
3. **Chasquea los dedos** con la mano visible: te desintegras desde la mano hacia afuera. Usa la cámara (pulgar y corazón juntos) y el micrófono (el sonido seco) a la vez. Repítelo para reaparecer.

### Efectos

| Tecla | `--effect` | Qué hace |
|---|---|---|
| `1` | `cyber` | holograma, bloques glitch de neón, onda de choque y HUD `SIGNAL_LOST` |
| `2` | `polvo` | te vuelves ceniza y te deshaces en polvo que se lleva el viento |
| `3` | `fuego` | un borde de brasa te consume dejando chispas que suben |
| `4` | `teleport` | te llenas de luz y destellos y te vas por un haz de energía |
| `5` | `derretir` | te derrites y escurres hacia abajo como cera |
| `6` | `pixel` | te pixelas en bloques cada vez más grandes que se sueltan y caen |
| `7` | `fantasma` | te vuelves translúcido y ondulas como un espejismo hasta desvanecerte |
| `8` | `matrix` | te conviertes en lluvia de código verde que se apaga |

Pulsa la tecla en la ventana de vista previa para cambiar de efecto en vivo (no a mitad de una transición; si estás invisible, sigues invisible y reapareces con el nuevo).

### Teclas

| Tecla | Acción |
|---|---|
| `Espacio` | Activar/desactivar el efecto |
| `1`–`8` | Elegir efecto |
| `B` | Capturar el fondo (cuenta atrás de 5 s) |
| `Q` / `Esc` | Salir |

### Opciones

```
--effect NOMBRE  efecto inicial (cyber por defecto)
--list-effects   lista los efectos y sale
--camera N       índice de la webcam real (por defecto 0)
--width/--height resolución (1280x720 por defecto; baja a 960x540 si va lento)
--device PATH    dispositivo de cámara virtual (p. ej. /dev/video10)
--duration S     segundos que tarda en desaparecer (3.0)
--mic NOMBRE     micrófono para oír el chasquido
--snap-sensitivity X  más alto = acepta chasquidos más suaves (1.0)
--no-mic         chasquido solo con la cámara
--no-hud         sin el texto SIGNAL_LOST (efecto cyber)
--ghost          parpadeos holográficos de vez en cuando estando invisible (efecto cyber)
--no-virtual     solo vista previa
--no-preview     sin ventana
--no-mirror      vista previa sin reflejar
--debug          muestra lo que oye y ve el detector de chasquido
```

## Consejos

- Cámara fija y luz estable: si la cámara se mueve, el fondo capturado deja de encajar.
- Si al desaparecer queda una sombra, vuelve a pulsar **B**.
- La vista previa sale reflejada como un espejo (el HUD se lee al revés ahí); lo que reciben los demás va sin reflejar.
- **¿En el celular?** No funciona en Android/iOS (ni con Termux): MediaPipe no tiene versión para Termux y el sistema no deja crear cámaras virtuales. Úsalo en el ordenador; si quieres usar el celular como cámara, conéctalo como webcam (DroidCam, Iriun o el modo webcam USB de Android 14+) y elígelo con `--camera N`.

## Crear tu propio efecto

Cada efecto es un archivo en `cybervanish/effects/` con una clase que hereda de `Vanish`. La base ya se encarga de los estados (visible → desapareciendo → invisible → apareciendo), del fondo, del grano de cámara y de las partículas; tú solo dibujas un frame de la transición.

1. Crea `cybervanish/effects/mi_efecto.py`:

   ```python
   import cv2
   import numpy as np

   from .base import Vanish, below, register


   @register
   class MiEfecto(Vanish):
       name = "mi-efecto"     # para --effect
       key = "9"              # tecla en vivo (única)
       summary = "lo que verá --list-effects"

       def _transition(self, frame, person, bg, now, t, p, vanishing):
           # p va de 0 a 1 (también al aparecer). d > 0: ese píxel aún no se ha ido.
           d = self._front(p, vanishing)
           gone = below(d, 0)
           out = self._background(frame, bg, person, gone, p, vanishing)
           # ...tu efecto encima de `out`...
           return out
   ```

2. Añádelo a la lista de imports de `cybervanish/effects/__init__.py`.
3. `pytest` lo prueba automáticamente (ciclo completo, sin rastro al final) y `python scripts/bench_effects.py --sheet capturas.png` mide su coste y guarda capturas.

Lo que te da la base:

| Pieza | Para qué |
|---|---|
| `self._front(p, vanishing, margin)` | distancia de cada píxel al frente que avanza desde la mano (o de lado) |
| `self._background(...)` | mezcla correcta de la imagen en vivo con el fondo guardado |
| `self._sample(mask, n)` / `self._emit(...)` | elegir píxeles y lanzar partículas desde ahí |
| `GRAVITY`, `DRAG`, `MAX_PARTICLES` | física de las partículas |
| `GRID`, `MIX`, `BLOB_SIGMA` | forma del frente (recorrido / manchas / ruido) |
| `_plan()`, `_draw_particles()`, `_vanished()`, `_finish()` | sobrescríbelos si necesitas algo distinto |

Un frame a 1280x720 debería costar menos de ~20 ms (los efectos de serie van de 9 a 25 ms).

## Desarrollo

```bash
pip install -e ".[dev]"
pytest                               # tests de los efectos (sin cámara)
ruff check .                         # estilo y errores comunes
python scripts/bench_effects.py      # ms por frame de cada efecto
```

La CI de GitHub ejecuta `ruff` y `pytest` en cada push y pull request.

## Estructura

```
main.py                   atajo para `python main.py`
cybervanish/
  app.py                  cámara, cámara virtual, bucle, teclado y opciones
  vision.py               silueta y manos con MediaPipe
  gesture.py              detector de chasquido
  audio.py                escucha el micrófono para el chasquido
  background.py           fondo de la habitación vacía
  effects/
    base.py               clase Vanish y registro de efectos
    cyber.py, dust.py,    un archivo por efecto
    fire.py, ...
tests/                    tests de los efectos
scripts/bench_effects.py  rendimiento y capturas
```
