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

1. Pulsa **B**, levántate y sal del encuadre: en 5 s captura la habitación. Es lo que mejor queda. (Si no lo haces, el fondo se va aprendiendo solo mientras te mueves, y lo que nunca se llegó a ver se reconstruye copiando textura real de tu habitación; ver [Sin capturar el fondo](#sin-capturar-el-fondo).)
   - Todo lo que esté pegado a ti y no estuviera en ese fondo desaparece contigo: silla, sombra, lo que tengas en la mano.
   - **Silla giratoria**: captura el fondo *sin la silla* (apártala) y desaparecerás con silla y todo.
2. En Meet → Configuración → Vídeo, elige **Cyber Cam** / **OBS Virtual Camera**.
3. **Chasquea los dedos** con la mano visible: te desintegras desde la mano hacia afuera. Usa la cámara (pulgar y corazón juntos) y el micrófono (el sonido seco) a la vez. Repítelo para reaparecer.

### Efectos

| Tecla | `--effect` | Qué hace |
|---|---|---|
| `1` | `cyber` | holograma, bloques glitch de neón, onda de choque y HUD `SIGNAL_LOST` |
| `2` | `polvo` | te vuelves ceniza con filo de carbón y te deshaces en granos y trozos que se lleva un viento racheado |
| `3` | `fuego` | un frente de brasa con llamas que lamen hacia arriba, grietas al rojo, aire que tiembla, chispas y humo |
| `4` | `teleport` | un escáner te llena de energía, te vuelves luz, te comprimes en una línea, luego en un punto, y estallas en un destello |
| `5` | `derretir` | te derrites como cera: chorretones brillantes, gotas que caen y un charco que se evapora |
| `6` | `pixel` | te pixelas a saltos con paleta retro en bloques con relieve que se sueltan, giran y rebotan |
| `7` | `fantasma` | te vuelves un espectro translúcido que se separa en colores, ondula en jirones y deja ecos que suben |
| `8` | `matrix` | te digitalizas en verde y te conviertes en lluvia de código con cabezas blancas que dibuja tu cara |

#### Inspirados en anime

Efectos originales que homenajean desapariciones icónicas del anime (no usan personajes, logos ni textos de las series).

| Tecla | `--effect` | Qué hace | Inspirado en |
|---|---|---|---|
| `9` | `humo` | ¡puf! una nube de humo con sombreado de dibujo animado estalla con líneas de impacto; desapareces dentro y se disipa en bolitas | los clones y sustituciones ninja (*Naruto*) |
| `0` | `hojas` | un remolino de hojas con trazos de viento te envuelve cada vez más rápido, te tapa y el viento se lo lleva | el desplazamiento instantáneo en un torbellino de hojas (*Naruto*) |
| `c` | `cuervos` | te vuelves tinta y te deshaces en una bandada de cuervos que aletean, entre plumas que caen meciéndose | la dispersión en cuervos (*Naruto*) |
| `r` | `remolino` | un vórtice espacio-temporal se abre en tu cara, te retuerce y te absorbe hacia un punto negro | el teletransporte por vórtice (*Naruto*) |
| `v` | `velocidad` | parpadeas y sales disparado de lado, estirado y con líneas de velocidad, dejando imágenes residuales que parpadean | las imágenes residuales de quien se mueve demasiado rápido (*Dragon Ball*, *Bleach*) |
| `t` | `tiempo` | el tiempo se detiene: una esfera de colores invertidos se expande, la imagen se congela, desapareces y al reanudarse ya no estás | el tiempo detenido (*JoJo's Bizarre Adventure*) |

Al **reaparecer**, todo ocurre al revés: los granos de polvo, los bloques y los cuervos vuelan de vuelta y aterrizan en su sitio justo cuando esa parte de ti aparece, el punto de luz se abre en una línea y luego en tu cuerpo, el remolino te expulsa, llegas a toda velocidad... En `humo` y `tiempo` no es al revés sino como en el anime: apareces dentro de la nube, o con el tiempo detenido.

Pulsa la tecla en la ventana de vista previa para cambiar de efecto en vivo, o `E` para pasar al siguiente (no a mitad de una transición; si estás invisible, sigues invisible y reapareces con el nuevo).

### Teclas

| Tecla | Acción |
|---|---|
| `Espacio` | Activar/desactivar el efecto |
| `1`–`9`, `0`, `C`, `R`, `T`, `V` | Elegir efecto (ver tablas de arriba) |
| `E` | Siguiente efecto |
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
- **`SSL: CERTIFICATE_VERIFY_FAILED` en Mac** al descargar los modelos: ya no debería pasar (se usan los certificados de `certifi`; reinstala con `pip install -e .`). Con una versión antigua, ejecuta `open "/Applications/Python 3.12/Install Certificates.command"` (cambia 3.12 por tu versión).
- **¿En el celular?** No funciona en Android/iOS (ni con Termux): MediaPipe no tiene versión para Termux y el sistema no deja crear cámaras virtuales. Úsalo en el ordenador; si quieres usar el celular como cámara, conéctalo como webcam (DroidCam, Iriun o el modo webcam USB de Android 14+) y elígelo con `--camera N`.

## Sin capturar el fondo

Si no pulsas **B**, la app aprende la habitación con lo que va quedando a la vista mientras te mueves. Lo que tu cuerpo nunca dejó ver se reconstruye en `cybervanish/fill.py` así:

1. **Copiando textura real de tu habitación** (completado por ejemplos, Wexler et al. 2007): cada trozo del hueco busca, de grueso a fino, el trozo de habitación que mejor encaja con lo que tiene alrededor, y al final se copian los píxeles reales a resolución completa.
2. **En trozos grandes y contiguos** (búsqueda coherente, Ashikhmin 2001): cada trozo prefiere seguir copiando de donde copia su vecino, así la pared o la estantería se ven nítidas en vez de una media borrosa.
3. **Con el tono del entorno**: el brillo y el color generales salen de una interpolación suave desde el borde, para que no quede una silueta más clara u oscura que delate dónde estabas.

Antes se usaba `cv2.inpaint`, que estira los bordes hacia dentro y deja las rayas en abanico. El relleno nuevo tarda ~2 s, así que se calcula en segundo plano desde que abres la app (solo mientras se te ve, para no frenar los efectos) y se rehace cuando se ha visto más fondo; hasta que está listo, se usa el rápido.

Lo que nunca se vio no se puede adivinar: si detrás de ti hay un objeto que la cámara no llegó a ver, se rellenará con algo plausible de alrededor, no con el objeto. Para un resultado perfecto, captura con **B**; si no, muévete un poco (inclínate a un lado y al otro) antes de desaparecer para que vea más.

Para no aprender basura: no se aprende nada mientras la cámara arranca (los primeros frames suelen salir negros o subiendo de brillo, y en ellos MediaPipe no ve a nadie; antes eso dejaba un parche negro donde estabas), y un píxel solo cuenta como fondo tras verse libre en 3 actualizaciones seguidas (si MediaPipe te pierde un instante, no te graba en el fondo).

Para ver cómo queda con una foto de tu habitación, sin cámara:

```bash
python scripts/bench_background.py --image foto_de_tu_habitacion.jpg --out fondo.png
```

Se probó también LaMa (red neuronal de inpainting, vía `cv2.dnn` y onnxruntime): con huecos del tamaño de una persona (~25 % de la imagen) dibuja una sombra con forma de persona o un gris plano, tarda 3–4 s en CPU y pesa 208 MB, así que se descartó.

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
       key = "x"              # tecla en vivo: número o letra minúscula, única (B, Q, E y espacio están reservadas)
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
| `self._arriving(d, margin, lead)` / `self._emit_homing(...)` | al reaparecer, partículas que vuelan a su sitio y aterrizan justo cuando ese píxel aparece |
| `self._splat(img, p)` | dibujar miles de partículas pequeñas de golpe |
| `glow`, `band`, `below`, `smoothstep`, `gray3` | utilidades de `base.py` para halos, bandas y mezclas |
| `body_box(person, w, h)` | centro y límites de tu silueta, para efectos que giran o salen de ti |
| `GRAVITY`, `DRAG`, `MAX_PARTICLES`, `_forces()` | física de las partículas (viento, flotación, rebotes...) |
| `GRID`, `MIX`, `BLOB_SIGMA` | forma del frente (recorrido / manchas / ruido) |
| `_plan()`, `_draw_particles()`, `_vanished()`, `_finish()` | sobrescríbelos si necesitas algo distinto |

Un frame a 1280x720 debería costar menos de ~25 ms (los efectos de serie van de 6 a 25 ms en un portátil normal). Trabaja a 1/4 de resolución lo que vaya a quedar difuminado (llamas, humo, halos).

## Desarrollo

```bash
pip install -e ".[dev]"
pytest                               # tests de los efectos (sin cámara)
ruff check .                         # estilo y errores comunes
python scripts/bench_effects.py      # ms por frame de cada efecto
python scripts/bench_background.py   # cómo queda el fondo sin capturarlo con B
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
  fill.py                 reconstrucción de lo que nunca se vio (copiando textura real)
  effects/
    base.py               clase Vanish y registro de efectos
    cyber.py, dust.py,    un archivo por efecto
    fire.py, ...
tests/                    tests de los efectos y del relleno del fondo
scripts/bench_effects.py  rendimiento y capturas de los efectos
scripts/bench_background.py  el fondo sin capturar, con una foto tuya
```
