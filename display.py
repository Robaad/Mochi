# ─── display.py ───────────────────────────────────────────────────────────────
# Gestión del OLED 128×64. Todas las operaciones van a una cola interna
# para que nunca bloqueen el hilo principal.

import time
import threading
import queue
from luma.core.interface.serial import i2c
from luma.oled.device import sh1106 as OledDriver
from PIL import Image, ImageDraw
from config import OLED_WIDTH, OLED_HEIGHT, I2C_ADDRESS

# ── Inicialización hardware ────────────────────────────────────────────────────
_serial = i2c(port=1, address=I2C_ADDRESS)
_device = OledDriver(_serial, width=OLED_WIDTH, height=OLED_HEIGHT)

W, H = OLED_WIDTH, OLED_HEIGHT
CX   = W // 2   # centro X = 64
CY   = H // 2   # centro Y = 32

# ── Cola de comandos al display ────────────────────────────────────────────────
# Evita que dos hilos intenten escribir al OLED a la vez.
_q: "queue.Queue[str | None]" = queue.Queue(maxsize=4)
_talking = threading.Event()   # True mientras Mochi está hablando
_current_emotion = "happy"

# ── Definición de caras ───────────────────────────────────────────────────────
# Cada cara es una función que recibe un ImageDraw y dibuja.

def _eyes_normal(draw):
    """Ojos circulares estándar."""
    draw.ellipse([38, 18, 52, 32], fill=1)
    draw.ellipse([76, 18, 90, 32], fill=1)

def _eyes_wide(draw):
    """Ojos muy abiertos — sorpresa."""
    draw.ellipse([34, 14, 54, 34], fill=1)
    draw.ellipse([74, 14, 94, 34], fill=1)

def _eyes_half(draw):
    """Medio cerrados — cansancio/somnolencia."""
    draw.rectangle([38, 24, 52, 32], fill=1)
    draw.rectangle([76, 24, 90, 32], fill=1)

def _eyes_closed(draw):
    """Ojos cerrados — dormido."""
    draw.line([38, 28, 52, 28], fill=1, width=2)
    draw.line([76, 28, 90, 28], fill=1, width=2)

def _eyes_squint(draw):
    """Entrecerrando — sospecha / broma."""
    draw.rectangle([38, 22, 52, 28], fill=1)
    draw.rectangle([76, 22, 90, 28], fill=1)

def _eyes_love(draw):
    """Corazones (simulados con X rotada)."""
    # Corazón izquierdo — dos rectángulos superpuestos
    for dx, dy in [(-2,-4),(2,-4),(0,-2)]:
        draw.rectangle([44+dx-2, 22+dy-2, 44+dx+2, 22+dy+2], fill=1)
    for dx, dy in [(-2,-4),(2,-4),(0,-2)]:
        draw.rectangle([82+dx-2, 22+dy-2, 82+dx+2, 22+dy+2], fill=1)

def _mouth_smile(draw):
    draw.arc([34, 32, 90, 56], start=0, end=180, fill=1, width=3)

def _mouth_big_smile(draw):
    """Sonrisa grande — felicidad máxima."""
    draw.arc([28, 28, 96, 58], start=0, end=180, fill=1, width=3)
    # Comisuras hacia arriba
    draw.line([28, 43, 32, 38], fill=1, width=2)
    draw.line([96, 43, 92, 38], fill=1, width=2)

def _mouth_sad(draw):
    draw.arc([34, 40, 90, 64], start=180, end=360, fill=1, width=3)

def _mouth_open_small(draw):
    """Boca abierta pequeña — hablar."""
    draw.ellipse([50, 40, 78, 56], outline=1, width=2)

def _mouth_open_big(draw):
    """Boca abierta grande — hablar emocionado."""
    draw.ellipse([44, 38, 84, 58], outline=1, width=2)

def _mouth_straight(draw):
    draw.line([42, 48, 86, 48], fill=1, width=2)

def _mouth_surprised_o(draw):
    draw.ellipse([52, 38, 74, 58], outline=1, width=2)

def _mouth_smirk(draw):
    """Sonrisa torcida — broma."""
    draw.line([42, 48, 62, 44], fill=1, width=2)
    draw.arc([52, 40, 86, 54], start=320, end=40, fill=1, width=2)

# ── Catálogo de emociones ──────────────────────────────────────────────────────
EMOTIONS = {
    "happy":     (_eyes_normal,  _mouth_smile),
    "very_happy":(_eyes_normal,  _mouth_big_smile),
    "sad":       (_eyes_normal,  _mouth_sad),
    "surprised": (_eyes_wide,    _mouth_surprised_o),
    "thinking":  (_eyes_squint,  _mouth_straight),
    "sleepy":    (_eyes_half,    _mouth_straight),
    "sleeping":  (_eyes_closed,  _mouth_straight),
    "love":      (_eyes_love,    _mouth_big_smile),
    "smirk":     (_eyes_squint,  _mouth_smirk),
    "excited":   (_eyes_wide,    _mouth_big_smile),
    "nervous":   (_eyes_wide,    _mouth_straight),
}

# Mapeo desde tags de la IA a emociones internas
EMOTION_TAGS = {
    "[happy]":     "happy",
    "[very_happy]":"very_happy",
    "[sad]":       "sad",
    "[surprised]": "surprised",
    "[thinking]":  "thinking",
    "[sleepy]":    "sleepy",
    "[love]":      "love",
    "[smirk]":     "smirk",
    "[excited]":   "excited",
    "[nervous]":   "nervous",
}

def _render(emotion: str, mouth_open: bool = False) -> Image.Image:
    """Genera un frame PIL con la emoción indicada."""
    img  = Image.new("1", (W, H))
    draw = ImageDraw.Draw(img)

    eye_fn, mouth_fn = EMOTIONS.get(emotion, EMOTIONS["happy"])
    eye_fn(draw)

    if mouth_open and emotion not in ("sleeping", "sleepy"):
        # Alternar boca abierta/cerrada para animación de habla
        _mouth_open_small(draw)
    else:
        mouth_fn(draw)

    return img

def _display_thread():
    """Hilo daemon: consume la cola y actualiza el OLED."""
    global _current_emotion
    mouth_toggle = False
    talk_tick = 0

    while True:
        try:
            # Intentamos sacar un comando sin bloquear
            cmd = _q.get_nowait()
            if cmd is None:      # señal de cierre
                break
            _current_emotion = cmd
        except queue.Empty:
            pass

        if _talking.is_set():
            # Animación de boca: alterna cada ~200 ms
            mouth_toggle = not mouth_toggle
            img = _render(_current_emotion, mouth_open=mouth_toggle)
            _device.display(img)
            time.sleep(0.20)
        else:
            img = _render(_current_emotion, mouth_open=False)
            _device.display(img)
            time.sleep(0.05)

_thread = threading.Thread(target=_display_thread, daemon=True)
_thread.start()

# ── API pública ────────────────────────────────────────────────────────────────

def draw_face(emotion: str):
    """Cambia la cara. Seguro llamar desde cualquier hilo."""
    try:
        _q.put_nowait(emotion)
    except queue.Full:
        pass   # Si la cola está llena simplemente descartamos

def start_talking():
    """Activa la animación de boca."""
    _talking.set()

def stop_talking():
    """Detiene la animación de boca."""
    _talking.clear()

def shutdown():
    """Apaga el OLED limpiamente."""
    _talking.clear()
    try:
        _q.put_nowait(None)
    except queue.Full:
        pass
    time.sleep(0.1)
    _device.cleanup()
