# ─── display.py ───────────────────────────────────────────────────────────────
# OLED 128×64 con animación de boca en hilo propio.
# Usa cola para que nunca bloquee el hilo principal.

import time, threading, queue
from luma.core.interface.serial import i2c
from luma.oled.device import sh1106 as OledDriver
from PIL import Image, ImageDraw
from config import OLED_WIDTH as W, OLED_HEIGHT as H, I2C_ADDRESS

_dev = OledDriver(i2c(port=1, address=I2C_ADDRESS), width=W, height=H)

# ── Caras ──────────────────────────────────────────────────────────────────────

def _eyes_normal(d):
    d.ellipse([38,18,52,32], fill=1); d.ellipse([76,18,90,32], fill=1)

def _eyes_wide(d):
    d.ellipse([33,13,54,34], fill=1); d.ellipse([74,13,95,34], fill=1)

def _eyes_half(d):
    d.rectangle([38,25,52,32], fill=1); d.rectangle([76,25,90,32], fill=1)

def _eyes_closed(d):
    d.line([38,28,52,28], fill=1, width=2); d.line([76,28,90,28], fill=1, width=2)

def _eyes_squint(d):
    d.rectangle([38,23,52,27], fill=1); d.rectangle([76,23,90,27], fill=1)

def _eyes_heart(d):
    for cx in (45, 83):
        for dx,dy in [(-3,-4),(3,-4),(0,-1)]:
            d.rectangle([cx+dx-2,22+dy-2,cx+dx+2,22+dy+2], fill=1)

def _mouth_smile(d):
    d.arc([34,32,90,56], start=0, end=180, fill=1, width=3)

def _mouth_big(d):
    d.arc([28,28,96,58], start=0, end=180, fill=1, width=3)
    d.line([28,43,33,37], fill=1, width=2); d.line([96,43,91,37], fill=1, width=2)

def _mouth_sad(d):
    d.arc([34,40,90,62], start=180, end=360, fill=1, width=3)

def _mouth_flat(d):
    d.line([42,48,86,48], fill=1, width=2)

def _mouth_o(d):
    d.ellipse([52,38,74,58], outline=1, width=2)

def _mouth_smirk(d):
    d.line([42,48,62,44], fill=1, width=2)
    d.arc([52,40,86,54], start=320, end=40, fill=1, width=2)

def _mouth_open(d):
    d.ellipse([50,40,78,56], outline=1, width=2)

FACES = {
    "happy":      (_eyes_normal, _mouth_smile),
    "very_happy": (_eyes_normal, _mouth_big),
    "excited":    (_eyes_wide,   _mouth_big),
    "sad":        (_eyes_normal, _mouth_sad),
    "surprised":  (_eyes_wide,   _mouth_o),
    "thinking":   (_eyes_squint, _mouth_flat),
    "sleepy":     (_eyes_half,   _mouth_flat),
    "sleeping":   (_eyes_closed, _mouth_flat),
    "love":       (_eyes_heart,  _mouth_big),
    "smirk":      (_eyes_squint, _mouth_smirk),
    "nervous":    (_eyes_wide,   _mouth_flat),
}

# ── Hilo del display ───────────────────────────────────────────────────────────
_q        = queue.Queue(maxsize=6)
_talking  = threading.Event()
_emotion  = "sleeping"

def _loop():
    global _emotion
    mouth = False
    while True:
        try:
            cmd = _q.get_nowait()
            if cmd is None:
                break
            _emotion = cmd
        except queue.Empty:
            pass

        img  = Image.new("1", (W, H))
        draw = ImageDraw.Draw(img)
        eye_fn, mouth_fn = FACES.get(_emotion, FACES["happy"])
        eye_fn(draw)

        if _talking.is_set() and _emotion not in ("sleeping","sleepy"):
            mouth = not mouth
            (_mouth_open if mouth else mouth_fn)(draw)
            _dev.display(img)
            time.sleep(0.18)
        else:
            mouth_fn(draw)
            _dev.display(img)
            time.sleep(0.06)

threading.Thread(target=_loop, daemon=True).start()

# ── API pública ────────────────────────────────────────────────────────────────

def face(emotion: str):
    try: _q.put_nowait(emotion)
    except queue.Full: pass

def talking(on: bool):
    _talking.set() if on else _talking.clear()

def off():
    talking(False)
    try: _q.put_nowait(None)
    except queue.Full: pass
    time.sleep(0.15)
    _dev.cleanup()
