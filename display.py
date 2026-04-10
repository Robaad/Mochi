from luma.core.interface.serial import i2c
from luma.oled.device import sh1106 as oled_driver
from PIL import Image, ImageDraw
from config import OLED_WIDTH, OLED_HEIGHT, I2C_ADDRESS
import time

serial = i2c(port=1, address=I2C_ADDRESS)
device = oled_driver(serial, width=OLED_WIDTH, height=OLED_HEIGHT)

def draw_face(emotion="happy"):
    img = Image.new("1", (OLED_WIDTH, OLED_HEIGHT))
    draw = ImageDraw.Draw(img)

    # Coordenadas: [x_inicial, y_inicial, x_final, y_final]
    faces = {
        "happy":     {"eyes": [(38,20,50,32),(78,20,90,32)], "mouth": [34, 30, 90, 50]},
        "sad":        {"eyes": [(38,20,50,32),(78,20,90,32)], "mouth": [34, 45, 90, 65]},
        "surprised": {"eyes": [(36,18,52,34),(76,18,92,34)], "mouth": [54, 42, 74, 58]},
        "thinking":  {"eyes": [(38,22,50,28),(78,22,90,28)], "mouth": [42, 50, 86, 50]},
        "sleepy":    {"eyes": [(38,26,50,28),(78,26,90,28)], "mouth": [44, 50, 84, 52]},
    }

    f = faces.get(emotion, faces["happy"])
    
    # Dibujar ojos
    for eye in f["eyes"]:
        draw.ellipse(eye, fill=1)

    # Dibujar boca
    m = f["mouth"]
    if emotion == "surprised":
        draw.ellipse(m, outline=1)
    elif emotion in ("thinking", "sleepy"):
        # Para líneas usamos draw.line con la lista de coordenadas
        draw.line(m, fill=1, width=2)
    elif emotion == "sad":
        # Sad: Arco invertido (parte de arriba)
        draw.arc(m, start=180, end=360, fill=1, width=3)
    else:
        # Happy y otros: Sonrisa (parte de abajo)
        # Usamos start=0, end=180 para que la curva vaya hacia abajo
        draw.arc(m, start=0, end=180, fill=1, width=3)

    device.display(img)

def show_talking(duration=2):
    """Hace que Mochi mueva la boca durante unos segundos."""
    end_time = time.time() + duration
    while time.time() < end_time:
        # Boca cerrada (Happy)
        draw_face("happy")
        time.sleep(0.2)
        
        # Boca abierta
        img = Image.new("1", (OLED_WIDTH, OLED_HEIGHT))
        draw = ImageDraw.Draw(img)
        draw.ellipse([38,20,50,32], fill=1) # Ojo izq
        draw.ellipse([78,20,90,32], fill=1) # Ojo der
        draw.ellipse([50,45,78,60], outline=1) # Boca abierta (óvalo)
        device.display(img)
        time.sleep(0.2)
