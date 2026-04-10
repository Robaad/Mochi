import os
import re
import subprocess
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import time
import atexit
from config import SAMPLE_RATE, RECORD_SECONDS, MIC_DEVICE

# --- CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "es_ES-sharvard-medium.onnx")
# Ya no usaremos un archivo fijo para evitar bloqueos de disco
piper_process = None

def start_piper():
    global piper_process
    if piper_process:
        try: piper_process.terminate()
        except: pass
    
    try:
        env = os.environ.copy()
        env["ONNXRUNTIME_DEVICE_PRIORITY"] = "CPU"
        
        # CAMBIO CLAVE: Quitamos --output_file y usamos STDOUT
        # Esto hace que Piper escupa el audio directamente a la memoria
        piper_cmd = [
            "piper", 
            "--model", MODEL_PATH, 
            "--output-raw", # Audio en bruto, ultra rápido
            "--length_scale", "1.10"
        ]
        
        piper_process = subprocess.Popen(
            piper_cmd, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, # Atrapamos el audio aquí
            stderr=subprocess.DEVNULL,
            text=False, # Importante: el audio es binario
            bufsize=0,
            env=env
        )
        print(f"✅ Motor de flujo directo iniciado.")
    except Exception as e:
        print(f"❌ Error Piper: {e}")

start_piper()

def record_audio():
    print("🎤 Escuchando...")
    try:
        audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16", device=MIC_DEVICE)
        sd.wait()
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav.write(tmp.name, SAMPLE_RATE, audio)
        return tmp.name
    except: return None

def speak(text):
    global piper_process
    if not text: return

    # Limpieza de texto
    clean_text = re.sub(r'\*.*?\*', '', text)
    clean_text = re.sub(r'[^\w\s,.:¡!¿?áéíóúÁÉÍÓÚñÑ]', '', clean_text).strip()
    
    if not clean_text: return
    print(f"Mochi dice: {clean_text}")

    try:
        # 1. Reiniciar si se colgó
        if piper_process.poll() is not None:
            start_piper()

        # 2. Enviamos el texto (con salto de línea para que procese)
        input_text = clean_text + "\n"
        piper_process.stdin.write(input_text.encode('utf-8'))
        piper_process.stdin.flush()
        
        # 3. REPRODUCCIÓN INSTANTÁNEA
        # Usamos 'aplay' leyendo directamente de la tubería de Piper
        # -t raw: audio en bruto
        # -r 22050: frecuencia típica de Piper
        # -f S16_LE: formato de 16 bits
        print("🔊 Reproduciendo...")
        
        # Este comando toma lo que Piper escupa y lo manda al altavoz Corsair (Card 1)
        # Es la forma más rápida posible en una Raspberry Pi
        subprocess.run(
            ["aplay", "-D", "plughw:1,0", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-q"],
            input=piper_process.stdout.read(len(clean_text) * 5000), # Leemos un buffer estimado
            timeout=10 # Por si se queda pillado
        )
            
    except Exception as e:
        print(f"❌ Error en flujo: {e}")
        start_piper()

def close_piper():
    if piper_process: piper_process.terminate()

atexit.register(close_piper)
