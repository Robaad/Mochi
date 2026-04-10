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
piper_process = None

def start_piper():
    global piper_process
    if piper_process:
        try: piper_process.terminate()
        except: pass
    
    try:
        env = os.environ.copy()
        env["ONNXRUNTIME_DEVICE_PRIORITY"] = "CPU"
        # Usamos salida raw para no pelear con cabeceras WAV
        piper_cmd = [
            "piper", "--model", MODEL_PATH, 
            "--output-raw", "--length_scale", "1.10"
        ]
        piper_process = subprocess.Popen(
            piper_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env
        )
        print(f"✅ Motor de voz iniciado.")
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

    # Limpieza de emojis y asteriscos (vital para Piper)
    clean_text = re.sub(r'\*.*?\*', '', text)
    clean_text = re.sub(r'[^\w\s,.:¡!¿?áéíóúÁÉÍÓÚñÑ]', '', clean_text).strip()
    if not clean_text: return

    print(f"Mochi dice: {clean_text}")

    try:
        # Si Piper se cerró, lo reiniciamos
        if piper_process.poll() is not None:
            start_piper()

        # Enviamos el texto y cerramos la entrada para que Piper escupa el audio
        # communicate() es la forma más segura de evitar bloqueos
        input_data = (clean_text + "\n").encode('utf-8')
        audio_data, _ = piper_process.communicate(input=input_data, timeout=10)

        if audio_data:
            print(f"🔊 Reproduciendo por Corsair (Card 1)...")
            # --- CAMBIO CRÍTICO AQUÍ: plughw:1,0 para tus Corsair ---
            aplay_cmd = ["aplay", "-D", "plughw:1,0", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-q"]
            subprocess.run(aplay_cmd, input=audio_data)
        else:
            print("⚠️ Piper no devolvió audio.")

        # Reiniciamos Piper para la siguiente frase (ya que communicate cierra el pipe)
        start_piper()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        start_piper()

def close_piper():
    if piper_process: piper_process.terminate()

atexit.register(close_piper)
