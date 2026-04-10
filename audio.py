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
# Usamos el modelo x_low de Carl (asegúrate de que el nombre coincida tras el wget)
MODEL_PATH = os.path.join(BASE_DIR, "es_ES-carlfm-x_low.onnx")
piper_process = None

def start_piper():
    """Inicia el proceso de Piper en modo streaming."""
    global piper_process
    if piper_process:
        try: piper_process.terminate()
        except: pass
    
    try:
        env = os.environ.copy()
        env["ONNXRUNTIME_DEVICE_PRIORITY"] = "CPU"
        
        # length_scale 0.85: Más rápido = más energía infantil
        piper_cmd = [
            "piper", "--model", MODEL_PATH, 
            "--output-raw", "--length_scale", "0.85"
        ]
        
        piper_process = subprocess.Popen(
            piper_cmd, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, 
            env=env
        )
    except Exception as e:
        print(f"❌ Error al iniciar Piper: {e}")

# Arrancamos el motor de voz al importar el módulo
start_piper()

def record_audio():
    """Graba audio del micrófono y lo guarda en un temporal."""
    print("🎤 Escuchando...")
    try:
        audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16", device=MIC_DEVICE)
        sd.wait()
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav.write(tmp.name, SAMPLE_RATE, audio)
        return tmp.name
    except Exception as e:
        print(f"❌ Error grabando: {e}")
        return None

def process_and_play(text_fragment):
    """Procesa una frase corta y la reproduce inmediatamente."""
    global piper_process
    if not text_fragment or len(text_fragment) < 2:
        return

    try:
        # Si el proceso Piper ha muerto, lo revivimos
        if piper_process.poll() is not None:
            start_piper()

        # Enviamos la frase a Piper
        input_data = (text_fragment + "\n").encode('utf-8')
        
        # Timeout de 10s para evitar bloqueos si la Pi está saturada
        audio_data, _ = piper_process.communicate(input=input_data, timeout=10)

        if audio_data:
            # -r 32000: Sube el tono (Pitch) para que la voz de Carl sea de niño/robot
            aplay_cmd = [
                "aplay", "-D", "plughw:1,0", 
                "-r", "32000", "-f", "S16_LE", 
                "-t", "raw", "-q"
            ]
            subprocess.run(aplay_cmd, input=audio_data)
        
        # Reiniciamos Piper para la siguiente frase (communicate cierra el flujo)
        start_piper()
            
    except Exception as e:
        print(f"❌ Error en fragmento de voz: {e}")
        start_piper()

def speak(text):
    """
    Función principal llamada desde main.py. 
    Divide el texto largo en frases para que Mochi empiece a hablar antes.
    """
    if not text:
        return

    # 1. Limpieza de texto (filtramos emojis y asteriscos de acciones)
    clean_text = re.sub(r'\*.*?\*', '', text)
    clean_text = re.sub(r'[^\w\s,.:¡!¿?áéíóúÁÉÍÓÚñÑ]', '', clean_text).strip()
    
    if not clean_text:
        return

    # 2. Troceado por signos de puntuación
    # Esto permite que si la IA responde un párrafo, Mochi diga la frase 1 mientras procesa la 2.
    frases = re.split(r'[.!?:]', clean_text)
    
    for frase in frases:
        f = frase.strip()
        if len(f) > 1:
            print(f"Mochi dice: {f}")
            process_and_play(f)

def close_piper():
    """Asegura que Piper se cierre al apagar el programa."""
    if piper_process:
        piper_process.terminate()

atexit.register(close_piper)
