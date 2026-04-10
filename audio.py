import os
import re
import subprocess
import shutil
import select
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import time
import atexit
import numpy as np
from config import SAMPLE_RATE, RECORD_SECONDS, MIC_DEVICE

# --- CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "es_ES-sharvard-medium.onnx")
piper_process = None
PIPER_SAMPLE_RATE = 22050
MIN_AUDIO_LEVEL = 180
MAX_RECORD_RETRIES = 2
DEFAULT_APLAY_DEVICE = os.getenv("MOCHI_SPEAKER_DEVICE", "default")

def _resolve_aplay_device():
    """Intenta elegir una salida de audio válida para evitar errores intermitentes."""
    if shutil.which("aplay") is None:
        return None
    try:
        res = subprocess.run(
            ["aplay", "-L"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False
        )
        available = res.stdout.splitlines()
        if DEFAULT_APLAY_DEVICE == "default":
            return "default"
        if DEFAULT_APLAY_DEVICE in available:
            return DEFAULT_APLAY_DEVICE
        # fallback común en Raspberry
        for candidate in ("plughw:1,0", "plughw:0,0", "sysdefault"):
            if any(candidate in line for line in available):
                return candidate
    except Exception:
        pass
    return "default"

APLAY_DEVICE = _resolve_aplay_device()

def start_piper():
    global piper_process
    if piper_process:
        try:
            piper_process.terminate()
        except Exception:
            pass
    
    try:
        env = os.environ.copy()
        env["ONNXRUNTIME_DEVICE_PRIORITY"] = "CPU"
        
        piper_cmd = [
            "piper", 
            "--model", MODEL_PATH, 
            "--output_raw",
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
        print("✅ Motor de voz iniciado.")
    except Exception as e:
        print(f"❌ Error Piper: {e}")

start_piper()

def record_audio():
    for attempt in range(MAX_RECORD_RETRIES + 1):
        print(f"🎤 Escuchando... (intento {attempt + 1})")
        try:
            audio = sd.rec(
                int(RECORD_SECONDS * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                device=MIC_DEVICE
            )
            sd.wait()
            audio_level = float(np.mean(np.abs(audio)))
            if audio_level < MIN_AUDIO_LEVEL:
                print(f"⚠️ Audio muy bajo ({audio_level:.1f}).")
                if attempt < MAX_RECORD_RETRIES:
                    time.sleep(0.2)
                    continue
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wav.write(tmp.name, SAMPLE_RATE, audio)
            return tmp.name
        except Exception as e:
            print(f"❌ Error grabando audio: {e}")
            if attempt < MAX_RECORD_RETRIES:
                time.sleep(0.2)
                continue
    return None

def speak(text):
    global piper_process
    if not text:
        return

    clean_text = re.sub(r'\*.*?\*', '', text)
    clean_text = re.sub(r'[^\w\s,.:;¡!¿?áéíóúÁÉÍÓÚñÑ\-]', '', clean_text).strip()
    
    if not clean_text:
        return
    print(f"Mochi dice: {clean_text}")

    try:
        if piper_process is None or piper_process.poll() is not None:
            start_piper()

        input_text = clean_text + "\n"
        piper_process.stdin.write(input_text.encode('utf-8'))
        piper_process.stdin.flush()

        # Piper no indica tamaño exacto por frase en flujo continuo.
        # Evitamos bloqueos usando select() sobre el pipe.
        chunks = []
        quiet_windows = 0
        max_quiet_windows = 6
        while quiet_windows < max_quiet_windows:
            ready, _, _ = select.select([piper_process.stdout], [], [], 0.25)
            if not ready:
                quiet_windows += 1
                continue

            data = os.read(piper_process.stdout.fileno(), 4096)
            if not data:
                quiet_windows += 1
                time.sleep(0.03)
                continue

            chunks.append(data)
            quiet_windows = 0 if any(b != 0 for b in data) else quiet_windows + 1
            if len(chunks) > 700:  # límite de seguridad
                break

        raw_audio = b"".join(chunks)
        if not raw_audio:
            print("⚠️ Piper no devolvió audio.")
            return

        if APLAY_DEVICE is None:
            print("⚠️ aplay no disponible, no se puede reproducir audio.")
            return

        print(f"🔊 Reproduciendo por {APLAY_DEVICE}...")
        subprocess.run(
            ["aplay", "-D", APLAY_DEVICE, "-r", str(PIPER_SAMPLE_RATE), "-f", "S16_LE", "-t", "raw", "-q"],
            input=raw_audio,
            timeout=12,
            check=False
        )
            
    except Exception as e:
        print(f"❌ Error en flujo: {e}")
        start_piper()

def close_piper():
    if piper_process:
        piper_process.terminate()

atexit.register(close_piper)
