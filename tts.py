# ─── tts.py ───────────────────────────────────────────────────────────────────
# Convierte texto a voz con Piper + aplay.
#
# Truco de fluidez: speak_streaming() acepta un iterador de frases.
# Mientras aplay reproduce la frase N, un hilo worker ya está corriendo
# Piper para generar la frase N+1. Así el silencio entre frases es ~0.
#
#   frase 1: [Piper 2s][aplay 1.5s]
#   frase 2:        [Piper 2s  ][aplay 1.5s]   ← solapado
#   frase 3:                 [Piper 2s  ][aplay 1.5s]

import os, subprocess, tempfile, threading, queue
from config import TTS_MODEL, TTS_SPEED, SPK_DEV

_PIPER_ENV = os.environ.copy()
_PIPER_ENV["ONNXRUNTIME_DEVICE_PRIORITY"] = "CPU"   # silencia warning GPU

def _synthesize(text: str) -> str | None:
    """Llama a Piper, devuelve ruta WAV o None si falla."""
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        r = subprocess.run(
            ["piper", "--model", TTS_MODEL,
             "--output_file", path,
             "--length_scale", TTS_SPEED],
            input=(text.strip() + "\n").encode(),
            capture_output=True,
            env=_PIPER_ENV,
            timeout=14,
        )
        if r.returncode == 0 and os.path.getsize(path) > 200:
            return path
    except Exception as e:
        print(f"✗ Piper: {e}")
    try: os.unlink(path)
    except: pass
    return None

def _play(wav_path: str):
    """Reproduce con aplay y borra el WAV al terminar."""
    try:
        subprocess.run(
            ["aplay", "-D", SPK_DEV, "-q", wav_path],
            timeout=30,
        )
    except Exception as e:
        print(f"✗ aplay: {e}")
    finally:
        try: os.unlink(wav_path)
        except: pass

def speak(text: str):
    """
    Habla el texto completo. Divide en frases y las solapa:
    Piper de la frase siguiente corre mientras aplay reproduce la actual.
    """
    import re
    clean = re.sub(r"\[[^\]]*\]|\*[^*]*\*", "", text).strip()
    clean = re.sub(r"[^\w\s,.:;¡!¿?áéíóúÁÉÍÓÚñÑüÜ]", "", clean).strip()
    if not clean:
        return

    # Partir en frases por puntuación
    frases = [f.strip() for f in re.split(r"(?<=[.!?¿¡;:])\s+", clean) if len(f.strip()) > 1]
    if not frases:
        frases = [clean]

    # Cola de WAVs pre-sintetizados
    wav_q: queue.Queue[str | None] = queue.Queue(maxsize=2)

    def _producer():
        for f in frases:
            wav_q.put(_synthesize(f))
        wav_q.put(None)  # fin

    prod = threading.Thread(target=_producer, daemon=True)
    prod.start()

    while True:
        wav = wav_q.get()
        if wav is None:
            break
        _play(wav)

def speak_simple(text: str):
    """Versión sin solapamiento — para frases cortas de arranque/despedida."""
    wav = _synthesize(text)
    if wav:
        _play(wav)
