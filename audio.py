# ─── audio.py ─────────────────────────────────────────────────────────────────
# Entrada:  micrófono USB → WAV temporal con detección de silencio por RMS
# Salida:   texto → Piper (WAV temporal) → pygame mixer
#
# Por qué WAV temporal en vez de --output-raw:
#   En RPi3, Piper con --output-raw pipe→aplay introduce latencia extra
#   porque aplay no empieza hasta recibir datos suficientes para su buffer.
#   Con --output-file generamos el WAV completo y pygame lo reproduce
#   directamente desde memoria, sin buffer de red ni ALSA extra.
#   El warning de onnxruntime GPU es inofensivo — RPi no tiene GPU ONNX.

import os
import re
import subprocess
import threading
import tempfile
import atexit
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav_io
import pygame

from config import (
    SAMPLE_RATE, CHANNELS, RECORD_SECONDS,
    MIC_DEVICE, SILENCE_THRESHOLD,
    TTS_MODEL_PATH, TTS_LENGTH_SCALE, AUDIO_OUT_DEV,
)

# ── pygame mixer ───────────────────────────────────────────────────────────────
# frequency=22050 coincide exactamente con la salida de Piper carlfm x_low.
# size=-16 = signed 16-bit. channels=1 = mono. buffer=512 = baja latencia.
# device: pygame usa ALSA_DEVICE o SDL_AUDIODEV — lo forzamos vía env.
os.environ["AUDIODEV"]      = AUDIO_OUT_DEV   # SDL_AUDIODEV para pygame
os.environ["ALSA_CARD"]     = AUDIO_OUT_DEV   # por si acaso

try:
    pygame.mixer.pre_init(frequency=22050, size=-16, channels=1, buffer=512)
    pygame.mixer.init()
    print(f"✓ pygame mixer listo ({pygame.mixer.get_init()})")
except Exception as e:
    print(f"✗ pygame mixer error: {e} — intentando con defaults")
    try:
        pygame.mixer.init()
    except Exception as e2:
        print(f"✗ pygame no disponible: {e2}")

# ── Verificar Piper ────────────────────────────────────────────────────────────
def _check_piper() -> bool:
    try:
        subprocess.run(["piper", "--version"], capture_output=True, timeout=3)
        print("✓ Piper TTS listo")
        return True
    except FileNotFoundError:
        print("✗ Piper no encontrado")
        return False
    except Exception:
        print("✓ Piper TTS listo")
        return True

_PIPER_OK = _check_piper()

# Lock para serializar reproducción
_speak_lock = threading.Lock()

# ── Grabación ─────────────────────────────────────────────────────────────────

def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2))) / 32768.0


def record_audio() -> "str | None":
    """Graba RECORD_SECONDS. Devuelve ruta WAV o None si silencio."""
    try:
        print("🎤 Escuchando…")
        audio = sd.rec(
            int(RECORD_SECONDS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            device=MIC_DEVICE,
        )
        sd.wait()

        if _rms(audio) < SILENCE_THRESHOLD:
            return None

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        wav_io.write(path, SAMPLE_RATE, audio)
        return path

    except Exception as e:
        print(f"✗ Error grabando: {e}")
        return None


# ── TTS ────────────────────────────────────────────────────────────────────────

def _speak_fragment(text: str):
    """Sintetiza una frase con Piper → WAV temporal → pygame. Bloqueante."""
    if not _PIPER_OK or not text.strip():
        return

    # Generamos WAV en un fichero temporal — evita el problema del pipe/buffer
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    cmd = [
        "piper",
        "--model",        TTS_MODEL_PATH,
        "--output_file",  wav_path,        # WAV completo, no raw
        "--length_scale", str(TTS_LENGTH_SCALE),
    ]

    with _speak_lock:
        try:
            result = subprocess.run(
                cmd,
                input=(text.strip() + "\n").encode("utf-8"),
                capture_output=True,
                timeout=15,
            )

            # El warning de onnxruntime va a stderr pero returncode=0 → ok
            if result.returncode != 0:
                err = result.stderr.decode(errors="ignore")
                # Ignorar warnings de GPU, solo mostrar errores reales
                real_errors = [l for l in err.splitlines()
                               if "[W:" not in l and "[I:" not in l and l.strip()]
                if real_errors:
                    print(f"✗ Piper error: {real_errors[0][:120]}")
                return

            if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 100:
                print("✗ Piper: WAV vacío")
                return

            # Reproducir con pygame
            try:
                sound = pygame.mixer.Sound(wav_path)
                channel = sound.play()
                while channel.get_busy():
                    pygame.time.wait(20)
            except Exception as e:
                print(f"✗ pygame play error: {e}")

        except subprocess.TimeoutExpired:
            print("✗ Piper timeout")
        except Exception as e:
            print(f"✗ TTS error: {e}")
        finally:
            try:
                os.unlink(wav_path)
            except Exception:
                pass


_SPLIT_RE = re.compile(r"(?<=[.!?¿¡:])\s+")
_CLEAN_RE  = re.compile(r"\*[^*]*\*|\[[^\]]*\]|[^\w\s,.:¡!¿?áéíóúÁÉÍÓÚñÑ]")


def speak(text: str):
    """Limpia, parte en frases y reproduce en orden."""
    clean = _CLEAN_RE.sub("", text).strip()
    if not clean:
        return

    frases = [f.strip() for f in _SPLIT_RE.split(clean) if len(f.strip()) > 1]
    if not frases:
        frases = [clean]

    for frase in frases:
        _speak_fragment(frase)


def shutdown():
    try:
        pygame.mixer.quit()
    except Exception:
        pass

atexit.register(shutdown)
