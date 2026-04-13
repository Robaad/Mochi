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

# ── Resolución robusta de salida ALSA ─────────────────────────────────────────
def _detect_playback_devices() -> list[tuple[int, str, int]]:
    """
    Devuelve [(card_idx, card_name, device_idx), ...] usando `aplay -l`.
    """
    try:
        out = subprocess.run(
            ["aplay", "-l"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        ).stdout
    except Exception:
        return []

    devices: list[tuple[int, str, int]] = []
    current_card: tuple[int, str] | None = None

    for line in out.splitlines():
        card_match = re.search(r"^card\s+(\d+):\s*([^\[]+)", line)
        if card_match:
            current_card = (int(card_match.group(1)), card_match.group(2).strip())
            dev_match = re.search(r"device\s+(\d+):", line)
            if dev_match:
                devices.append((current_card[0], current_card[1], int(dev_match.group(1))))
            continue

        dev_match = re.search(r"^\s*device\s+(\d+):", line)
        if dev_match and current_card:
            devices.append((current_card[0], current_card[1], int(dev_match.group(1))))

    return devices


def _resolve_output_device(configured: str) -> tuple[str, int | None]:
    """
    Elige dispositivo ALSA:
      1) Usa el configurado si parece válido.
      2) Si no, prioriza CORSAIR/USB.
      3) Si no, usa el primero disponible.
    Devuelve ("hw:X,Y", card_idx_para_env_o_None).
    """
    cfg = (configured or "").strip()
    m = re.match(r"^hw:(\d+),(\d+)$", cfg)
    if m:
        return cfg, int(m.group(1))

    devices = _detect_playback_devices()
    if not devices:
        return (cfg or "default"), None

    prefer = None
    for card_idx, card_name, device_idx in devices:
        name = card_name.lower()
        if any(k in name for k in ("corsair", "slipstream", "usb")):
            prefer = (card_idx, device_idx)
            break

    if prefer is None:
        prefer = (devices[0][0], devices[0][2])

    return f"hw:{prefer[0]},{prefer[1]}", prefer[0]


_AUDIO_OUT_DEV_RESOLVED, _AUDIO_CARD_IDX = _resolve_output_device(AUDIO_OUT_DEV)

# ── pygame mixer ───────────────────────────────────────────────────────────────
# frequency=22050 coincide exactamente con la salida de Piper carlfm x_low.
# size=-16 = signed 16-bit. channels=1 = mono. buffer=512 = baja latencia.
# device: pygame usa ALSA_DEVICE o SDL_AUDIODEV — lo forzamos vía env.
os.environ["SDL_AUDIODRIVER"] = "alsa"
os.environ["AUDIODEV"] = _AUDIO_OUT_DEV_RESOLVED
if _AUDIO_CARD_IDX is not None:
    os.environ["ALSA_CARD"] = str(_AUDIO_CARD_IDX)
    os.environ["ALSA_PCM_CARD"] = str(_AUDIO_CARD_IDX)

print(f"🔊 Salida de audio: {_AUDIO_OUT_DEV_RESOLVED}")

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
    """Graba a 44.1kHz y convierte a 16kHz para Whisper."""
    try:
        print("🎤 Escuchando…")
        # Grabamos a la frecuencia nativa del micro (44.1k)
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

        # --- TRUCO: Re-muestreo para Whisper ---
        from scipy.signal import resample
        # Calculamos cuántas muestras son a 16000Hz
        num_samples_16k = int(len(audio) * 16000 / SAMPLE_RATE)
        audio_16k = resample(audio, num_samples_16k).astype(np.int16)

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        # Guardamos a 16000 para que la IA no se líe
        wav_io.write(path, 16000, audio_16k) 
        return path

    except Exception as e:
        print(f"✗ Error grabando: {e}")
        return None

# ── TTS ────────────────────────────────────────────────────────────────────────

def _speak_fragment(text: str):
    if not _PIPER_OK or not text.strip():
        return

    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    # Asegúrate de que MODEL_PATH sea absoluto
    cmd = [
        "piper",
        "--model", TTS_MODEL_PATH,
        "--output_file", wav_path,
        "--length_scale", str(TTS_LENGTH_SCALE),
    ]

    with _speak_lock:
        try:
            # Añadimos env=os.environ para que Piper vea las librerías del sistema
            result = subprocess.run(
                cmd,
                input=(text.strip() + "\n").encode("utf-8"),
                capture_output=True,
                env=os.environ.copy(), 
                timeout=15
            )

            if result.returncode != 0:
                print(f"✗ Piper error code {result.returncode}")
                # Imprime el error real para debuguear
                print(result.stderr.decode())
                return

            if os.path.exists(wav_path) and os.path.getsize(wav_path) > 100:
                sound = pygame.mixer.Sound(wav_path)
                channel = sound.play()
                while channel.get_busy():
                    pygame.time.wait(10)
        except Exception as e:
            print(f"✗ TTS error: {e}")
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)

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
