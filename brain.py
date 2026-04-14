# ─── brain.py ─────────────────────────────────────────────────────────────────
# Grabación → Voxtral → Mistral (streaming) → frases para tts.py
#
# Mistral streaming: en vez de esperar la respuesta completa (~2s),
# recibimos token a token y enviamos cada frase a Piper en cuanto
# termina con punto/signo. Así Piper empieza a hablar ~0.5s después
# de que Mistral empieza a responder.

import os, re, subprocess, tempfile, httpx
from mistralai import Mistral
from config import (
    MISTRAL_API_KEY, CHAT_MODEL, VOXTRAL_MODEL,
    MIC_DEV, RECORD_SECONDS, SAMPLE_RATE, SILENCE_RMS,
    HISTORY_TURNS,
)

_client = Mistral(api_key=MISTRAL_API_KEY)
_http   = httpx.Client(timeout=20.0)

# ── Personalidad ───────────────────────────────────────────────────────────────
_SYSTEM = """Eres Mochi, un robot con personalidad. Hablas con una niña.
Eres directo, curioso y divertido. Nada empalagoso.

EMOCIONES — pon UNA al final de cada respuesta:
[happy] [very_happy] [excited] [sad] [surprised] [thinking] [sleepy] [love] [smirk] [nervous]

REGLAS ESTRICTAS:
- Máximo 2 frases. Respuestas cortas.
- No uses asteriscos de acción.
- Haz una pregunta para continuar la conversación.
- Termina SIEMPRE con [emoción].

Ejemplos:
"Los pulpos tienen tres corazones. ¿Lo sabías? [surprised]"
"Ja, eso tiene gracia. ¿Y qué pasó después? [smirk]"
"Buenas noches. Descansa bien. [sleepy]"
"""

_history: list[dict] = []

def reset():
    global _history
    _history = []

# ── Audio entrada ──────────────────────────────────────────────────────────────

def _rms_wav(path: str) -> float:
    """RMS rápido leyendo los bytes de datos del WAV."""
    import struct, math
    try:
        with open(path, "rb") as f:
            f.seek(44)          # saltar cabecera WAV
            raw = f.read()
        if not raw:
            return 0.0
        n   = len(raw) // 2
        samples = struct.unpack(f"<{n}h", raw[:n*2])
        rms = math.sqrt(sum(s*s for s in samples) / n) / 32768.0
        return rms
    except Exception:
        return 0.0

def record() -> str | None:
    """
    Graba con arecord directamente — sin Python en el path de audio.
    Devuelve ruta WAV o None si silencio / error.
    """
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(
            ["arecord",
             "-D", MIC_DEV,
             "-f", "S16_LE",
             "-r", str(SAMPLE_RATE),
             "-c", "1",
             "-d", str(RECORD_SECONDS),
             "-q",
             path],
            timeout=RECORD_SECONDS + 3,
            check=True,
        )
        if _rms_wav(path) < SILENCE_RMS:
            os.unlink(path)
            return None
        return path
    except Exception as e:
        print(f"✗ arecord: {e}")
        try: os.unlink(path)
        except: pass
        return None

# ── Transcripción ──────────────────────────────────────────────────────────────

def transcribe(wav_path: str) -> str:
    """Voxtral vía REST — el SDK 1.x no tiene client.audio."""
    try:
        with open(wav_path, "rb") as f:
            data = f.read()
        r = _http.post(
            "https://api.mistral.ai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
            files={"file": ("audio.wav", data, "audio/wav")},
            data={"model": VOXTRAL_MODEL, "language": "es"},
        )
        r.raise_for_status()
        return r.json().get("text", "").strip()
    except Exception as e:
        print(f"✗ Voxtral: {e}")
        return ""

# ── Chat con streaming ─────────────────────────────────────────────────────────

_EMOTION_RE = re.compile(r"\[(happy|very_happy|excited|sad|surprised|thinking|sleepy|love|smirk|nervous)\]")
_SPLIT_RE   = re.compile(r"(?<=[.!?¿¡;:])\s+")
_CLEAN_RE   = re.compile(r"\[[^\]]*\]|\*[^*]*\*")

def chat_stream(user_text: str):
    """
    Generador que produce (frase, emocion_o_None) a medida que llegan tokens.
    La emoción solo aparece en la última tupla: ("", "happy").
    Permite que tts.py empiece a hablar antes de tener la respuesta completa.
    """
    _history.append({"role": "user", "content": user_text})

    ctx = [{"role": "system", "content": _SYSTEM}]
    ctx += _history[-(HISTORY_TURNS * 2):]

    buffer   = ""
    full_reply = ""
    emotion  = "happy"

    try:
        with _client.chat.stream(
            model=CHAT_MODEL,
            messages=ctx,
            max_tokens=120,
            temperature=0.8,
        ) as stream:
            for chunk in stream:
                delta = chunk.data.choices[0].delta.content or ""
                buffer     += delta
                full_reply += delta

                # Emitir frases completas en cuanto llega la puntuación
                parts = _SPLIT_RE.split(buffer)
                for frase in parts[:-1]:          # todas menos el fragmento final
                    frase = frase.strip()
                    if not frase:
                        continue
                    # Extraer emoción si viene en esta frase
                    m = _EMOTION_RE.search(frase)
                    if m:
                        emotion = m.group(1)
                    clean = _CLEAN_RE.sub("", frase).strip()
                    if clean:
                        yield clean, None          # None = todavía no sabemos emoción final
                buffer = parts[-1]                 # guardar el fragmento incompleto

    except Exception as e:
        print(f"✗ Mistral stream: {e}")
        yield "Algo ha fallado. Prueba otra vez.", "nervous"
        _history.append({"role": "assistant", "content": "[nervous]"})
        return

    # Procesar lo que quedó en el buffer
    if buffer.strip():
        m = _EMOTION_RE.search(buffer)
        if m:
            emotion = m.group(1)
        clean = _CLEAN_RE.sub("", buffer).strip()
        if clean:
            yield clean, None

    # Señal de fin con la emoción final
    yield "", emotion

    _history.append({"role": "assistant", "content": full_reply})


def chat_simple(user_text: str) -> tuple[str, str]:
    """Versión no-streaming para compatibilidad. Devuelve (texto, emocion)."""
    frases = []
    emotion = "happy"
    for frase, em in chat_stream(user_text):
        if em is not None:
            emotion = em
        elif frase:
            frases.append(frase)
    return " ".join(frases), emotion
