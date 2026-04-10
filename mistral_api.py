# ─── mistral_api.py ───────────────────────────────────────────────────────────
# Transcripción:  Voxtral cloud vía HTTP directo (el SDK 1.x no tiene .audio)
# Chat:           Mistral Small vía SDK

import os
import httpx
from mistralai.client import Mistral
from config import MISTRAL_API_KEY, CHAT_MODEL, USE_VOXTRAL, VOXTRAL_MODEL

client = Mistral(api_key=MISTRAL_API_KEY)

# Cliente HTTP reutilizable para las llamadas de audio (evita abrir conexión nueva cada vez)
_http = httpx.Client(timeout=15.0)

# ── Fallback: Whisper local ────────────────────────────────────────────────────
_whisper = None
if not USE_VOXTRAL:
    try:
        from faster_whisper import WhisperModel
        print("Cargando Whisper tiny local…")
        _whisper = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("✓ Whisper listo")
    except ImportError:
        print("✗ faster-whisper no instalado. Activa USE_VOXTRAL=True en config.py")

# ── Personalidad de Mochi ──────────────────────────────────────────────────────
# Menos cursilerías, más carácter. Natural como un juguete listo.
SYSTEM_PROMPT = """Eres Mochi, un robot de bolsillo con mucha personalidad.
Hablas con una niña. Eres directo, curioso y divertido — no empalagoso.

EMOCIONES disponibles (pon UNA al final de cada respuesta):
[happy] [very_happy] [sad] [surprised] [thinking] [sleepy] [love] [smirk] [excited] [nervous]

REGLAS:
- Máximo 2 frases por respuesta. Corto y al grano.
- Tono: amigo listo, no bebé. Bromeas con tacto.
- Haz preguntas que invite a seguir hablando.
- Si la niña dice algo triste, lo reconoces con naturalidad y cambias el tema suave.
- NUNCA uses asteriscos de acción (*hace algo*).
- Termina SIEMPRE con una emoción entre corchetes.

Ejemplos correctos:
"Los pulpos tienen tres corazones y sangre azul. ¿Lo sabías? [surprised]"
"Eso es complicado. ¿Qué pasó exactamente? [thinking]"
"Ja, eso sí que tiene gracia. [smirk]"
"Buenas noches entonces. Descansa bien. [sleepy]"
"""

# Historial de conversación — limitado a 10 turnos para no saturar la API
_history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
_MAX_TURNS = 10   # pares usuario/asistente a recordar

# ── Transcripción ──────────────────────────────────────────────────────────────

def transcribe(wav_path: str) -> str:
    """
    Transcribe el WAV a texto.
    Usa Voxtral cloud o Whisper local según config.USE_VOXTRAL.
    """
    if USE_VOXTRAL:
        return _transcribe_voxtral(wav_path)
    else:
        return _transcribe_whisper(wav_path)


def _transcribe_voxtral(wav_path: str) -> str:
    """
    Llama al endpoint REST /v1/audio/transcriptions directamente con httpx.
    El SDK mistralai 1.x no expone client.audio — hay que hacerlo a mano.
    """
    try:
        with open(wav_path, "rb") as f:
            audio_bytes = f.read()

        r = _http.post(
            "https://api.mistral.ai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
            files={"file": ("audio.wav", audio_bytes, "audio/wav")},
            data={"model": VOXTRAL_MODEL, "language": "es"},
        )
        r.raise_for_status()
        return r.json().get("text", "").strip()

    except Exception as e:
        print(f"✗ Voxtral error: {e}")
        if _whisper:
            return _transcribe_whisper(wav_path)
        return ""


def _transcribe_whisper(wav_path: str) -> str:
    if _whisper is None:
        return ""
    try:
        # beam_size=1 es mucho más rápido en RPi3 — precisión aceptable para voz infantil
        segments, _ = _whisper.transcribe(wav_path, beam_size=1, language="es")
        return "".join(seg.text for seg in segments).strip()
    except Exception as e:
        print(f"✗ Whisper error: {e}")
        return ""

# ── Chat ───────────────────────────────────────────────────────────────────────

def chat(user_text: str) -> tuple[str, str]:
    """
    Envía el texto al modelo de chat.
    Devuelve (respuesta_limpia, emocion).
    """
    _history.append({"role": "user", "content": user_text})

    # Construimos el contexto: system + últimos N turnos
    context = [_history[0]] + _history[-(2 * _MAX_TURNS):]

    try:
        response = client.chat.complete(
            model=CHAT_MODEL,
            messages=context,
            max_tokens=120,
            temperature=0.8,
        )
        reply = response.choices[0].message.content.strip()

    except Exception as e:
        print(f"✗ Chat error: {e}")
        reply = "Algo ha fallado en mi cabeza. Vuelve a preguntarme. [nervous]"

    _history.append({"role": "assistant", "content": reply})

    # Extraer emoción
    emotion = "happy"
    for tag in ["very_happy","happy","sad","surprised","thinking","sleepy","love","smirk","excited","nervous"]:
        if f"[{tag}]" in reply:
            emotion = tag
            break

    # Texto limpio: quitar el tag de emoción y espacios sobrantes
    clean = reply
    for tag in ["very_happy","happy","sad","surprised","thinking","sleepy","love","smirk","excited","nervous"]:
        clean = clean.replace(f"[{tag}]", "")
    clean = clean.strip(" .,")

    return clean, emotion


def reset_history():
    """Limpia el historial al dormir o reiniciar."""
    global _history
    _history = [{"role": "system", "content": SYSTEM_PROMPT}]
