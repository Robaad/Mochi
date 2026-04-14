# ─── main.py ──────────────────────────────────────────────────────────────────
# Bucle principal de Mochi. Arquitectura:
#
#  ESPERA (botón) → GRABA (arecord) → TRANSCRIBE (Voxtral) →
#  STREAM CHAT (Mistral) → HABLA solapado (Piper + aplay) → vuelta
#
# Un solo hilo principal + hilo del display + hilo producer de Piper.
# Sin pygame, sin sounddevice, sin resample.

import os, sys, time, signal, threading
import gpiozero
import display
import tts
import brain
from config import BUTTON_PIN, INACTIVITY_SEC

# ── Estado ─────────────────────────────────────────────────────────────────────
_awake    = threading.Event()
_quit     = threading.Event()
_last_t   = 0.0

# ── Botón ──────────────────────────────────────────────────────────────────────
try:
    _btn = gpiozero.Button(BUTTON_PIN, pull_up=True, bounce_time=0.15)
except Exception as e:
    print(f"⚠ GPIO no disponible ({e}) — usa Enter")
    _btn = None

def _wake():
    global _last_t
    if _awake.is_set():
        # Si ya está despierto, el botón reinicia el timer de inactividad
        _last_t = time.time()
        return
    print("✨ Mochi despierto")
    _awake.set()
    _last_t = time.time()
    display.face("excited")
    tts.speak_simple("¿Qué hay?")

def _sleep():
    print("💤 Mochi duerme")
    _awake.clear()
    brain.reset()
    display.face("sleeping")
    tts.speak_simple("Hasta luego.")

if _btn:
    _btn.when_pressed = _wake

# ── Señales ────────────────────────────────────────────────────────────────────

def _bye(sig, frame):
    print("\nApagando…")
    _quit.set()
    display.face("sleeping")
    time.sleep(0.2)
    display.off()
    sys.exit(0)

signal.signal(signal.SIGINT,  _bye)
signal.signal(signal.SIGTERM, _bye)

# ── Turno de conversación ──────────────────────────────────────────────────────

def _turn():
    """Un turno completo: graba → transcribe → responde."""
    global _last_t

    # 1. Grabar
    display.face("happy")
    wav = brain.record()

    if wav is None:
        # Silencio — comprobar inactividad
        if _awake.is_set() and time.time() - _last_t > INACTIVITY_SEC:
            _sleep()
        elif _awake.is_set() and time.time() - _last_t > INACTIVITY_SEC * 0.6:
            display.face("sleepy")
        return

    # 2. Transcribir (cara pensando mientras espera)
    display.face("thinking")
    text = brain.transcribe(wav)
    try:
        os.unlink(wav)
    except Exception:
        pass

    if not text:
        display.face("happy")
        return

    print(f"  Tú  → {text}")
    _last_t = time.time()

    # 3. Streaming: Mistral genera y Piper habla en paralelo
    #    chat_stream() es un generador → tts.speak() consume frases
    #    conforme llegan, solapando síntesis y reproducción.

    frases_iter  = brain.chat_stream(text)
    emotion      = "happy"
    frase_buffer = []

    display.face("thinking")

    def _producer_and_play():
        """Recorre el stream y habla cada frase en cuanto llega."""
        nonlocal emotion
        import queue as _queue
        import tts as _tts
        import re as _re

        wav_q: "_queue.Queue[str | None]" = _queue.Queue(maxsize=3)

        def _synth_worker():
            for frase, em in frases_iter:
                if em is not None:
                    emotion = em
                    wav_q.put(None)   # fin
                    return
                if frase:
                    frase_buffer.append(frase)
                    wav = _tts._synthesize(frase)
                    wav_q.put(wav)
            wav_q.put(None)

        synth_t = threading.Thread(target=_synth_worker, daemon=True)
        synth_t.start()

        first = True
        while True:
            wav = wav_q.get()
            if wav is None:
                break
            if first:
                # Primera frase lista — ahora sí mostramos emoción
                display.face(emotion if emotion != "happy" else "happy")
                display.talking(True)
                first = False
            _tts._play(wav)

        display.talking(False)

    _producer_and_play()

    # Emoción final en la cara
    display.face(emotion)
    resp_text = " ".join(frase_buffer)
    print(f"Mochi [{emotion}] → {resp_text}")

# ── Bucle principal ────────────────────────────────────────────────────────────

def main():
    print("🤖 Mochi listo —", "pulsa el botón" if _btn else "pulsa Enter para despertar")
    display.face("sleeping")

    while not _quit.is_set():

        if not _awake.is_set():
            if not _btn:
                try:
                    input()
                    _wake()
                except EOFError:
                    time.sleep(0.3)
            else:
                time.sleep(0.1)
            continue

        _turn()

if __name__ == "__main__":
    main()
