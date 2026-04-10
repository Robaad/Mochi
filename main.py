# ─── main.py ──────────────────────────────────────────────────────────────────
# Arquitectura pipeline:
#
#   [Botón GPIO]
#        │
#        ▼
#   record_audio()          ← hilo principal, bloquea RECORD_SECONDS
#        │
#        ▼
#   transcribe()            ← hilo worker (no bloquea display ni botón)
#        │
#        ▼
#   chat()                  ← en el mismo worker
#        │
#        ▼
#   draw_face() + speak()   ← display en su hilo, TTS en worker
#
# Gracias al hilo del display (display.py) la cara se actualiza aunque
# el worker esté ocupado con la API. Piper corre en proceso persistente.

import os
import sys
import time
import threading
import signal

import gpiozero

import display
import audio
from mistral_api import transcribe, chat, reset_history
from config import BUTTON_PIN, INACTIVITY_LIMIT

# ── Estado global ──────────────────────────────────────────────────────────────
_active   = threading.Event()    # True = Mochi despierto
_busy     = threading.Event()    # True = procesando respuesta (no grabar)
_shutdown = threading.Event()    # True = salir

_last_interaction = 0.0

# ── Botón ──────────────────────────────────────────────────────────────────────
try:
    _button = gpiozero.Button(BUTTON_PIN, pull_up=True, bounce_time=0.1)
except Exception as e:
    print(f"⚠ Botón GPIO no disponible ({e}). Usa Enter para activar.")
    _button = None

# ── Helpers ────────────────────────────────────────────────────────────────────

def _wake():
    global _last_interaction
    if _active.is_set():
        return
    print("✨ Mochi despertando…")
    _active.set()
    _last_interaction = time.time()
    display.draw_face("excited")
    # Hablar en hilo para no bloquear
    t = threading.Thread(target=audio.speak,
                         args=("¿Qué hay?",), daemon=True)
    t.start()

def _sleep():
    print("💤 Mochi a dormir")
    _active.clear()
    reset_history()
    display.draw_face("sleeping")
    audio.speak("Hasta luego.")

# ── Conectar botón ─────────────────────────────────────────────────────────────
if _button:
    _button.when_pressed = _wake

# ── Worker de conversación ─────────────────────────────────────────────────────

def _conversation_worker(wav_path: str):
    """
    Hilo que procesa un turno completo: transcribe → chat → habla.
    Se marca _busy durante todo el proceso.
    """
    global _last_interaction

    _busy.set()
    display.draw_face("thinking")

    try:
        # 1. Transcripción
        text = transcribe(wav_path)
        if os.path.exists(wav_path):
            os.unlink(wav_path)

        if not text:
            display.draw_face("happy")
            _busy.clear()
            return

        print(f"  Tú → {text}")

        # 2. Chat
        reply, emotion = chat(text)
        print(f"Mochi [{emotion}] → {reply}")

        _last_interaction = time.time()

        # 3. Cara + animación de boca + voz
        display.draw_face(emotion)
        display.start_talking()
        audio.speak(reply)
        display.stop_talking()
        display.draw_face(emotion)

    except Exception as e:
        print(f"✗ Worker error: {e}")
        display.draw_face("nervous")
        display.stop_talking()

    finally:
        _busy.clear()

# ── Bucle principal ────────────────────────────────────────────────────────────

def _main_loop():
    global _last_interaction

    print("🤖 Mochi listo. " + ("Pulsa el botón." if _button else "Pulsa Enter para despertar."))
    display.draw_face("sleeping")

    while not _shutdown.is_set():

        # Modo espera — esperando botón (o Enter si no hay botón)
        if not _active.is_set():
            if not _button:
                try:
                    input()   # Enter en consola como fallback
                    _wake()
                except EOFError:
                    time.sleep(0.5)
            else:
                time.sleep(0.1)
            continue

        # Comprobar inactividad
        if time.time() - _last_interaction > INACTIVITY_LIMIT:
            _sleep()
            continue

        # Si el worker anterior todavía está procesando, esperamos
        if _busy.is_set():
            time.sleep(0.05)
            continue

        # ── Grabar ────────────────────────────────────────────────────────────
        display.draw_face("happy")
        wav = audio.record_audio()

        if wav is None:
            # Silencio detectado — no lanzamos worker
            elapsed = time.time() - _last_interaction
            if elapsed > INACTIVITY_LIMIT * 0.7:
                display.draw_face("sleepy")
            continue

        # Lanzar worker en hilo para no bloquear el loop
        t = threading.Thread(target=_conversation_worker, args=(wav,), daemon=True)
        t.start()

# ── Señales del SO ─────────────────────────────────────────────────────────────

def _handle_signal(sig, frame):
    print("\nApagando Mochi…")
    _shutdown.set()
    display.draw_face("sleeping")
    time.sleep(0.3)
    display.shutdown()
    audio.shutdown()
    sys.exit(0)

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _main_loop()
