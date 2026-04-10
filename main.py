import os, time, gpiozero
from display import draw_face, show_talking
from audio import record_audio, speak
from mistral_api import transcribe, chat
from config import BUTTON_PIN

# Configurar el botón
button = gpiozero.Button(BUTTON_PIN, pull_up=True)

active_mode = False
last_interaction = 0
INACTIVITY_LIMIT = 120 # 2 minutos

print("🤖 Mochi en modo espera...")
draw_face("sleepy")

while True:
    try:
        # MODO ESPERA: Esperando que alguien pulse el botón
        if not active_mode:
            if button.is_pressed:
                print("✨ Mochi despertando...")
                draw_face("surprised")
                speak("¡Hola! Ya estoy despierto. ¿Jugamos?")
                active_mode = True
                last_interaction = time.time()
            else:
                time.sleep(0.1)
                continue

        # MODO ACTIVO: Mochi está escuchando o pensando
        draw_face("happy") # Cara de "te escucho"
        wav_path = record_audio()
        
        # Inmediatamente después de grabar, ponemos cara de pensar para ganar tiempo
        draw_face("thinking")
        
        try:
            text = transcribe(wav_path)
            if os.path.exists(wav_path):
                os.unlink(wav_path)
        except Exception as e:
            print(f"Error: {e}")
            text = ""

        if text and text.strip():
            # ¡Hay conversación!
            print(f"Tú: {text}")
            last_interaction = time.time()

            # Mistral genera respuesta y emoción
            reply, emotion = chat(text)
            print(f"Mochi [{emotion}]: {reply}")

            # ACTUALIZAMOS CARA ANTES DE HABLAR
            draw_face(emotion)
            
            # Hablar (show_talking debería ser una función rápida o un hilo)
            speak(reply)
            
        else:
            # No se escuchó nada en esta vuelta
            current_time = time.time()
            if current_time - last_interaction > INACTIVITY_LIMIT:
                print("💤 Mochi se duerme")
                speak("Me voy a descansar un ratito. ¡Pulsa mi botón si me necesitas!")
                draw_face("sleepy")
                active_mode = False
            else:
                # Si no ha pasado el tiempo límite, sigue atento pero con cara neutra
                draw_face("thinking")
                time.sleep(0.5)

    except KeyboardInterrupt:
        draw_face("sleepy")
        break
