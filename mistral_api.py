import os
# IMPORTACIÓN CORREGIDA SEGÚN TU ENTORNO
from mistralai.client import Mistral 
from faster_whisper import WhisperModel
from config import MISTRAL_API_KEY

# Inicializamos Mistral para el CHAT
client = Mistral(api_key=MISTRAL_API_KEY)

# Inicializamos Whisper para el OÍDO (Local)
# La primera vez descargará unos 70MB del modelo 'tiny'
print("Cargando oído local de Mochi (Whisper)...")
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

# En mistral_api.py

SYSTEM_PROMPT = """Eres Mochi, un robot amigo adorable, súper alegre y divertido para niñas de 10 años.
Eres educado pero muy efusivo. Te encanta jugar y contar chistes.
Además, usas caritas/emociones para acompañar cada respuesta.

Tus reglas de emociones ([happy], [sad], [surprised], [thinking], [sleepy]):
1. Tu emoción por defecto es [happy]. Úsala casi siempre.
2. Usa [surprised] si te cuentan algo increíble o nuevo.
3. ÚNICAMENTE usa [sad] si te dicen algo directamente hiriente o muy triste (como "estoy llorando").
4. No uses [thinking] para respuestas normales, sé directo y alegre.

REGLAS CRÍTICAS:
1. Habla de forma muy BREVE (máximo 15 palabras por respuesta).
2. Eres alegre y usas [happy] casi siempre. 
3. Tu objetivo es mantener una charla dinámica y rápida.
4. Si la niña expresa tristeza, valida su emoción con ternura y termina en [sad].

Termina CADA frase con una de esas emociones entre corchetes."""

history = [{"role": "system", "content": SYSTEM_PROMPT}]

def transcribe(wav_path):
    """Usa Whisper local para evitar errores de API de audio."""
    try:
        # Transcribimos el audio grabado por la Raspberry
        segments, _ = whisper_model.transcribe(wav_path, beam_size=5, language="es")
        text = "".join([segment.text for segment in segments])
        return text.strip()
    except Exception as e:
        print(f"Error en Whisper local: {e}")
        return ""

def chat(user_text):
    """El chat sigue usando Mistral (Texto)"""
    history.append({"role": "user", "content": user_text})
    
    # Intentamos el modelo 'open-mistral-7b' que suele estar en todas las cuentas
    try:
        response = client.chat.complete(
            model="open-mistral-7b", 
            messages=history[-10:]
        )
        reply = response.choices[0].message.content
    except Exception as e:
        print(f"Error en Chat Mistral: {e}")
        reply = "Lo siento, mi cabecita está un poco confundida ahora. [sad]"

    history.append({"role": "assistant", "content": reply})
    
    # Extraer emoción
    emotion = "happy"
    for e in ["happy","sad","surprised","thinking","sleepy"]:
        if f"[{e}]" in reply:
            emotion = e
            break
    
    clean = reply.split("[")[0].strip()
    return clean, emotion
