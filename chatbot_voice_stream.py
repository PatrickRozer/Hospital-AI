"""
Segment 10H – Real-Time Multilingual Voice Chat (HealthAI)
----------------------------------------------------------
Continuously listens, detects language, sends to chatbot_translator,
and speaks replies aloud.
"""
import sys, os
sys.path.insert(0, os.path.abspath("src"))
import os
import streamlit as st
import tempfile
import time
import threading
import speech_recognition as sr
from gtts import gTTS
import pygame
from langdetect import detect
from health_translator.chatbot_translator import chatbot_reply
from health_translator.monitor import log_interaction


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="HealthAI Voice Assistant", page_icon="🎙", layout="centered")
st.title("🎙 HealthAI – Real-Time Multilingual Voice Chat")
st.markdown("Speak naturally — I’ll listen, understand, and reply in your language 🗣")

# ---------------- Audio Helpers ----------------
def speak_in_language(text, lang_code="en"):
    """Speak text aloud using gTTS + pygame."""
    try:
        tts = gTTS(text=text, lang=lang_code)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp_path = tmp.name
            tts.save(tmp_path)

        pygame.mixer.init()
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.quit()
        os.remove(tmp_path)
    except Exception as e:
        st.warning(f"🔈 Audio error: {e}")


def detect_lang_code(text):
    """Map detected language to gTTS-compatible code."""
    try:
        lang = detect(text)
        if lang.startswith("ta"): return "ta"
        if lang.startswith("hi"): return "hi"
        if lang.startswith("te"): return "te"
        if lang.startswith("ml"): return "ml"
        return "en"
    except:
        return "en"


# ---------------- Continuous Listener ----------------
running = False

def continuous_listen_loop():
    global running
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        st.session_state["status"] = "Listening..."

    while running:
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=7)
            text = recognizer.recognize_google(audio)
            if text.strip():
                handle_user_message(text)
        except sr.WaitTimeoutError:
            continue
        except Exception as e:
            st.session_state["status"] = f"Error: {e}"
            time.sleep(1)


def handle_user_message(user_text):
    """Process text → chatbot → speak + log"""
    detected_lang = detect_lang_code(user_text)
    result = chatbot_reply(user_text)
    reply_text = result.get("reply", "Sorry, I couldn’t process that.")
    reply_lang = result.get("input_lang", detected_lang)

    st.session_state["chat"].append(
        {"user": user_text, "bot": reply_text, "lang": reply_lang, "sent": result.get("sentiment")}
    )
    log_interaction(user_text, result)
    speak_in_language(reply_text, reply_lang)


# ---------------- Streamlit Controls ----------------
if "chat" not in st.session_state:
    st.session_state["chat"] = []
if "status" not in st.session_state:
    st.session_state["status"] = "Idle"

col1, col2 = st.columns(2)

if col1.button("🎤 Start Voice Chat"):
    if not running:
        running = True
        st.session_state["status"] = "Listening..."
        thread = threading.Thread(target=continuous_listen_loop)
        thread.start()

if col2.button("🛑 Stop"):
    running = False
    st.session_state["status"] = "Stopped"

st.markdown(f"**Status:** {st.session_state['status']}")

# ---------------- Show Conversation ----------------
st.markdown("---")
for msg in reversed(st.session_state["chat"]):
    st.markdown(f"**You ({msg['lang']}):** {msg['user']}")
    st.markdown(
        f"<div style='background:#e8f4ff;padding:8px;border-radius:8px;'>"
        f"🤖 **Bot ({msg['sent']}):** {msg['bot']}</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption("© 2025 HealthAI – Segment 10H Voice Streaming Assistant")
