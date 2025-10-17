"""
chatbot_multilingual_voice.py
---------------------------------------------------
HealthAI Multilingual Voice Chatbot (Segment 10G)
- Detects spoken or typed language
- Replies empathetically using chatbot_translator.py
- Speaks reply aloud in the detected language
"""
import sys, os
sys.path.insert(0, os.path.abspath("src"))
import os
import streamlit as st
import tempfile
import speech_recognition as sr
from gtts import gTTS
import pygame
from datetime import datetime
from langdetect import detect
from health_translator.chatbot_translator import chatbot_reply
from health_translator.monitor import log_interaction

# -------------------------------------------------------
# SETTINGS
# -------------------------------------------------------
st.set_page_config(page_title="HealthAI Multilingual Voice Chat", page_icon="🌍", layout="centered")
st.title("🌍 HealthAI — Multilingual Voice Chatbot")

st.markdown("""
🩺 Speak or type in your language.  
HealthAI will translate, understand your sentiment, reply empathetically — and **talk back** in your language.  
Supports **English, Tamil, Hindi, Telugu, and more!**
""")

# -------------------------------------------------------
# FUNCTIONS
# -------------------------------------------------------
def record_audio():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        st.info("🎤 Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=6)
    try:
        st.info("🎧 Processing speech...")
        text = recognizer.recognize_google(audio)
        st.success(f"🗣 You said: {text}")
        return text
    except Exception as e:
        st.warning(f"⚠️ Speech error: {e}")
        return None


def speak_in_language(text, lang_code="en"):
    """Use gTTS + pygame to speak the reply in target language."""
    try:
        tts = gTTS(text=text, lang=lang_code)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp_path = tmp.name
            tts.save(tmp_path)

        pygame.mixer.init()
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue
        pygame.mixer.quit()
        os.remove(tmp_path)
    except Exception as e:
        st.warning(f"🔈 Playback failed: {e}")


def detect_lang_code(text):
    """Detect language and return gTTS compatible code."""
    try:
        lang = detect(text)
        if lang.startswith("ta"): return "ta"   # Tamil
        if lang.startswith("hi"): return "hi"   # Hindi
        if lang.startswith("te"): return "te"   # Telugu
        if lang.startswith("ml"): return "ml"   # Malayalam
        return "en"
    except:
        return "en"

# -------------------------------------------------------
# CHAT SECTION
# -------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

col1, col2 = st.columns([3, 1])
user_text = col1.text_input("💬 Type your message:")
if col2.button("🎙 Speak"):
    spoken = record_audio()
    if spoken:
        user_text = spoken

if st.button("Send"):
    if user_text.strip():
        detected_lang = detect_lang_code(user_text)
        st.info(f"Detected language: {detected_lang.upper()}")

        result = chatbot_reply(user_text)
        reply_text = result.get("reply", "Sorry, I couldn't process that.")
        reply_lang = result.get("input_lang", detected_lang)

        log_interaction(user_text, result)
        st.session_state.chat_history.append(
            {"user": user_text, "bot": reply_text, "lang": reply_lang, "sent": result.get("sentiment")}
        )

        speak_in_language(reply_text, lang_code=reply_lang)
    else:
        st.warning("Please enter or speak a message.")

# -------------------------------------------------------
# DISPLAY CHAT
# -------------------------------------------------------
st.markdown("---")
st.header("💬 Conversation History")

for chat in reversed(st.session_state.chat_history):
    st.markdown(f"**You ({chat['lang']}):** {chat['user']}")
    st.markdown(
        f"<div style='background:#f0f8ff; padding: 8px; border-radius: 8px;'>"
        f"🤖 **Bot ({chat['sent']}):** {chat['bot']}</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption("© 2025 HealthAI – Multilingual Voice Chatbot v10G")
