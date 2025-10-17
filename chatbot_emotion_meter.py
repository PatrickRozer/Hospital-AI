"""
Segment 10I – Real-Time Emotion Gauge & Voice Energy Meter
----------------------------------------------------------
Extends HealthAI voice assistant with:
  • Live mic input energy display  
  • Emotion intensity gauge from sentiment
"""
import sys, os
sys.path.insert(0, os.path.abspath("src"))
import os, tempfile, threading, time
import numpy as np
import streamlit as st
import sounddevice as sd
import plotly.graph_objects as go
from gtts import gTTS
from langdetect import detect
import speech_recognition as sr
import pygame
from health_translator.chatbot_translator import chatbot_reply
from health_translator.monitor import log_interaction


# -------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------
st.set_page_config(page_title="HealthAI Emotion Voice Assistant", page_icon="💬", layout="wide")
st.title("💬 HealthAI – Voice Assistant with Emotion Meter")

col1, col2 = st.columns([2, 1])
status_box = st.empty()
gauge_box = col2.empty()
energy_box = col2.empty()

# -------------------------------------------------------
# UTILITIES
# -------------------------------------------------------
def detect_lang_code(text):
    try:
        lang = detect(text)
        return lang if lang in ["en","ta","hi","te","ml"] else "en"
    except:
        return "en"

def speak_text(text, lang="en"):
    try:
        tts = gTTS(text=text, lang=lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tts.save(tmp.name)
            pygame.mixer.init()
            pygame.mixer.music.load(tmp.name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy(): time.sleep(0.1)
            pygame.mixer.quit()
            os.remove(tmp.name)
    except Exception as e:
        st.warning(f"🔈 Audio error: {e}")

# -------------------------------------------------------
# LIVE VOICE ENERGY METER
# -------------------------------------------------------
def listen_energy(stop_flag):
    energy = 0
    while not stop_flag.is_set():
        try:
            samples = sd.rec(int(0.2 * 44100), samplerate=44100, channels=1, dtype="float64")
            sd.wait()
            energy = float(np.linalg.norm(samples)) * 100
            energy_box.progress(min(100, int(energy)))
        except:
            pass
        time.sleep(0.1)

# -------------------------------------------------------
# EMOTION GAUGE
# -------------------------------------------------------
def show_emotion_gauge(sentiment):
    colors = {"negative":"red","neutral":"gold","positive":"green"}
    value = {"negative":20,"neutral":50,"positive":90}.get(sentiment, 50)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        gauge={"axis":{"range":[0,100]}, "bar":{"color":colors[sentiment]}},
        title={"text":f"Emotion: {sentiment.upper()}"}
    ))
    gauge_box.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------
# LISTEN AND RESPOND
# -------------------------------------------------------
def process_voice_once():
    rec = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        status_box.info("🎤 Listening...")
        rec.adjust_for_ambient_noise(source)
        audio = rec.listen(source, timeout=5)
    try:
        text = rec.recognize_google(audio)
        status_box.success(f"🗣 You said: {text}")
        result = chatbot_reply(text)
        log_interaction(text, result)
        show_emotion_gauge(result.get("sentiment","neutral"))
        speak_text(result["reply"], result.get("input_lang","en"))
    except Exception as e:
        status_box.warning(f"⚠️ Speech error: {e}")

# -------------------------------------------------------
# STREAMLIT BUTTONS
# -------------------------------------------------------
if "stop_flag" not in st.session_state:
    st.session_state.stop_flag = threading.Event()

colA, colB = st.columns(2)

if colA.button("🎙 Start Voice Assistant"):
    st.session_state.stop_flag.clear()
    threading.Thread(target=listen_energy, args=(st.session_state.stop_flag,), daemon=True).start()
    process_voice_once()

if colB.button("🛑 Stop Voice Assistant"):
    st.session_state.stop_flag.set()
    status_box.warning("Assistant Stopped")

st.caption("© 2025 HealthAI Segment 10I – Emotion & Voice Energy Meter")
