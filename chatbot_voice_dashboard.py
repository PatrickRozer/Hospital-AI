"""
chatbot_voice_dashboard.py
------------------------------------------------------
HealthAI Multilingual Chatbot with Voice I/O:
- Speech-to-Text input (microphone)
- Text-to-Speech output (gTTS)
- Works with your chatbot_translator + monitor
"""

import streamlit as st
import pandas as pd
import os
import tempfile
import speech_recognition as sr
from gtts import gTTS
from datetime import datetime
from playsound import playsound

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_FILE = os.path.join(ROOT, "C:/Users/Bernietta/OneDrive/guvi/guvi_project/main_project/src/chatbot/chatbot_logs.csv")

# Import chatbot + monitor
try:
    from health_translator.chatbot_translator import chatbot_reply
except Exception as e:
    chatbot_reply = None
    CHATBOT_ERROR = str(e)
else:
    CHATBOT_ERROR = None

try:
    from health_translator.monitor import log_interaction
except Exception:
    import csv
    def log_interaction(user_input, result):
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "lang": result.get("input_lang", "unknown"),
            "sentiment": result.get("sentiment", "neutral"),
            "reply": result.get("reply", ""),
            "user_input": user_input
        }
        write_header = not os.path.exists(LOG_FILE)
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

# -------------------------------------------------------
# PAGE SETUP
# -------------------------------------------------------
st.set_page_config(page_title="HealthAI Voice Chatbot", page_icon="🎙", layout="centered")
st.title("🎙 HealthAI – Voice-Enabled Chatbot")

st.markdown("""
🩺 **Talk to your HealthAI assistant!**  
You can type or speak in any language —  
it will translate, analyze your sentiment, and reply empathetically.
""")

# -------------------------------------------------------
# STATE
# -------------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

# -------------------------------------------------------
# SPEECH-TO-TEXT
# -------------------------------------------------------
def record_and_transcribe():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        st.info("🎤 Listening... please speak clearly.")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=5)
    try:
        st.info("⏳ Recognizing speech...")
        text = recognizer.recognize_google(audio)
        st.success(f"🗣 You said: {text}")
        return text
    except sr.UnknownValueError:
        st.warning("Could not understand audio.")
        return None
    except Exception as e:
        st.error(f"Speech recognition error: {e}")
        return None

# -------------------------------------------------------
# TEXT-TO-SPEECH
# -------------------------------------------------------
def speak_text(text, lang_code="en"):
    try:
        tts = gTTS(text=text, lang=lang_code)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            temp_path = tmp.name
            tts.save(temp_path)
        # Use Windows default player to play in background
        os.system(f'start /min wmplayer "{temp_path}"')
    except Exception as e:
        st.warning(f"Audio playback failed: {e}")

# -------------------------------------------------------
# USER INPUT
# -------------------------------------------------------
col1, col2 = st.columns(2)
user_text = col1.text_input("💬 Type your message here:")
if col2.button("🎤 Speak"):
    transcribed = record_and_transcribe()
    if transcribed:
        user_text = transcribed

if st.button("Send"):
    if not user_text.strip():
        st.warning("Please enter or speak a message.")
    else:
        if chatbot_reply:
            try:
                result = chatbot_reply(user_text)
            except Exception as e:
                result = {
                    "input_lang": "en",
                    "sentiment": "neutral",
                    "reply": f"⚠️ Error: {e}"
                }
        else:
            text = user_text.lower()
            if any(w in text for w in ["bad", "rude", "dirty", "terrible", "pain"]):
                sent, rep = "negative", "I'm sorry to hear that. We'll improve!"
            elif any(w in text for w in ["good", "great", "excellent", "thank", "happy"]):
                sent, rep = "positive", "That's wonderful to hear — thank you!"
            else:
                sent, rep = "neutral", "Thanks for the feedback. How can I assist you further?"
            result = {"input_lang": "en", "sentiment": sent, "reply": rep}

        log_interaction(user_text, result)

        st.session_state["history"].append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "user": user_text,
            "bot": result["reply"],
            "sentiment": result["sentiment"]
        })

        # Speak bot reply
        speak_text(result["reply"], lang_code=result.get("input_lang", "en"))

# -------------------------------------------------------
# DISPLAY CHAT
# -------------------------------------------------------
st.markdown("### 💬 Conversation History")
colors = {"positive": "#DCFCE7", "neutral": "#FEF9C3", "negative": "#FEE2E2"}

for chat in reversed(st.session_state["history"]):
    st.markdown(f"🧍 **You:** {chat['user']}")
    st.markdown(
        f"<div style='background:{colors.get(chat['sentiment'],'#E5E7EB')}; padding:8px; border-radius:8px;'>"
        f"🤖 **Bot ({chat['sentiment']}):** {chat['bot']}</div>",
        unsafe_allow_html=True,
    )
    st.caption(chat["time"])

# -------------------------------------------------------
# ANALYTICS
# -------------------------------------------------------
st.markdown("---")
st.header("📊 Sentiment Analytics")

if os.path.exists(LOG_FILE):
    df = pd.read_csv(LOG_FILE)
    if "sentiment" in df.columns:
        st.bar_chart(df["sentiment"].value_counts())
        st.write("Total interactions:", len(df))
    else:
        st.warning("Sentiment column missing in logs.")
else:
    st.info("No logs found yet.")

st.markdown("---")
st.caption("© 2025 HealthAI – Voice Enabled Chatbot")
