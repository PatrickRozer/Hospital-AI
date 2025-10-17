"""
healthai_multiuser_dashboard.py
--------------------------------
Unified multi-user HealthAI Streamlit portal (Segment 10J+):
- SQLite DB for users, sessions, chats
- Login / Logout / (optional) signup
- Role-based admin panel
- Text & voice multilingual chatbot integration (uses chatbot_translator.chatbot_reply)
- Emotion gauge and energy meter
- Exports and analytics

Paths & DB:
- DB file created at DATA_DIR/healthai.db
"""
import sys, os
sys.path.insert(0, os.path.abspath("src"))
import os
import streamlit as st
import sqlite3
from datetime import datetime
import pandas as pd
import tempfile
import time
import threading
import speech_recognition as sr
from gtts import gTTS
import pygame
from langdetect import detect
import sounddevice as sd
import numpy as np
from passlib.hash import pbkdf2_sha256

# Optional import of your chatbot pipeline
try:
    from health_translator.chatbot_translator import chatbot_reply
except Exception as e:
    chatbot_reply = None
    CHATBOT_IMPORT_ERROR = str(e)
else:
    CHATBOT_IMPORT_ERROR = None


# ---------------------------
# CONFIGURATION
# ---------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data_multiuser")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "healthai.db")


# ---------------------------
# DATABASE SETUP
# ---------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'patient',
        created_at TEXT NOT NULL
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        login_time TEXT,
        logout_time TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        timestamp TEXT,
        lang TEXT,
        sentiment TEXT,
        reply TEXT,
        user_input TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)
    conn.commit()
    conn.close()


def create_default_admin():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as c FROM users")
    if cur.fetchone()["c"] == 0:
        pw = pbkdf2_sha256.hash("admin123")
        cur.execute("INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                    ("admin", pw, "admin", datetime.utcnow().isoformat()))
        conn.commit()
    conn.close()


def verify_user(username, password):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, password_hash, role FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if row and pbkdf2_sha256.verify(password, row["password_hash"]):
        return {"id": row["id"], "username": username, "role": row["role"]}
    return None


def add_user(username, password, role="patient"):
    conn = get_conn()
    cur = conn.cursor()
    pw_hash = pbkdf2_sha256.hash(password)
    try:
        cur.execute("INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                    (username, pw_hash, role, datetime.utcnow().isoformat()))
        conn.commit()
        uid = cur.lastrowid
    except sqlite3.IntegrityError:
        uid = None
    conn.close()
    return uid


def record_session_login(user_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO sessions (user_id, login_time) VALUES (?, ?)",
                (user_id, datetime.utcnow().isoformat()))
    conn.commit()
    sid = cur.lastrowid
    conn.close()
    return sid


def record_session_logout(session_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE sessions SET logout_time = ? WHERE id = ?",
                (datetime.utcnow().isoformat(), session_id))
    conn.commit()
    conn.close()


def insert_chat(user_id, lang, sentiment, reply, user_input):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO chats (user_id, timestamp, lang, sentiment, reply, user_input) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, datetime.utcnow().isoformat(), lang, sentiment, reply, user_input))
    conn.commit()
    conn.close()


def fetch_chats(limit=500):
    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT c.*, u.username
        FROM chats c LEFT JOIN users u ON c.user_id = u.id
        ORDER BY c.timestamp DESC
        LIMIT ?
    """, conn, params=(limit,))
    conn.close()
    return df


def fetch_users():
    conn = get_conn()
    df = pd.read_sql_query("SELECT id, username, role, created_at FROM users ORDER BY created_at DESC", conn)
    conn.close()
    return df


# Initialize database
init_db()
create_default_admin()


# ---------------------------
# UTILITIES
# ---------------------------
def detect_lang_code(text):
    try:
        lang = detect(text)
        return lang if lang in ["ta", "hi", "ml", "te"] else "en"
    except Exception:
        return "en"


def speak_text_local(text, lang="en"):
    try:
        tts = gTTS(text=text, lang=lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tts.save(tmp.name)
            pygame.mixer.init()
            pygame.mixer.music.load(tmp.name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            pygame.mixer.quit()
        os.remove(tmp.name)
    except Exception as e:
        st.warning(f"Audio error: {e}")


# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="HealthAI Portal", page_icon="🩺", layout="wide")
st.title("🩺 HealthAI Multi-User Dashboard")

if "user" not in st.session_state:
    st.session_state.user = None
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# LOGIN PANEL
if st.session_state.user is None:
    st.sidebar.header("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        user = verify_user(username, password)
        if user:
            st.session_state.user = user
            st.session_state.session_id = record_session_login(user["id"])
            st.success(f"✅ Logged in as {user['username']} ({user['role']})")
            st.rerun()  # ✅ Updated API
        else:
            st.sidebar.error("❌ Invalid credentials")

    st.stop()

# LOGGED-IN UI
user = st.session_state.user
st.sidebar.markdown(f"**Logged in as:** {user['username']} ({user['role']})")
if st.sidebar.button("Logout"):
    if st.session_state.session_id:
        record_session_logout(st.session_state.session_id)
    st.session_state.user = None
    st.session_state.session_id = None
    st.rerun()  # ✅ Updated API

# Sidebar menu
menu = st.sidebar.radio("Navigation", [
    "Dashboard",
    "Text Chat",
    "Voice Chat",
    "Emotion Meter",
    "Chat Logs",
    "Admin Panel"
])

# ---------------------------
# PAGES
# ---------------------------

if menu == "Dashboard":
    st.header("📊 Dashboard Overview")
    df = fetch_chats(limit=200)
    if df.empty:
        st.info("No chat data available yet.")
    else:
        st.dataframe(df[["username", "sentiment", "user_input", "reply", "timestamp"]])
        st.bar_chart(df["sentiment"].value_counts())

elif menu == "Text Chat":
    st.header("💬 Multilingual Text Chat")
    msg = st.text_area("Enter your message")
    if st.button("Send"):
        if chatbot_reply:
            try:
                result = chatbot_reply(msg)
                reply = result["reply"]
                sentiment = result["sentiment"]
                lang = result["input_lang"]
            except Exception as e:
                reply, sentiment, lang = str(e), "neutral", "en"
        else:
            # fallback logic
            msg_lower = msg.lower()
            if any(x in msg_lower for x in ["rude", "dirty", "bad"]):
                sentiment, reply = "negative", "I'm sorry about your experience."
            elif any(x in msg_lower for x in ["good", "excellent", "thank"]):
                sentiment, reply = "positive", "That's great to hear!"
            else:
                sentiment, reply = "neutral", "Thank you for your feedback."
            lang = detect_lang_code(msg)

        insert_chat(user["id"], lang, sentiment, reply, msg)
        st.success(f"🤖 {reply}")
        st.info(f"Sentiment: {sentiment}")

elif menu == "Voice Chat":
    st.header("🎙️ Voice Chat")
    if st.button("Record & Process"):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Recording... Speak now.")
            audio = r.listen(source, timeout=5, phrase_time_limit=6)
        try:
            text = r.recognize_google(audio)
            st.write("🗣️ You said:", text)
        except Exception:
            st.error("Speech not recognized.")
            text = ""

        if text:
            result = chatbot_reply(text) if chatbot_reply else {"reply": "Voice reply", "sentiment": "neutral", "input_lang": "en"}
            insert_chat(user["id"], result["input_lang"], result["sentiment"], result["reply"], text)
            st.success(f"🤖 {result['reply']}")
            speak_text_local(result["reply"], lang=result["input_lang"])

elif menu == "Emotion Meter":
    st.header("💓 Emotion and Energy Meter")
    sentiment = st.selectbox("Select sentiment", ["positive", "neutral", "negative"])
    energy = np.random.randint(30, 90)
    st.metric(label="Energy Level", value=f"{energy}%")
    st.progress(energy / 100)
    st.info(f"Detected sentiment: {sentiment}")

elif menu == "Chat Logs":
    st.header("📚 Chat Logs")
    df = fetch_chats(limit=500)
    st.dataframe(df)
    st.download_button("Download Logs CSV", df.to_csv(index=False), "chat_logs.csv")

elif menu == "Admin Panel":
    if user["role"] != "admin":
        st.error("🚫 Admin access required.")
    else:
        st.header("👨‍💼 Admin Panel")
        st.subheader("Registered Users")
        st.dataframe(fetch_users())
        st.subheader("Recent Chats")
        st.dataframe(fetch_chats(limit=100))
        st.subheader("Add New User")
        new_username = st.text_input("New Username")