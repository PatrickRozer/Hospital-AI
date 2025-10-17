"""
healthai_multiuser_dashboard_v3.py
---------------------------------
Final v3 unified HealthAI Streamlit dashboard.

Features:
 - Multi-user sign in / sign up / admin-create (roles: admin, staff, patient)
 - SQLite persistence for users, sessions, chats (data_multiuser/healthai.db)
 - Text chat (integrates optional health_translator.chatbot_translator.chatbot_reply)
 - Voice chat (single-shot + continuous fallback)
 - TTS using gTTS + pygame with safe temp-file lifecycle
 - Mic energy sampling via sounddevice with robust fallback
 - Dark/Light theme toggle
 - Daily chat counts chart (last 30 days)
 - All code defensive for missing libraries / hardware
"""
import sys, os
sys.path.insert(0, os.path.abspath("src"))
import os
import sqlite3
import tempfile
import threading
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# audio / STT / TTS imports (some are optional; guard with try/except)
try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    import pygame
except Exception:
    pygame = None

# sounddevice for mic energy (optional)
try:
    import sounddevice as sd
except Exception:
    sd = None

# language detect (optional)
try:
    from langdetect import detect
except Exception:
    detect = None

# password hashing
try:
    from passlib.hash import pbkdf2_sha256
except Exception:
    pbkdf2_sha256 = None

# Optional chatbot pipeline (your existing module)
try:
    from health_translator.chatbot_translator import chatbot_reply
except Exception:
    chatbot_reply = None

# ---------- Config ----------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data_multiuser")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "healthai.db")

# ---------- Database helpers ----------
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

def create_default_admin_if_missing():
    if pbkdf2_sha256 is None:
        return
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as c FROM users")
    c = cur.fetchone()["c"]
    if c == 0:
        pw = pbkdf2_sha256.hash("admin123")
        cur.execute("INSERT INTO users (username,password_hash,role,created_at) VALUES (?,?,?,?)",
                    ("admin", pw, "admin", datetime.utcnow().isoformat()))
        conn.commit()
    conn.close()

def add_user(username, password, role="patient"):
    if pbkdf2_sha256 is None:
        return None
    conn = get_conn(); cur = conn.cursor()
    pw_hash = pbkdf2_sha256.hash(password)
    try:
        cur.execute("INSERT INTO users (username,password_hash,role,created_at) VALUES (?,?,?,?)",
                    (username, pw_hash, role, datetime.utcnow().isoformat()))
        conn.commit()
        uid = cur.lastrowid
    except sqlite3.IntegrityError:
        uid = None
    conn.close(); return uid

def verify_user(username, password):
    if pbkdf2_sha256 is None:
        return None
    conn = get_conn(); cur = conn.cursor()
    cur.execute("SELECT id, password_hash, role FROM users WHERE username = ?", (username,))
    row = cur.fetchone(); conn.close()
    if row and pbkdf2_sha256.verify(password, row["password_hash"]):
        return {"id": row["id"], "username": username, "role": row["role"]}
    return None

def record_session_login(user_id):
    conn = get_conn(); cur = conn.cursor()
    cur.execute("INSERT INTO sessions (user_id, login_time) VALUES (?, ?)",
                (user_id, datetime.utcnow().isoformat()))
    conn.commit(); sid = cur.lastrowid; conn.close(); return sid

def record_session_logout(session_id):
    conn = get_conn(); cur = conn.cursor()
    cur.execute("UPDATE sessions SET logout_time = ? WHERE id = ?", (datetime.utcnow().isoformat(), session_id))
    conn.commit(); conn.close()

def insert_chat(user_id, lang, sentiment, reply, user_input):
    conn = get_conn(); cur = conn.cursor()
    cur.execute("INSERT INTO chats (user_id, timestamp, lang, sentiment, reply, user_input) VALUES (?,?,?,?,?,?)",
                (user_id, datetime.utcnow().isoformat(), lang, sentiment, reply, user_input))
    conn.commit(); conn.close()

def fetch_chats(limit=500):
    conn = get_conn()
    df = pd.read_sql_query("SELECT c.*, u.username FROM chats c LEFT JOIN users u ON c.user_id = u.id ORDER BY c.timestamp DESC LIMIT ?",
                           conn, params=(limit,))
    conn.close(); return df

def fetch_users():
    conn = get_conn()
    df = pd.read_sql_query("SELECT id,username,role,created_at FROM users ORDER BY created_at DESC", conn)
    conn.close(); return df

def daily_chat_counts(days=30):
    conn = get_conn()
    df = pd.read_sql_query("SELECT timestamp FROM chats", conn)
    conn.close()
    if df.empty:
        return pd.DataFrame({"date": [], "count": []})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    min_date = (datetime.utcnow() - timedelta(days=days)).date()
    df = df[df["timestamp"].dt.date >= min_date]
    counts = df.groupby(df["timestamp"].dt.date).size().rename("count").reset_index()
    counts.columns = ["date", "count"]
    all_dates = pd.date_range(start=min_date, end=datetime.utcnow().date())
    all_df = pd.DataFrame({"date": all_dates})
    all_df["date"] = all_df["date"].dt.date
    counts = counts.set_index("date").reindex(all_df["date"], fill_value=0).reset_index()
    counts.columns = ["date", "count"]
    return counts

# ---------- Initialize DB ----------
init_db()
create_default_admin_if_missing()

# ---------- Utilities ----------
def detect_lang_code(text):
    if detect is None:
        return "en"
    try:
        lang = detect(text)
        return lang if lang else "en"
    except Exception:
        return "en"

def speak_text_safe(text, lang="en", blocking=True):
    """
    Safe TTS: create a persistent temp file, play via pygame if available.
    The file is removed after playback completes.
    """
    if gTTS is None:
        st.warning("gTTS not available for TTS.")
        return
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_path = tmp.name
    tmp.close()
    try:
        tts = gTTS(text=text, lang=lang)
        tts.save(tmp_path)
    except Exception as e:
        st.warning(f"TTS generation failed: {e}")
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return

    # Playback
    if pygame is None:
        st.warning("pygame not installed. Skipping playback.")
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return

    try:
        pygame.mixer.init()
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        if blocking:
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        pygame.mixer.quit()
    except Exception as e:
        st.warning(f"Audio playback failed: {e}")
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

def safe_mic_energy_sample(duration_s=0.3):
    """
    Returns integer energy 0-100.
    Uses sounddevice if available, otherwise returns simulated value.
    """
    if sd is None:
        return np.random.randint(30, 85), "simulated"
    try:
        samples = sd.rec(int(duration_s * 44100), samplerate=44100, channels=1, dtype="float64")
        sd.wait()
        energy = float(np.linalg.norm(samples)) * 100
        energy = min(100, int(energy))
        return energy, "realtime"
    except Exception:
        return np.random.randint(30, 85), "simulated"

# ---------- Streamlit UI ----------
st.set_page_config(page_title="HealthAI Multi-User Dashboard v3", page_icon="🩺", layout="wide")
st.title("🩺 HealthAI — Multi-User Dashboard (v3)")

# Theme toggle: simple CSS injection
if "theme" not in st.session_state:
    st.session_state.theme = "light"

def inject_theme(theme):
    if theme == "dark":
        css = """
        <style>
        html, body, .block-container { background: #0b0f14; color: #e6eef6; }
        .stButton>button { background-color:#1f6feb; color:white; }
        .sidebar .sidebar-content { background-color:#071019; color:#e6eef6; }
        </style>
        """
    else:
        css = """
        <style>
        html, body, .block-container { background: white; color: black; }
        .stButton>button { background-color:#1f6feb; color:white; }
        .sidebar .sidebar-content { background-color:#f8f9fa; color:black; }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

inject_theme(st.session_state.theme)

# Sidebar: Account + theme
st.sidebar.title("Account")
st.sidebar.write("Theme:")
theme_choice = st.sidebar.radio("Theme:", ["light", "dark"], index=0 if st.session_state.theme=="light" else 1)
if theme_choice != st.session_state.theme:
    st.session_state.theme = theme_choice
    inject_theme(st.session_state.theme)

# show chatbot import status
if chatbot_reply is None:
    st.sidebar.warning("Optional chatbot pipeline not loaded — app will use simple heuristics.")

# session state for user
if "user" not in st.session_state:
    st.session_state.user = None
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# show login/signup/admin create when not signed in
if st.session_state.user is None:
    st.sidebar.subheader("Sign in / Sign up")
    choice = st.sidebar.selectbox("Action", ["Login", "Sign up (open)", "Admin: Create user (admin only)"])

    if choice == "Login":
        uname = st.sidebar.text_input("Username", key="login_user")
        pwd = st.sidebar.text_input("Password", type="password", key="login_pw")
        if st.sidebar.button("Login"):
            user = verify_user(uname.strip(), pwd.strip())
            if user:
                st.session_state.user = user
                st.session_state.session_id = record_session_login(user["id"])
                st.success(f"Logged in as {user['username']} ({user['role']})")
                st.rerun()
            else:
                st.sidebar.error("Invalid credentials")

    elif choice.startswith("Sign up"):
        su = st.sidebar.text_input("Choose username", key="signup_user")
        sp = st.sidebar.text_input("Choose password", type="password", key="signup_pw")
        role = st.sidebar.selectbox("Role", ["patient", "staff"], key="signup_role")
        if st.sidebar.button("Create account"):
            if not su or not sp:
                st.sidebar.error("Provide username and password.")
            else:
                uid = add_user(su.strip(), sp.strip(), role=role)
                if uid:
                    st.sidebar.success("Account created. Please login.")
                else:
                    st.sidebar.error("Username already exists.")

    else:
        # admin create user
        admin_un = st.sidebar.text_input("Admin username", key="adm_user")
        admin_pw = st.sidebar.text_input("Admin password", type="password", key="adm_pw")
        newu = st.sidebar.text_input("New username", key="adm_newu")
        newpw = st.sidebar.text_input("New password", type="password", key="adm_newpw")
        newrole = st.sidebar.selectbox("New user role", ["patient", "staff", "admin"], key="adm_newrole")
        if st.sidebar.button("Create user (admin)"):
            admin = verify_user(admin_un.strip(), admin_pw.strip())
            if admin and admin["role"] == "admin":
                uid = add_user(newu.strip(), newpw.strip(), role=newrole)
                if uid:
                    st.sidebar.success(f"User {newu} created.")
                else:
                    st.sidebar.error("Username exists.")
            else:
                st.sidebar.error("Invalid admin credentials.")
    st.stop()

# logged in UI
user = st.session_state.user
st.sidebar.markdown(f"**Signed in as:** {user['username']} ({user['role']})")
if st.sidebar.button("Logout"):
    if st.session_state.session_id:
        record_session_logout(st.session_state.session_id)
    st.session_state.user = None
    st.session_state.session_id = None
    st.rerun()

# navigation
menu = st.sidebar.radio("Navigate:", ["Dashboard", "Text Chat", "Voice Chat (single)", "Voice Chat (continuous)", "Emotion Meter", "Chat Logs", "User Management"])

# ---------- Pages ----------
if menu == "Dashboard":
    st.header("Dashboard — Analytics")
    st.markdown("Daily chat counts (last 30 days)")
    counts = daily_chat_counts(days=30)
    if counts.empty:
        st.info("No chat data yet.")
    else:
        fig = px.line(counts, x="date", y="count", title="Daily chat counts", markers=True)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(counts.tail(30))

    st.markdown("Recent chats")
    st.dataframe(fetch_chats(limit=200))

elif menu == "Text Chat":
    st.header("Multilingual Text Chat")
    input_text = st.text_area("Enter message", height=140)
    if st.button("Send"):
        if not input_text.strip():
            st.warning("Type something.")
        else:
            if chatbot_reply is not None:
                try:
                    res = chatbot_reply(input_text)
                    reply = res.get("reply", "")
                    sentiment = res.get("sentiment", "neutral")
                    lang = res.get("input_lang", detect_lang_code(input_text))
                except Exception as e:
                    reply = f"Error: {e}"; sentiment = "neutral"; lang = detect_lang_code(input_text)
            else:
                # simple heuristic fallback
                txt = input_text.lower()
                if any(w in txt for w in ["rude", "dirty", "ignored", "pain", "worst", "bad"]):
                    sentiment = "negative"; reply = "I'm sorry you had a bad experience."
                elif any(w in txt for w in ["good", "great", "thank", "excellent", "helpful"]):
                    sentiment = "positive"; reply = "That's wonderful to hear — thank you!"
                else:
                    sentiment = "neutral"; reply = "Thanks for the feedback."
                lang = detect_lang_code(input_text)

            insert_chat(user["id"], lang, sentiment, reply, input_text)
            st.success("Bot reply: " + reply)
            st.info("Sentiment: " + sentiment)

elif menu == "Voice Chat (single)":
    st.header("Voice Chat — single-shot")
    st.write("This uses your microphone to record a short snippet and returns a reply.")
    if sr is None:
        st.error("speech_recognition library is not installed — voice chat unavailable.")
    else:
        if st.button("Record & Send"):
            recognizer = sr.Recognizer()
            try:
                with sr.Microphone() as source:
                    st.info("Recording — please speak now")
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = recognizer.listen(source, timeout=6, phrase_time_limit=8)
                text = recognizer.recognize_google(audio)
                st.write("You said:", text)
            except Exception as e:
                st.error("Speech capture failed: " + str(e))
                text = ""
            if text:
                if chatbot_reply is not None:
                    try:
                        res = chatbot_reply(text)
                        reply = res.get("reply", "")
                        sentiment = res.get("sentiment", "neutral")
                        lang = res.get("input_lang", detect_lang_code(text))
                    except Exception as e:
                        reply = f"Error: {e}"; sentiment = "neutral"; lang = detect_lang_code(text)
                else:
                    lt = text.lower()
                    if any(w in lt for w in ["rude", "dirty"]):
                        sentiment = "negative"; reply = "Sorry you faced that."
                    elif any(w in lt for w in ["good", "thank"]):
                        sentiment = "positive"; reply = "Glad to hear that!"
                    else:
                        sentiment = "neutral"; reply = "Thanks for sharing."
                    lang = detect_lang_code(text)
                insert_chat(user["id"], lang, sentiment, reply, text)
                st.success("Bot: " + reply)
                # play TTS in background thread to avoid blocking UI
                threading.Thread(target=speak_text_safe, args=(reply, lang, True), daemon=True).start()

elif menu == "Voice Chat (continuous)":
    st.header("Voice Chat — continuous (press Start, then Stop)")
    if sr is None:
        st.error("speech_recognition not installed — continuous voice chat is unavailable.")
    else:
        if "listening" not in st.session_state:
            st.session_state.listening = False
        if not st.session_state.listening:
            if st.button("Start Listening"):
                st.session_state.listening = True
                st.session_state.mic_stop = threading.Event()
                st.info("Listening... press Stop to end.")
                def loop_listen():
                    r = sr.Recognizer()
                    mic = sr.Microphone()
                    with mic as source:
                        r.adjust_for_ambient_noise(source, duration=0.5)
                    while not st.session_state.mic_stop.is_set():
                        try:
                            with mic as source:
                                audio = r.listen(source, timeout=5, phrase_time_limit=6)
                            text = r.recognize_google(audio)
                            if not text:
                                continue
                            # compute response
                            if chatbot_reply:
                                res = chatbot_reply(text)
                                reply = res.get("reply",""); sentiment = res.get("sentiment","neutral"); lang = res.get("input_lang", detect_lang_code(text))
                            else:
                                lt = text.lower()
                                if any(w in lt for w in ["rude","dirty"]):
                                    sentiment="negative"; reply="Sorry you had a poor experience."
                                elif any(w in lt for w in ["good","thank"]):
                                    sentiment="positive"; reply="Glad to hear that!"
                                else:
                                    sentiment="neutral"; reply="Thanks for sharing."
                                lang = detect_lang_code(text)
                            insert_chat(user["id"], lang, sentiment, reply, text)
                            st.write("You:", text)
                            st.write("Bot:", reply)
                            speak_text_safe(reply, lang, blocking=True)
                        except Exception:
                            time.sleep(0.2)
                    st.info("Stopped listening.")
                threading.Thread(target=loop_listen, daemon=True).start()
        else:
            if st.button("Stop Listening"):
                st.session_state.mic_stop.set()
                st.session_state.listening = False
                st.success("Stopped listening.")

elif menu == "Emotion Meter":
    st.header("Emotion Meter & Mic Energy")
    demo_sent = st.selectbox("Demo sentiment", ["positive", "neutral", "negative"])
    mapping = {"positive": 90, "neutral": 50, "negative": 20}
    st.metric(label="Emotion score (demo)", value=mapping[demo_sent])
    if st.button("Sample mic energy"):
        energy, source = safe_mic_energy_sample(duration_s=0.25)
        if source == "simulated":
            st.warning("Microphone unavailable — using simulated energy value.")
        st.progress(min(1.0, energy/100))
        st.write(f"Mic energy ~ {energy} (source: {source})")

elif menu == "Chat Logs":
    st.header("Chat Logs")
    df = fetch_chats(limit=2000)
    st.dataframe(df)
    if not df.empty:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="chat_logs.csv")

elif menu == "User Management":
    if user["role"] != "admin":
        st.error("Admin access required.")
    else:
        st.header("User Management (Admin)")
        st.subheader("Create user")
        newu = st.text_input("Username", key="um_newu")
        newpw = st.text_input("Password", type="password", key="um_newpw")
        newrole = st.selectbox("Role", ["patient","staff","admin"], key="um_role")
        if st.button("Create user"):
            if not newu or not newpw:
                st.error("Provide username & password.")
            else:
                uid = add_user(newu.strip(), newpw.strip(), role=newrole)
                if uid:
                    st.success(f"User {newu} created (id={uid}).")
                else:
                    st.error("Username already exists.")
        st.markdown("---")
        st.subheader("All users")
        st.dataframe(fetch_users())

# Footer
st.markdown("---")
st.caption("HealthAI Multi-User Dashboard v3 — default admin (first-run): admin / admin123 — change immediately.")
