"""
healthai_multiuser_dashboard_v2.py
---------------------------------
Unified HealthAI multi-user dashboard (improved)
- Fixes TTS file deletion/playback race
- Adds Sign-up / Admin create-user with role assignment
- Adds dark/light theme toggle (runtime)
- Adds session analytics: daily chat counts chart
- Uses SQLite backend (data_multiuser/healthai.db)
- Uses modern Streamlit API (st.rerun)
"""
import sys, os
sys.path.insert(0, os.path.abspath("src"))
import os
import streamlit as st
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import tempfile
import time
import threading
import sounddevice as sd 
import speech_recognition as sr
from gtts import gTTS
import pygame
from langdetect import detect
import numpy as np
from passlib.hash import pbkdf2_sha256
import plotly.express as px

# -----------------------------
# Optional chatbot import
# -----------------------------
try:
    from health_translator.chatbot_translator import chatbot_reply
except Exception as e:
    chatbot_reply = None
    CHATBOT_IMPORT_ERROR = str(e)
else:
    CHATBOT_IMPORT_ERROR = None

# -----------------------------
# Configuration & DB paths
# -----------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data_multiuser")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "healthai.db")

# -----------------------------
# Database helpers
# -----------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'patient',
                    created_at TEXT NOT NULL
                   )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    login_time TEXT,
                    logout_time TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                   )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS chats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    timestamp TEXT,
                    lang TEXT,
                    sentiment TEXT,
                    reply TEXT,
                    user_input TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                   )""")
    conn.commit(); conn.close()

def create_default_admin_if_missing():
    conn = get_conn(); cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as c FROM users")
    if cur.fetchone()["c"] == 0:
        pw = pbkdf2_sha256.hash("admin123")
        cur.execute("INSERT INTO users (username,password_hash,role,created_at) VALUES (?,?,?,?)",
                    ("admin", pw, "admin", datetime.utcnow().isoformat()))
        conn.commit()
    conn.close()

def add_user(username, password, role="patient"):
    conn = get_conn(); cur = conn.cursor()
    pw_hash = pbkdf2_sha256.hash(password)
    try:
        cur.execute("INSERT INTO users (username,password_hash,role,created_at) VALUES (?,?,?,?)",
                    (username, pw_hash, role, datetime.utcnow().isoformat()))
        conn.commit(); uid = cur.lastrowid
    except sqlite3.IntegrityError:
        uid = None
    conn.close(); return uid

def verify_user(username, password):
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
    cur.execute("UPDATE sessions SET logout_time = ? WHERE id = ?",
                (datetime.utcnow().isoformat(), session_id))
    conn.commit(); conn.close()

def insert_chat(user_id, lang, sentiment, reply, user_input):
    conn = get_conn(); cur = conn.cursor()
    cur.execute("INSERT INTO chats (user_id, timestamp, lang, sentiment, reply, user_input) VALUES (?,?,?,?,?,?)",
                (user_id, datetime.utcnow().isoformat(), lang, sentiment, reply, user_input))
    conn.commit(); conn.close()

def fetch_chats(limit=500):
    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT c.*, u.username FROM chats c LEFT JOIN users u ON c.user_id = u.id
        ORDER BY c.timestamp DESC LIMIT ?
    """, conn, params=(limit,))
    conn.close(); return df

def fetch_users():
    conn = get_conn(); df = pd.read_sql_query("SELECT id,username,role,created_at FROM users ORDER BY created_at DESC", conn)
    conn.close(); return df

def daily_chat_counts(days=30):
    conn = get_conn()
    df = pd.read_sql_query("SELECT timestamp FROM chats", conn)
    conn.close()
    if df.empty:
        return pd.DataFrame({"date":[], "count":[]})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    min_date = (datetime.utcnow() - timedelta(days=days)).date()
    df = df[df["timestamp"].dt.date >= min_date]
    counts = df.groupby(df["timestamp"].dt.date).size().reset_index(name="count").rename(columns={"timestamp":"date"})
    counts["date"] = pd.to_datetime(counts["timestamp"]).dt.date if "timestamp" in counts.columns else counts["date"]
    # Ensure contiguous dates
    all_dates = pd.date_range(start=min_date, end=datetime.utcnow().date())
    out = pd.DataFrame({"date": all_dates})
    out["date"] = out["date"].dt.date
    counts = counts.rename(columns={"timestamp":"date"}) if "timestamp" in counts.columns else counts
    counts = counts.groupby("date")["count"].sum().reindex(out["date"], fill_value=0).reset_index()
    counts.columns = ["date","count"]
    return counts

# -----------------------------
# Init DB and default admin
# -----------------------------
init_db()
create_default_admin_if_missing()

# -----------------------------
# Utilities: language detection & safe TTS playback
# -----------------------------
def detect_lang_code(text):
    try:
        lang = detect(text)
        if lang.startswith("ta"): return "ta"
        if lang.startswith("hi"): return "hi"
        if lang.startswith("te"): return "te"
        if lang.startswith("ml"): return "ml"
        return "en"
    except Exception:
        return "en"

def speak_text_safe(text, lang="en", blocking=True):
    """
    Create an mp3 using gTTS, ensure file persists until playback finishes, then delete.
    Use pygame for playback if available. This function handles file lifecycle.
    """
    try:
        tts = gTTS(text=text, lang=lang)
    except Exception as e:
        st.warning(f"TTS generation failed: {e}")
        return

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_path = tmp.name
    tmp.close()  # ensure gTTS can write to it on Windows
    try:
        tts.save(tmp_path)
    except Exception as e:
        st.warning(f"TTS save failed: {e}")
        try:
            os.remove(tmp_path)
        except:
            pass
        return

    # Playback
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        if blocking:
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        # quit mixer to release device
        pygame.mixer.quit()
    except Exception as e:
        st.warning(f"Audio playback failed: {e}")
    finally:
        # ensure file removed
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

# -----------------------------
# Streamlit UI setup
# -----------------------------
st.set_page_config(page_title="HealthAI Multi-User Dashboard", page_icon="🩺", layout="wide")
# Theme toggle stored in session_state
if "theme" not in st.session_state:
    st.session_state.theme = "light"  # default

def inject_theme_css(theme):
    if theme == "dark":
        css = """
        <style>
        .reportview-container { background: #0e1117; color: #e6eef6; }
        .stButton>button { background-color:#1f6feb; color:white; }
        .sidebar .sidebar-content { background-color:#0b0f15; color:#e6eef6; }
        </style>
        """
    else:
        css = """
        <style>
        .reportview-container { background: white; color: black; }
        .stButton>button { background-color:#1f6feb; color:white; }
        .sidebar .sidebar-content { background-color: #f8f9fa; color: black; }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

inject_theme_css(st.session_state.theme)

# -----------------------------
# Authentication UI (sidebar)
# -----------------------------
st.sidebar.title("Account")
# toggle signup availability
if "allow_signup" not in st.session_state:
    st.session_state.allow_signup = True  # set to False to disable open signup

st.sidebar.write("Theme:")
theme_choice = st.sidebar.radio("Theme", ["light","dark"], index=0 if st.session_state.theme=="light" else 1)
if theme_choice != st.session_state.theme:
    st.session_state.theme = theme_choice
    inject_theme_css(st.session_state.theme)

# Show chatbot pipeline warning if missing
if CHATBOT_IMPORT_ERROR:
    st.sidebar.warning("Chatbot not loaded. Text/voice will use fallback heuristics.")

# Session user state
if "user" not in st.session_state:
    st.session_state.user = None
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# If not logged in, show login / signup / admin-create
if st.session_state.user is None:
    st.sidebar.subheader("Sign in / Sign up")
    login_mode = st.sidebar.selectbox("Action", ["Login", "Sign up" if st.session_state.allow_signup else "Login", "Admin: Create user (admin only)"])
    if login_mode == "Login":
        su = st.sidebar.text_input("Username", key="login_user")
        sp = st.sidebar.text_input("Password", type="password", key="login_pw")
        if st.sidebar.button("Login"):
            u = verify_user(su.strip(), sp.strip())
            if u:
                st.session_state.user = u
                st.session_state.session_id = record_session_login(u["id"])
                st.success(f"Logged in as {u['username']} ({u['role']})")
                st.rerun()
            else:
                st.sidebar.error("Invalid credentials")
    elif login_mode.startswith("Sign up"):
        st.sidebar.info("Create an account")
        new_user = st.sidebar.text_input("Choose username", key="signup_user")
        new_pw = st.sidebar.text_input("Choose password", type="password", key="signup_pw")
        role = st.sidebar.selectbox("Role", ["patient","staff"], key="signup_role")
        if st.sidebar.button("Create account"):
            if not new_user or not new_pw:
                st.sidebar.error("Username & password required")
            else:
                uid = add_user(new_user.strip(), new_pw.strip(), role=role)
                if uid:
                    st.sidebar.success("Account created. Please login.")
                else:
                    st.sidebar.error("Username already exists.")
    else:
        # Admin create user (admin credentials required)
        st.sidebar.info("Admin can create a user")
        admin_user = st.sidebar.text_input("Admin username", key="admin_user")
        admin_pw = st.sidebar.text_input("Admin password", type="password", key="admin_pw")
        cu = st.sidebar.text_input("New username", key="create_user")
        cpw = st.sidebar.text_input("New password", type="password", key="create_pw")
        crole = st.sidebar.selectbox("New user role", ["patient","staff","admin"], key="create_role")
        if st.sidebar.button("Create user (admin)"):
            admin = verify_user(admin_user.strip(), admin_pw.strip())
            if admin and admin["role"] == "admin":
                uid = add_user(cu.strip(), cpw.strip(), role=crole)
                if uid:
                    st.sidebar.success(f"Created {cu} (role={crole})")
                else:
                    st.sidebar.error("Username exists")
            else:
                st.sidebar.error("Invalid admin credentials")

    st.stop()

# If here: user logged in
user = st.session_state.user
st.sidebar.markdown(f"**Signed in as:** {user['username']} ({user['role']})")
if st.sidebar.button("Logout"):
    if st.session_state.session_id:
        record_session_logout(st.session_state.session_id)
    st.session_state.user = None
    st.session_state.session_id = None
    st.rerun()

# -----------------------------
# Main navigation
# -----------------------------
menu = st.sidebar.radio("Navigate", [
    "Dashboard", "Text Chat", "Voice Chat (single)", "Voice Chat (continuous)",
    "Emotion Meter", "Chat Logs", "User Management"
])

# -----------------------------
# Pages
# -----------------------------
if menu == "Dashboard":
    st.header("Dashboard — Analytics")
    st.markdown("Daily chat counts (last 30 days)")
    counts = daily_chat_counts(days=30)
    if counts.empty:
        st.info("No chat data yet")
    else:
        fig = px.line(counts, x="date", y="count", title="Daily chat count", markers=True)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(counts.tail(30))

    st.markdown("Recent chats")
    rch = fetch_chats(limit=200)
    st.dataframe(rch)

elif menu == "Text Chat":
    st.header("Multilingual Text Chat")
    user_input = st.text_area("Enter message", height=120)
    if st.button("Send"):
        if not user_input.strip():
            st.warning("Please enter text")
        else:
            if chatbot_reply:
                try:
                    res = chatbot_reply(user_input)
                    reply = res.get("reply","")
                    sentiment = res.get("sentiment","neutral")
                    lang = res.get("input_lang", detect_lang_code(user_input))
                except Exception as e:
                    reply = f"Error: {e}"; sentiment="neutral"; lang="en"
            else:
                txt = user_input.lower()
                if any(w in txt for w in ["rude","dirty","ignored","pain","bad","slow"]):
                    sentiment="negative"; reply="I'm sorry you had a poor experience."
                elif any(w in txt for w in ["good","great","thank","excellent","helpful"]):
                    sentiment="positive"; reply="That's wonderful to hear!"
                else:
                    sentiment="neutral"; reply="Thanks for your feedback."
                lang = detect_lang_code(user_input)

            insert_chat(user["id"], lang, sentiment, reply, user_input)
            st.success("Bot reply: " + reply)
            st.info("Sentiment: " + sentiment)

elif menu == "Voice Chat (single)":
    st.header("Voice Chat — single-shot")
    if st.button("Record & send"):
        r = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                st.info("Recording — speak now")
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio = r.listen(source, timeout=6, phrase_time_limit=8)
            text = r.recognize_google(audio)
            st.write("You said:", text)
        except Exception as e:
            st.error("Speech recognition failed: " + str(e))
            text = ""

        if text:
            if chatbot_reply:
                try:
                    res = chatbot_reply(text)
                    reply = res.get("reply","")
                    sentiment = res.get("sentiment","neutral")
                    lang = res.get("input_lang", detect_lang_code(text))
                except Exception as e:
                    reply = f"Error: {e}"; sentiment="neutral"; lang="en"
            else:
                lt = text.lower()
                if any(w in lt for w in ["rude","dirty"]):
                    sentiment="negative"; reply="I'm sorry you had that experience."
                elif any(w in lt for w in ["good","thank"]):
                    sentiment="positive"; reply="Glad to hear that!"
                else:
                    sentiment="neutral"; reply="Thanks for sharing."
                lang = detect_lang_code(text)

            insert_chat(user["id"], lang, sentiment, reply, text)
            st.success("Bot: " + reply)
            # Play audio (non-blocking)
            threading.Thread(target=speak_text_safe, args=(reply, lang, True), daemon=True).start()

elif menu == "Voice Chat (continuous)":
    st.header("Voice Chat — continuous mode")
    if "listening" not in st.session_state:
        st.session_state.listening = False
    status = st.empty()
    if not st.session_state.listening:
        if st.button("Start Listening (continuous)"):
            st.session_state.listening = True
            st.session_state.mic_stop = threading.Event()
            status.info("Listening... press Stop to end")
            def loop():
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
                        status.write("You: " + text)
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
                                sentiment="neutral"; reply="Thanks."
                            lang = detect_lang_code(text)
                        insert_chat(user["id"], lang, sentiment, reply, text)
                        status.write("Bot: " + reply)
                        speak_text_safe(reply, lang, blocking=True)
                    except sr.WaitTimeoutError:
                        continue
                    except Exception as e:
                        status.warning("Speech error: " + str(e))
                        time.sleep(0.5)
                status.warning("Stopped listening")
            threading.Thread(target=loop, daemon=True).start()
    else:
        if st.button("Stop Listening"):
            st.session_state.mic_stop.set()
            st.session_state.listening = False
            st.success("Stopped")

elif menu == "Emotion Meter":
    st.header("Emotion Gauge & Mic Energy Sample")
    demo_sent = st.selectbox("Demo sentiment", ["positive","neutral","negative"])
    mapping = {"positive":90, "neutral":50, "negative":20}
    val = mapping[demo_sent]
    fig = px.line(pd.DataFrame({"x":[0,1,2], "y":[val,val,val]}), x="x", y="y", labels={"y":"Emotion score"})
    st.plotly_chart(fig, use_container_width=True)
    if st.button("Sample mic energy"):
        try:
            samples = sd.rec(int(0.3 * 44100), samplerate=44100, channels=1, dtype="float64")
            sd.wait()
            energy = float(np.linalg.norm(samples)) * 100
            energy = min(100, int(energy))
            st.progress(energy)
            st.write(f"Mic energy ~ {energy}")
        except Exception as e:
            st.error("Mic sample failed: " + str(e))

elif menu == "Chat Logs":
    st.header("Chat Logs")
    df = fetch_chats(limit=1000)
    st.dataframe(df)
    if not df.empty:
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "chat_logs.csv")

elif menu == "User Management":
    if user["role"] != "admin":
        st.error("Admin only section")
    else:
        st.header("User Management (Admin)")
        st.subheader("Create user")
        nu = st.text_input("username", key="um_nu")
        npw = st.text_input("password", type="password", key="um_np")
        nrole = st.selectbox("role", ["patient","staff","admin"], key="um_role")
        if st.button("Create user (admin)"):
            if not nu or not npw:
                st.error("username & password required")
            else:
                uid = add_user(nu.strip(), npw.strip(), role=nrole)
                if uid:
                    st.success(f"Created {nu} (id={uid})")
                else:
                    st.error("Username exists")
        st.markdown("---")
        st.subheader("Registered users")
        users = fetch_users()
        st.dataframe(users)

# Footer
st.markdown("---")
st.caption("HealthAI — unified multi-user dashboard. Default admin: admin / admin123 (change immediately).")
