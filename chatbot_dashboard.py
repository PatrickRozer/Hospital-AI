"""
chatbot_dashboard.py
--------------------------------------------------
Streamlit dashboard for the HealthAI multilingual empathy chatbot.
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_FILE = os.path.join(ROOT, "C:/Users/Bernietta/OneDrive/guvi/guvi_project/main_project/src/chatbot/chatbot_logs.csv")

# --- Safe imports for chatbot + monitor -------------------
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
        """Fallback logger (creates log file if needed)."""
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "lang": result.get("input_lang", "unknown"),
            "sentiment": result.get("sentiment", "neutral"),
            "reply": result.get("reply", ""),
            "user_input": user_input,
        }
        write_header = not os.path.exists(LOG_FILE)
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

# ----------------------------------------------------------
# STREAMLIT PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(page_title="HealthAI Chatbot", page_icon="💬", layout="centered")
st.title("💬 HealthAI — Multilingual Empathy Chatbot")

st.markdown("""
🩺 **Type a message** in any language —  
our chatbot will translate, analyze sentiment, and respond empathetically.
""")

# ----------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

# ----------------------------------------------------------
# USER INPUT AREA
# ----------------------------------------------------------
user_msg = st.text_input("Your Message:", placeholder="Type something...")

if st.button("Send"):
    if not user_msg.strip():
        st.warning("Please enter a message.")
    else:
        # --- Chatbot reply ---
        if chatbot_reply:
            try:
                result = chatbot_reply(user_msg)
            except Exception as e:
                result = {
                    "input_lang": "en",
                    "sentiment": "neutral",
                    "reply": f"⚠️ Error: {e}"
                }
        else:
            # fallback sentiment logic
            text = user_msg.lower()
            if any(w in text for w in ["bad", "rude", "dirty", "terrible", "pain", "slow"]):
                sent, rep = "negative", "I'm really sorry to hear that. We'll improve!"
            elif any(w in text for w in ["good", "great", "excellent", "helpful", "kind", "thank"]):
                sent, rep = "positive", "That's wonderful to hear — thank you!"
            else:
                sent, rep = "neutral", "Thanks for the feedback. How can I assist you further?"
            result = {"input_lang": "en", "sentiment": sent, "reply": rep}

        # --- Log the chat ---
        try:
            log_interaction(user_msg, result)
        except Exception as e:
            st.error(f"Logging failed: {e}")

        # --- Save to session history ---
        st.session_state["history"].append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "user": user_msg,
            "bot": result["reply"],
            "sentiment": result["sentiment"]
        })

# ----------------------------------------------------------
# DISPLAY CHAT HISTORY
# ----------------------------------------------------------
st.markdown("### 💬 Conversation History")

if not st.session_state["history"]:
    st.info("Start chatting above to see messages here.")
else:
    colors = {"positive": "#DCFCE7", "neutral": "#FEF9C3", "negative": "#FEE2E2"}
    for chat in reversed(st.session_state["history"]):
        st.markdown(f"🧍 **You:** {chat['user']}")
        st.markdown(
            f"<div style='background:{colors.get(chat['sentiment'],'#E5E7EB')}; "
            f"padding:8px; border-radius:8px;'>🤖 **Bot ({chat['sentiment']}):** {chat['bot']}</div>",
            unsafe_allow_html=True
        )
        st.caption(chat["time"])

# ----------------------------------------------------------
# ANALYTICS SECTION
# ----------------------------------------------------------
st.markdown("---")
st.header("📊 Chat Analytics")

if os.path.exists(LOG_FILE):
    try:
        df = pd.read_csv(LOG_FILE)
        if "sentiment" not in df.columns:
            st.warning("⚠️ Log file exists but missing 'sentiment' column.")
        else:
            counts = df["sentiment"].fillna("unknown").value_counts()
            st.bar_chart(counts)
            st.write("Total interactions:", len(df))
    except Exception as e:
        st.error(f"Error reading log file: {e}")
else:
    st.info("No chat logs yet. Start chatting to generate logs.")

st.markdown("---")
st.caption("© 2025 HealthAI – Multilingual Empathy Chatbot")
