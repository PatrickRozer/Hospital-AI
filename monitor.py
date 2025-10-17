"""
monitor.py
----------------------------------------------------
Universal logger for chatbot interactions.
Ensures every log entry includes: timestamp, language, sentiment, reply, and user_input.
Creates CSV automatically if missing.
"""

import csv
import os
from datetime import datetime
import pandas as pd

# Unified log path
LOG_FILE = "C:/Users/Bernietta/OneDrive/guvi/guvi_project/main_project/src/chatbot/chatbot_logs.csv"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def log_interaction(user_input: str, result: dict):
    """
    Logs a single chatbot interaction safely with all columns.
    """
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

def read_logs():
    """
    Loads the log CSV safely.
    If missing or incomplete, fixes structure automatically.
    """
    if not os.path.exists(LOG_FILE):
        # Create empty file with correct columns
        df = pd.DataFrame(columns=["timestamp", "lang", "sentiment", "reply", "user_input"])
        df.to_csv(LOG_FILE, index=False)
        return df

    try:
        df = pd.read_csv(LOG_FILE)
    except Exception:
        df = pd.DataFrame(columns=["timestamp", "lang", "sentiment", "reply", "user_input"])

    # Ensure missing columns are filled in
    for col in ["timestamp", "lang", "sentiment", "reply", "user_input"]:
        if col not in df.columns:
            df[col] = None

    return df
