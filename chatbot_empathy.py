"""
chatbot_empathy.py
- Simple console chatbot demo using the TF-IDF sentiment model + hospital context
"""
import os
import joblib
import pandas as pd
import numpy as np

MODEL_DIR = "C:/Users/Bernietta/OneDrive/guvi/guvi_project/main_project/src/chatbot"
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
CLF_PATH = os.path.join(MODEL_DIR, "sentiment_clf.joblib")
HOSPITAL_DATA = "C:/Users/Bernietta/OneDrive/guvi/guvi_project/main_project/testing data/patient_satisfaction/merged_patient_satisfaction.csv"  # used for facility-level context

def load_resources():
    tfidf = joblib.load(TFIDF_PATH)
    clf = joblib.load(CLF_PATH)
    df_hosp = pd.read_csv(HOSPITAL_DATA, low_memory=False)
    # Convert Patient Survey Star Rating to numeric for aggregation
    if "Patient Survey Star Rating" in df_hosp.columns:
        df_hosp["Patient Survey Star Rating"] = pd.to_numeric(df_hosp["Patient Survey Star Rating"], errors="coerce")
        summary = df_hosp.groupby("Facility ID")["Patient Survey Star Rating"].median().reset_index().rename(columns={"Patient Survey Star Rating":"median_star"})
    else:
        summary = pd.DataFrame()
    return tfidf, clf, summary

def sentiment_to_label(s):
    return {0:"negative", 1:"neutral", 2:"positive"}.get(int(s),"neutral")

def empathetic_reply(sentiment_label, user_text, facility_name=None, facility_star=None):
    if sentiment_label == "negative":
        base = "I'm really sorry you had a poor experience. That matters. "
        if facility_name and not np.isnan(facility_star) and facility_star < 3:
            base += f"I see that {facility_name} has a lower survey rating ({facility_star} stars). Would you like me to connect you with patient relations or provide a complaint form?"
        else:
            base += "Can you tell me more so I can help or escalate this to patient relations?"
    elif sentiment_label == "neutral":
        base = "Thanks for that feedback. I can help with information or forward this to the support team. What would you like to do?"
    else:  # positive
        base = "That's great to hear — thank you! If you'd like, you can leave a review or tell us what went well so we can pass it along."
    return base

def chat_loop():
    tfidf, clf, hosp_summary = load_resources()
    print("🤖 Empathy Chatbot (type 'exit' to quit)")
    while True:
        text = input("\nYou: ").strip()
        if text.lower() in ("exit","quit"):
            break
        X = tfidf.transform([text])
        pred = clf.predict(X)[0]
        label = sentiment_to_label(pred)
        # try to detect if user mentions a facility name (naive)
        facility_name = None; facility_star = None
        for _, row in hosp_summary.iterrows():
            # naive check: facility id or name in text (we only have facility_id summary here)
            pass
        reply = empathetic_reply(label, text, facility_name, facility_star)
        print("\nBot:", reply)

if __name__ == "__main__":
    chat_loop()
