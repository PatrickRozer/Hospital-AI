"""
chatbot_translator.py
--------------------------------------------------
Multilingual Empathy-Driven Chatbot
- Detects input language
- Translates message → English
- Predicts sentiment using retrained model
- Responds empathetically
- Translates reply back to user's language
- Logs all interactions via monitor.py
--------------------------------------------------
"""
import sys, os
sys.path.insert(0, os.path.abspath("src"))
import os
import joblib
import numpy as np
from health_translator.translator import translate, detect_language
from health_translator.monitor import log_interaction

# -------------------------------------------------------------------------
# ✅ Load paths for retrained sentiment model and TF-IDF vectorizer
# -------------------------------------------------------------------------
TFIDF_PATH = "C:/Users/Bernietta/OneDrive/guvi/guvi_project/main_project/src/sentiment/tfidf_vectorizer.joblib"
CLF_PATH   = "C:/Users/Bernietta/OneDrive/guvi/guvi_project/main_project/src/sentiment/sentiment_clf.joblib"

# -------------------------------------------------------------------------
# ✅ Load the trained sentiment model
# -------------------------------------------------------------------------
def load_chatbot_models():
    if not os.path.exists(TFIDF_PATH) or not os.path.exists(CLF_PATH):
        raise FileNotFoundError("❌ Sentiment model or TF-IDF vectorizer not found. Please retrain first.")
    tfidf = joblib.load(TFIDF_PATH)
    clf = joblib.load(CLF_PATH)
    print("✅ Loaded sentiment model and TF-IDF vectorizer successfully.")
    return tfidf, clf


# -------------------------------------------------------------------------
# ✅ Convert numeric or string prediction to standardized sentiment label
# -------------------------------------------------------------------------
def sentiment_to_label(pred):
    if isinstance(pred, (int, np.integer)):
        return {0: "negative", 1: "neutral", 2: "positive"}.get(int(pred), "neutral")
    pred = str(pred).strip().lower()
    if "neg" in pred:
        return "negative"
    elif "pos" in pred:
        return "positive"
    elif "neu" in pred:
        return "neutral"
    else:
        return "neutral"


# -------------------------------------------------------------------------
# ✅ Generate empathetic chatbot response
# -------------------------------------------------------------------------
def empathetic_reply(sentiment_label):
    if sentiment_label == "negative":
        return "I'm really sorry you had a poor experience. That matters. Can I help escalate this to patient relations?"
    elif sentiment_label == "neutral":
        return "Thanks for sharing your feedback. How can I assist you further?"
    elif sentiment_label == "positive":
        return "That’s wonderful to hear! We truly appreciate your kind words and trust."
    else:
        return "Thanks for your feedback. How can we assist you today?"


# -------------------------------------------------------------------------
# ✅ Main chatbot function
# -------------------------------------------------------------------------
def chatbot_reply(user_text: str):
    # 1️⃣ Detect language and translate to English
    src_lang, text_en = translate(user_text, target_lang="en")
    print(f"[Detected: {src_lang}] Translated → EN: {text_en}")

    # 2️⃣ Load sentiment model
    tfidf, clf = load_chatbot_models()
    X = tfidf.transform([text_en])
    pred = clf.predict(X)[0]
    label = sentiment_to_label(pred)
    print(f"🧠 Predicted sentiment: {label}")

    # 3️⃣ Generate empathetic reply (in English)
    reply_en = empathetic_reply(label)

    # 4️⃣ Translate back to user’s language (if not English)
    if src_lang != "en":
        _, reply_local = translate(reply_en, target_lang=src_lang)
    else:
        reply_local = reply_en

    # 5️⃣ Save conversation log
    result = {
        "input_lang": src_lang,
        "user_input_en": text_en,
        "sentiment": label,
        "reply": reply_local
    }
    log_interaction(user_text, result)

    return result


# -------------------------------------------------------------------------
# ✅ Test loop for interactive chat
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n💬 HealthAI Multilingual Chatbot (with new Sentiment Model)")
    print("Type your message (or 'exit' to quit):\n")
    while True:
        user = input("👤 You: ")
        if user.lower() in ["exit", "quit"]:
            print("👋 Goodbye! Take care.")
            break
        response = chatbot_reply(user)
        print(f"🤖 Bot ({response['sentiment']}): {response['reply']}\n")
