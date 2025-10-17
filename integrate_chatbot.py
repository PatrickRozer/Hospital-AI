"""
integrate_chatbot.py
- Shows how to call translator.translate() around existing chatbot pipeline
- Example: detect language -> translate user input to English -> run sentiment model -> translate reply back
"""
import sys, os
sys.path.insert(0, os.path.abspath("src"))

from health_translator.translator import translate, detect_language
import joblib
import os
import sentencepiece as spm

# paths for existing chatbot artifacts
TFIDF_PATH = "C:/Users/Bernietta/OneDrive/guvi/guvi_project/main_project/src/chatbot/tfidf_vectorizer.joblib"
CLF_PATH = "C:/Users/Bernietta/OneDrive/guvi/guvi_project/main_project/src/chatbot/sentiment_clf.joblib"

_tfidf, _clf = None, None
def load_chatbot_models():
    global _tfidf, _clf
    if _tfidf is None or _clf is None:
        if not os.path.exists(TFIDF_PATH) or not os.path.exists(CLF_PATH):
            raise FileNotFoundError("Chatbot models missing. Run training pipeline first.")
        _tfidf = joblib.load(TFIDF_PATH)
        _clf = joblib.load(CLF_PATH)
    return _tfidf, _clf

def sentiment_to_label(s):
    return {0:"negative", 1:"neutral", 2:"positive"}.get(int(s),"neutral")

def empathetic_reply(sentiment_label, facility_name=None, facility_star=None):
    if sentiment_label == "negative":
        base = "I'm really sorry you had a poor experience. That matters. Can I help escalate this to patient relations?"
    elif sentiment_label == "neutral":
        base = "Thanks for the feedback. How can I assist you further?"
    else:
        base = "That's great to hear — thank you! Would you like to share more details?"
    return base

def chat_once(user_text):
    # 1️⃣ Detect language and translate to English
    result = translate(user_text, target_lang="en")
    if isinstance(result, tuple):
        src_lang, text_en = result
    else:
        src_lang = detect_language(user_text)
        text_en = result

    print(f"[Detected: {src_lang}] Translated → EN: {text_en}")

    # 2️⃣ Run sentiment model
    tfidf, clf = load_chatbot_models()
    X = tfidf.transform([text_en])
    pred = clf.predict(X)[0]
    label = sentiment_to_label(pred)

    # 3️⃣ Generate empathetic reply in English
    reply_en = empathetic_reply(label)

    # 4️⃣ Translate reply back to user's language
    reply_local = reply_en if src_lang == "en" else translate(reply_en, target_lang=src_lang)

    return {"detected_lang": src_lang, "user_en": text_en, "sentiment": label, "reply": reply_local}


if __name__ == "__main__":
    print("Example: multilingual chat integration")
    user = input("User text: ")
    out = chat_once(user)
    print("Bot reply:", out["reply"])
