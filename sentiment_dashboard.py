"""
sentiment_dashboard.py
Streamlit dashboard for Patient Sentiment Analysis.
Requires: src/chatbot/cleaned_patient_feedback.csv
Optional: models/sentiment_bert/ (DistilBERT), or models/sentiment/ (TF-IDF)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# -------------------- Config --------------------
DATA_FILE = os.path.join("C:/Users/Bernietta/OneDrive/guvi/guvi_project/main_project/src/chatbot/cleaned_patient_feedback.csv")
BERT_DIR = os.path.join("C:/Users/Bernietta/OneDrive/guvi/guvi_project/main_project/src/sentiment/sentiment_bert")   # optional
TFIDF_DIR = os.path.join("C:/Users/Bernietta/OneDrive/guvi/guvi_project/main_project/src/sentiment")      # optional
MAX_PREVIEW = 200

# -------------------- Utilities --------------------
@st.cache_data
def load_df(path):
    df = pd.read_csv(path, low_memory=False)
    return df

@st.cache_resource
def load_bert_model(model_dir=BERT_DIR):
    if not os.path.exists(model_dir):
        return None
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    le = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))
    model.eval()
    return tokenizer, model, le

@st.cache_resource
def load_tfidf(tfidf_dir=TFIDF_DIR):
    if not os.path.exists(tfidf_dir):
        return None
    vect = joblib.load(os.path.join(tfidf_dir, "tfidf_vectorizer.joblib"))
    clf = joblib.load(os.path.join(tfidf_dir, "sentiment_clf.joblib"))
    return vect, clf

def predict_with_bert(texts, tokenizer, model, le):
    inputs = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
    return [le.classes_[p] for p in preds]

def predict_with_tfidf(texts, vect, clf):
    X = vect.transform(texts)
    preds = clf.predict(X)
    return preds

# -------------------- App --------------------
st.set_page_config(page_title="Patient Sentiment Dashboard", layout="wide")
st.title("Patient Sentiment Dashboard")

# Load data
if not os.path.exists(DATA_FILE):
    st.error(f"Data file not found: {DATA_FILE}. Run the data preparation step first.")
    st.stop()

df = load_df(DATA_FILE)
st.sidebar.header("Options")

# Detect text and label columns if not named correctly
if "text" not in df.columns:
    text_candidates = [c for c in df.columns if any(k in c.lower() for k in ["text","review","feedback","comment","response","description"])]
    if text_candidates:
        df.rename(columns={text_candidates[0]:"text"}, inplace=True)

if "sentiment" not in df.columns:
    label_candidates = [c for c in df.columns if any(k in c.lower() for k in ["sentiment","rating","score","label","category"])]
    if label_candidates:
        df.rename(columns={label_candidates[0]:"sentiment"}, inplace=True)

if "text" not in df.columns:
    st.error("No text column found in dataset; please ensure a column with patient feedback exists.")
    st.stop()
if "sentiment" not in df.columns:
    st.warning("No sentiment column available in the dataset. Dashboard will still allow live predictions.")

st.markdown("**Data preview:**")
st.dataframe(df.head(MAX_PREVIEW))

# Load models (if available)
bert = load_bert_model()
tfidf = load_tfidf()

st.sidebar.subheader("Model selection")
model_choice = st.sidebar.selectbox("Use model for live prediction", ["DistilBERT (if available)", "TF-IDF (if available)", "No model (just view)"])

# Live prediction input
st.header("Try Live Prediction")
user_text = st.text_area("Paste patient feedback here", height=120)
if st.button("Predict"):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        if model_choice.startswith("DistilBERT") and bert:
            tokenizer, model, le = bert
            pred = predict_with_bert([user_text], tokenizer, model, le)[0]
            st.success(f"Predicted (BERT): **{pred}**")
        elif model_choice.startswith("TF-IDF") and tfidf:
            vect, clf = tfidf
            pred = predict_with_tfidf([user_text], vect, clf)[0]
            st.success(f"Predicted (TF-IDF): **{pred}**")
        else:
            st.info("No model available. Please train / place models in models/sentiment_bert or models/sentiment.")

# Sidebar filters
st.sidebar.header("Filters")
if "sentiment" in df.columns:
    unique_sentiments = sorted(df["sentiment"].dropna().astype(str).unique().tolist())
else:
    unique_sentiments = []
selected_sent = st.sidebar.multiselect("Sentiments", unique_sentiments, default=unique_sentiments)
search_text = st.sidebar.text_input("Search text contains (optional)")

filtered = df.copy()
if selected_sent:
    filtered = filtered[filtered["sentiment"].isin(selected_sent)]
if search_text.strip():
    filtered = filtered[filtered["text"].str.contains(search_text, case=False, na=False)]

st.header("Sentiment distribution")
if "sentiment" in filtered.columns and not filtered.empty:
    counts = filtered["sentiment"].value_counts()
    st.bar_chart(counts)
else:
    st.info("No labeled sentiments to show distribution.")

# Word clouds safely
st.header("Word Clouds (by sentiment)")
if "sentiment" in df.columns:
    for s in unique_sentiments:
        subset_texts = filtered[filtered["sentiment"]==s]["text"].dropna().astype(str)
        if subset_texts.empty:
            st.info(f"No texts for '{s}'")
            continue
        combined = " ".join(subset_texts)
        if not combined.strip():
            st.info(f"Skipping {s} — no words")
            continue
        wc = WordCloud(width=800, height=400, background_color="white").generate(combined)
        st.subheader(s.capitalize())
        fig, ax = plt.subplots(figsize=(10,4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

# Download filtered data
st.download_button("Download filtered data CSV", filtered.to_csv(index=False).encode("utf-8"), "filtered_feedback.csv", "text/csv")

st.info("If you want real-time chat integrated here, we can add a chat box that pushes messages to the logs and updates visuals.")
