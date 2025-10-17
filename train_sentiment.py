"""
Segment 10: Sentiment Analysis
Analyze patient satisfaction survey text using traditional ML + BERT.
"""

import os
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
import joblib

DATA_PATH = "C:/Users/Bernietta/OneDrive/guvi/guvi_project/main_project/testing data/patient_satisfaction/merged_patient_satisfaction.csv"
MODEL_DIR = "C:/Users/Bernietta/OneDrive/guvi/guvi_project/main_project/src/sentiment"
os.makedirs(MODEL_DIR, exist_ok=True)

def clean_text(text):
    """Basic cleaning for text"""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    # Combine question and answer text
    df["text"] = df["HCAHPS Question"].astype(str) + " " + df["HCAHPS Answer Description"].astype(str)
    df["clean_text"] = df["text"].apply(clean_text)
    
    # Map star rating to sentiment labels
    df["Patient Survey Star Rating"] = pd.to_numeric(df["Patient Survey Star Rating"], errors="coerce")
    df = df.dropna(subset=["Patient Survey Star Rating"])
    df["sentiment"] = df["Patient Survey Star Rating"].apply(lambda x: 
        "positive" if x >= 4 else "neutral" if x == 3 else "negative")
    
    print(df["sentiment"].value_counts())
    return df[["clean_text", "sentiment"]]

def train_tfidf_model():
    df = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        df["clean_text"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"]
    )
    
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_vec, y_train)
    
    preds = clf.predict(X_val_vec)
    print("✅ TF-IDF Model Evaluation")
    print(classification_report(y_val, preds))
    
    # Save models
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(clf, os.path.join(MODEL_DIR, "sentiment_clf.joblib"))
    print("✅ Saved TF-IDF model + classifier in", MODEL_DIR)

if __name__ == "__main__":
    train_tfidf_model()
