"""
train_sentiment_model.py
- Trains TF-IDF + LogisticRegression to predict sentiment (0=neg,1=neu,2=pos)
- Saves model, vectorizer, label encoder to models/chatbot/
"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

IN_FILE = "C:/Users/Bernietta/OneDrive/guvi/guvi_project/main_project/src/chatbot/satisfaction_for_model.csv"
OUT_DIR = "C:/Users/Bernietta/OneDrive/guvi/guvi_project/main_project/src/chatbot"
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    print("Loading preprocessed data...")
    df = pd.read_csv(IN_FILE, low_memory=False)
    # drop rows with NaN text or sentiment
    df = df.dropna(subset=["text_input","sentiment"])
    X = df["text_input"].astype(str).values
    y = df["sentiment"].astype(int).values

    # simple split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Building TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=3)
    X_train_t = tfidf.fit_transform(X_train)
    X_val_t = tfidf.transform(X_val)

    print("Training Logistic Regression...")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X_train_t, y_train)

    print("Evaluating...")
    preds = clf.predict(X_val_t)
    print("Accuracy:", accuracy_score(y_val, preds))
    print(classification_report(y_val, preds, digits=4))

    # Save artifacts
    joblib.dump(tfidf, os.path.join(OUT_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(clf, os.path.join(OUT_DIR, "sentiment_clf.joblib"))
    print(f"Saved vectorizer and model to {OUT_DIR}")

if __name__ == "__main__":
    main()
