"""
Preprocess clinical notes for BioBERT / ClinicalBERT analysis.
Extract text from NOTEEVENTS.csv, clean it, and sample manageable size for embeddings.
"""
import os
import pandas as pd
import re

DATA_FILE = os.path.join("C:/Users/Bernietta/OneDrive", "guvi/guvi_project/main_project/testing data", "NOTEEVENTS_sorted.csv")
OUT_FILE = os.path.join("C:/Users/Bernietta/OneDrive", "guvi/guvi_project/main_project/src/pretrained_models", "cleaned_notes.csv")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s.,]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def preprocess_notes(sample_size=1000):
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"NOTEEVENTS.csv not found at {DATA_FILE}")

    print("🔹 Loading notes...")
    df = pd.read_csv(DATA_FILE, usecols=["ROW_ID", "CATEGORY", "TEXT"], low_memory=False)

    print(f"✅ Loaded {len(df)} notes. Sampling {sample_size} for embedding generation...")
    df = df.sample(n=min(sample_size, len(df)), random_state=42)
    df["clean_text"] = df["TEXT"].apply(clean_text)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    df.to_csv(OUT_FILE, index=False)
    print(f"✅ Saved cleaned sample to {OUT_FILE}")
    return OUT_FILE

if __name__ == "__main__":
    preprocess_notes()
