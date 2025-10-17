import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

DATA_FILE = "C:/Users/Bernietta/OneDrive/guvi/guvi_project/main_project/testing data/faq_corpus.csv"
OUT_DIR = "C:/Users/Bernietta/OneDrive/guvi/guvi_project/main_project/src/chatbot"
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    df = pd.read_csv(DATA_FILE)
    model = SentenceTransformer('all-MiniLM-L6-v2')  # lightweight, works offline

    print("🔹 Encoding FAQ questions...")
    embeddings = model.encode(df['Question'].tolist(), convert_to_numpy=True, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, os.path.join(OUT_DIR, "faq_index.faiss"))

    np.save(os.path.join(OUT_DIR, "faq_questions.npy"), df['Question'].values)
    df.to_csv(os.path.join(OUT_DIR, "faq_with_answers.csv"), index=False)

    print(f"✅ Saved FAISS index and FAQ data to {OUT_DIR}")

if __name__ == "__main__":
    main()
