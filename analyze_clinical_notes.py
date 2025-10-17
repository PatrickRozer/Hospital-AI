"""
Use pretrained BioBERT / ClinicalBERT to embed and classify clinical notes.
Outputs embeddings and example classification.
"""
import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
os.environ["OMP_NUM_THREADS"] = "4"

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
DATA_FILE = os.path.join("C:/Users/Bernietta/OneDrive", "guvi/guvi_project/main_project/src/pretrained_models", "cleaned_notes.csv")
OUT_DIR = os.path.join("models", "pretrained")
os.makedirs(OUT_DIR, exist_ok=True)

def load_model():
    print(f"🔹 Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    return tokenizer, model

def compute_embeddings(texts, tokenizer, model, batch_size=8):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch = texts[i:i+batch_size]
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            outputs = model(**tokens)
            # mean pooling over tokens
            emb = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(emb.cpu())
    return torch.cat(embeddings)

def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Cleaned notes not found. Run preprocess_text.py first.")

    df = pd.read_csv(DATA_FILE)
    texts = df["clean_text"].tolist()

    tokenizer, model = load_model()
    print("🔹 Computing embeddings for clinical notes...")
    note_embeddings = compute_embeddings(texts, tokenizer, model)
    print(f"✅ Embeddings shape: {note_embeddings.shape}")

    emb_file = os.path.join(OUT_DIR, "note_embeddings.pt")
    torch.save(note_embeddings, emb_file)
    print(f"✅ Saved embeddings to {emb_file}")

    # Example: clustering or classification
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(note_embeddings)
    df["cluster"] = kmeans.labels_
    df.to_csv(os.path.join(OUT_DIR, "clustered_notes.csv"), index=False)
    print(f"✅ Saved clustered notes with labels → {OUT_DIR}/clustered_notes.csv")

if __name__ == "__main__":
    main()
