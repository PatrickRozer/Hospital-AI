"""
Visualize Bio_ClinicalBERT embeddings of clinical notes using PCA and t-SNE.
Generates 2D scatter plots colored by cluster.
"""
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

MODEL_DIR = os.path.join("C:/Users/Bernietta/OneDrive", "guvi/guvi_project/main_project/src/pretrained_models")
EMB_FILE = os.path.join(MODEL_DIR, "note_embeddings.pt")
CLUSTER_FILE = os.path.join(MODEL_DIR, "clustered_notes.csv")
OUT_DIR = os.path.join(MODEL_DIR, "plots")

os.makedirs(OUT_DIR, exist_ok=True)

def main():
    if not os.path.exists(EMB_FILE) or not os.path.exists(CLUSTER_FILE):
        raise FileNotFoundError("Missing embeddings or clustered notes. Run analyze_clinical_notes.py first.")

    print("🔹 Loading embeddings and cluster assignments...")
    embeddings = torch.load(EMB_FILE).numpy()
    df = pd.read_csv(CLUSTER_FILE)

    if "cluster" not in df.columns:
        raise ValueError("Cluster column missing in clustered_notes.csv")

    # --- PCA ---
    print("🔹 Computing PCA...")
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(embeddings)

    plt.figure(figsize=(8,6))
    plt.scatter(pca_result[:,0], pca_result[:,1], c=df["cluster"], cmap="tab10", s=15, alpha=0.7)
    plt.colorbar(label="Cluster ID")
    plt.title("Bio_ClinicalBERT Embeddings (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "embeddings_pca.png"))
    print(f"✅ Saved PCA plot → {OUT_DIR}/embeddings_pca.png")

    # --- t-SNE ---
    print("🔹 Computing t-SNE (this may take ~2–5 min)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
    tsne_result = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8,6))
    plt.scatter(tsne_result[:,0], tsne_result[:,1], c=df["cluster"], cmap="tab10", s=15, alpha=0.7)
    plt.colorbar(label="Cluster ID")
    plt.title("Bio_ClinicalBERT Embeddings (t-SNE 2D)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "embeddings_tsne.png"))
    print(f"✅ Saved t-SNE plot → {OUT_DIR}/embeddings_tsne.png")

if __name__ == "__main__":
    main()
