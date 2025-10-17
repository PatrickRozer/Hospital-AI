import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.clustering.preprocess_clustering import load_and_build_clustering
from src.clustering.models_clustering import run_kmeans, run_hdbscan, plot_clusters

MODEL_DIR = "models/clustering"
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    X, df_full = load_and_build_clustering()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---- KMeans ----
    kmeans, labels_kmeans = run_kmeans(X_scaled, n_clusters=4)
    df_full["cluster_kmeans"] = labels_kmeans
    print("KMeans cluster sizes:\n", df_full["cluster_kmeans"].value_counts())

    # ---- HDBSCAN ----
    hdbscan_model, labels_hdb = run_hdbscan(X_scaled, min_cluster_size=50)
    df_full["cluster_hdbscan"] = labels_hdb
    print("HDBSCAN cluster sizes:\n", df_full["cluster_hdbscan"].value_counts())

    # ---- Save results ----
    df_full.to_csv(os.path.join(MODEL_DIR, "patient_clusters.csv"), index=False)
    print("✅ Saved patient clusters to models/clustering/patient_clusters.csv")

    # ---- Plot clusters ----
    plot_clusters(X_scaled, labels_kmeans, title="KMeans Clusters")
    plot_clusters(X_scaled, labels_hdb, title="HDBSCAN Clusters")

if __name__ == "__main__":
    main()
