from sklearn.cluster import KMeans
import hdbscan
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def run_kmeans(X, n_clusters=4):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X)
    return model, labels

def run_hdbscan(X, min_cluster_size=50):
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = model.fit_predict(X)
    return model, labels

def plot_clusters(X, labels, title="Clusters"):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    plt.figure(figsize=(6,6))
    plt.scatter(X_2d[:,0], X_2d[:,1], c=labels, cmap="tab10", s=10)
    plt.title(title)
    plt.show()
