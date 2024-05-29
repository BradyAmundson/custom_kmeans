import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from data_processing import load_data, preprocess_data
from clustering import kmeans_clustering, visualize_clusters_3d


def main():
    data = load_data(
        '/Users/bradyamundson/Documents/gruuperML/big5data/IPIP-FFM-data-8Nov2018 2/data-final.csv')
    X = preprocess_data(data)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, np.zeros(len(X)))

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=knn.predict(X),
                cmap='viridis', s=50, alpha=0.5)
    plt.title('K Nearest Neighbors Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster')
    plt.show()

    cluster_labels, _ = kmeans_clustering(X, n_clusters=3)
    visualize_clusters_3d(X, cluster_labels)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    cluster_labels, _ = kmeans_clustering(X_pca, n_clusters=3)
    visualize_clusters_3d(X_pca, cluster_labels)


if __name__ == "__main__":
    main()
