import numpy as np
from collections import deque


def kmeans_custom(X, n_clusters, tol=1e-4):
    """
    Custom implementation of KMeans clustering.

    Parameters:
    - X (np.ndarray): Data points to cluster.
    - n_clusters (int): Number of clusters.
    - tol (float): Tolerance for convergence.

    Returns:
    - cluster_labels (np.ndarray): Array of cluster labels for each data point.
    - cluster_centers (np.ndarray): Coordinates of cluster centers.
    """
    cluster_centers = initialize_centroids(X, n_clusters)
    cluster_labels = np.zeros(X.shape[0])
    new_cluster_centers = np.zeros(cluster_centers.shape)
    while not check_convergence(cluster_centers, new_cluster_centers, tol):
        cluster_labels = np.argmin(sqeucdist(X, cluster_centers), axis=1)
        new_cluster_centers = np.array(
            [X[cluster_labels == i].mean(axis=0) for i in range(n_clusters)])
        cluster_centers = new_cluster_centers

    cluster_labels = balance_groups(
        X, cluster_labels, cluster_centers, n_clusters)

    return cluster_labels, cluster_centers


def initialize_centroids(X, n_clusters):
    """
    Initialize centroids for KMeans clustering.

    Parameters:
    - X (np.ndarray): Data points.
    - n_clusters (int): Number of clusters.

    Returns:
    - centroids (np.ndarray): Initialized centroids.
    """
    centroids = X[np.random.choice(X.shape[0])]
    for _ in range(n_clusters - 1):
        dist = np.min(sqeucdist(X, centroids), axis=1)
        prob = dist / np.sum(dist)
        centroids = np.vstack(
            [centroids, X[np.random.choice(X.shape[0], p=prob)]])
    return centroids


def check_convergence(cluster_centers, new_cluster_centers, tol):
    """
    Check if the centroids have converged.

    Parameters:
    - cluster_centers (np.ndarray): Current cluster centers.
    - new_cluster_centers (np.ndarray): Updated cluster centers.
    - tol (float): Tolerance for convergence.

    Returns:
    - (bool): True if converged, False otherwise.
    """

    return np.linalg.norm(cluster_centers - new_cluster_centers) < tol


def sqeucdist(p, q):
    """
    Compute the squared Euclidean distance between each pair of points in p and q.

    Parameters:
    - p (np.ndarray): Array of points.
    - q (np.ndarray): Array of points.

    Returns:
    - (np.ndarray): Squared Euclidean distances.
    """
    return np.sum((p[:, None] - q) ** 2, axis=2)


def sqeucdist_1d(p, q):
    """
    Compute the squared Euclidean distance between two points.

    Parameters:
    - p (np.ndarray): First point.
    - q (np.ndarray): Second point.

    Returns:
    - (float): Squared Euclidean distance.
    """
    return np.sum((p - q) ** 2)


def gen_group_sizes(p, n):
    """
    Generate group sizes for balancing clusters.

    Parameters:
    - p (int): Total number of points.
    - n (int): Number of clusters.

    Returns:
    - result (list): List of sizes for each cluster.
    """
    quotient = p // n
    remainder = p % n
    result = [quotient + 1] * remainder + [quotient] * (n - remainder)
    return result


def sort_cluster_centers(cluster_centers, labels):
    """
    Sort cluster centers based on the number of points assigned to them.

    Parameters:
    - cluster_centers (np.ndarray): Coordinates of cluster centers.
    - labels (np.ndarray): Cluster labels for each data point.

    Returns:
    - sorted_centers (np.ndarray): Sorted cluster centers.
    """
    # Count occurrences of labels for each cluster center
    counts = [0] * cluster_centers.shape[0]
    for num in labels:
        counts[num] += 1

    indexed_numbers = list(enumerate(counts))

    # Sort the list of tuples based on the numbers in reverse order
    sorted_indices = [index for index, _ in sorted(
        indexed_numbers, key=lambda x: x[1], reverse=True)]

    sorted_centers = deque()
    for i in sorted_indices:
        sorted_centers.append(cluster_centers[i])

    return np.array(sorted_centers)


def balance_groups(X, cluster_labels, cluster_centers, n_clusters):
    """
    Balance the number of points in each cluster.

    Parameters:
    - X (np.ndarray): Data points.
    - cluster_labels (np.ndarray): Cluster labels for each data point.
    - cluster_centers (np.ndarray): Coordinates of cluster centers.
    - n_clusters (int): Number of clusters wanted.

    Returns:
    - new_cluster_labels (np.ndarray): Balanced cluster labels.
    """
    # Define the number of points in each group
    group_sizes = gen_group_sizes(X.shape[0], n_clusters)  # TODO: need to flip
    new_cluster_labels = np.full(X.shape[0], np.inf)
    # Sort the cluster centers based on the number of points assigned to them
    sorted_centers = sort_cluster_centers(cluster_centers, cluster_labels)
    points_left = X.copy()
    # Assign the points to the clusters based on the sorted centers
    mask = np.ones(points_left.shape[0], dtype=bool)
    for idx, center in enumerate(sorted_centers):
        # Sort the points based on the distance to the current center
        points_left = X[mask]
        sorted_X = sorted(
            points_left, key=lambda point: sqeucdist_1d(center, point))
        # Assign the points to the current cluster
        for i in range(group_sizes[idx]):
            for cluster_index in np.where((X == sorted_X[i]).all(axis=1))[0]:
                # if cluster_index has been set, skip
                if new_cluster_labels[cluster_index] != np.inf:
                    continue
                new_cluster_labels[cluster_index] = np.where(
                    (cluster_centers == center).all(axis=1))[0]
                mask[cluster_index] = False
                break

    return new_cluster_labels
