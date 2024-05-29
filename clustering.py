from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from collections import deque


def kmeans_clustering(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    return kmeans.labels_, kmeans.cluster_centers_


def kmeans_custom(X, n_clusters, tol=1e-4):
    cluster_centers = initialize_centroids(X, n_clusters)
    cluster_labels = np.zeros(X.shape[0])
    new_cluster_centers = np.zeros(cluster_centers.shape)
    while not check_convergence(cluster_centers, new_cluster_centers, tol):
        cluster_labels = np.argmin(sqeucdist(X, cluster_centers), axis=1)
        new_cluster_centers = np.array(
            [X[cluster_labels == i].mean(axis=0) for i in range(n_clusters)])
        cluster_centers = new_cluster_centers

    cluster_labels = balance_groups_3(
        X, cluster_labels, cluster_centers, n_clusters)

    return cluster_labels, cluster_centers


def initialize_centroids(X, n_clusters):
    centroids = X[np.random.choice(X.shape[0])]
    for _ in range(n_clusters - 1):
        dist = np.min(sqeucdist(X, centroids), axis=1)
        prob = dist / np.sum(dist)
        centroids = np.vstack(
            [centroids, X[np.random.choice(X.shape[0], p=prob)]])
    return centroids


def check_convergence(cluster_centers, new_cluster_centers, tol):
    return np.linalg.norm(cluster_centers - new_cluster_centers) < tol


def sqeucdist(p, q):
    return np.sum((p[:, None] - q) ** 2, axis=2)


def sqeucdist_1d(p, q):
    return np.sum((p - q) ** 2)


def gen_group_sizes(p, n):
    quotient = p // n
    remainder = p % n
    result = [quotient + 1] * remainder + [quotient] * (n - remainder)
    return result


def sort_cluster_centers(cluster_centers, labels):
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


def balance_groups_3(X, cluster_labels, cluster_centers, n_clusters):
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


def balance_groups(X, cluster_labels, cluster_centers, n_clusters):
    """
    Balance the sizes of clusters by redistributing points to achieve more uniform group sizes.

    Parameters:
    X (numpy.ndarray): A 2D array where each row represents a data point and each column represents a feature.
    cluster_labels (numpy.ndarray): An array of integers where each value represents the cluster assignment of the corresponding data point in X.
    cluster_centers (numpy.ndarray): A 2D array where each row represents the centroid of a cluster.
    n_clusters (int): The total number of clusters.

    Returns:
    numpy.ndarray: An updated array of cluster labels after balancing the clusters.

    The function iteratively adjusts cluster memberships to ensure that no cluster has significantly more members than the average number of members per cluster.

    Steps:
    1. Calculate the initial sizes of the clusters.
    2. While any cluster size exceeds the average size (n_clusters / p), do the following:
       a. Identify the largest cluster.
       b. Find the data point in this cluster that is farthest from its centroid.
       c. Determine the nearest centroid of another cluster that isn't already full.
       d. Move the farthest data point to this new cluster.
       e. Update the sizes of the affected clusters.
    3. Return the updated cluster labels.

    Example:
    >>> X = np.array([[1, 2], [2, 1], [3, 4], [5, 6]])
    >>> cluster_labels = np.array([0, 0, 1, 1])
    >>> cluster_centers = np.array([[1.5, 1.5], [4, 5]])
    >>> n_clusters = 2
    >>> balance_groups(X, cluster_labels, cluster_centers, n_clusters)
    array([0, 0, 1, 1])  # Example output, the actual result may vary depending on input data
    """
    p = cluster_centers.shape[0]
    group_sizes = np.bincount(cluster_labels, minlength=p)

    while np.max(group_sizes) > n_clusters/p:
        # Find the group with the largest size
        largest_group_index = np.argmax(group_sizes)

        # Find the person in the largest group who is farthest from the centroid
        largest_group_indices = np.where(
            cluster_labels == largest_group_index)[0]
        distances_to_centroid = np.linalg.norm(
            X[largest_group_indices] - cluster_centers[largest_group_index], axis=1)
        farthest_person_index = largest_group_indices[np.argmax(
            distances_to_centroid)]

        # Move the farthest person to the group with the nearest centroid
        distances_to_centroids = np.linalg.norm(
            X[farthest_person_index] - cluster_centers, axis=1)

        # Exclude distance to the original centroid
        distances_to_centroids[largest_group_index] = np.inf

        # Find the index of the minimum distance
        new_group_index = np.argmin(distances_to_centroids)

        while group_sizes[new_group_index] >= n_clusters/p:
            # Set the distance to the current centroid to infinity to exclude it
            distances_to_centroids[new_group_index] = np.inf
            # Find the index of the next best option
            new_group_index = np.argmin(distances_to_centroids)

        cluster_labels[farthest_person_index] = new_group_index

        # Update group sizes
        group_sizes[largest_group_index] -= 1
        group_sizes[new_group_index] += 1

    return cluster_labels


def find_nearest_group(person_data, clusters):
    min_distance = float('inf')
    nearest_group_index = -1
    for i, centroid in enumerate(clusters):
        distance = sqeucdist(person_data, centroid)
        if distance < min_distance:
            min_distance = distance
            nearest_group_index = i
    return nearest_group_index


def balance_cluster_sizes(X, cluster_labels, cluster_centers):
    target_size = X.shape[0] / len(cluster_centers)
    cluster_counts = np.bincount(cluster_labels)

    for i in range(len(X)):
        current_cluster = cluster_labels[i]
        if cluster_counts[current_cluster] > target_size:
            distances = np.linalg.norm(cluster_centers - X[i], axis=1)
            nearest_smaller_cluster = np.argmin(
                distances[cluster_counts < target_size])
            cluster_labels[i] = nearest_smaller_cluster
            cluster_counts[current_cluster] -= 1
            cluster_counts[nearest_smaller_cluster] += 1

    return cluster_labels


def visualize_clusters_3d(X, cluster_labels):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i, label in enumerate(set(cluster_labels)):
        group = X[cluster_labels == label]
        ax.scatter(group[:, 0], group[:, 1], group[:, 2],
                   c=colors[i], marker='o', alpha=0.5)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('PCA Visualization')
    plt.show(block=True)


def pca_transform(X):
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    return X_pca

# testingVV


data = [["Alice", 1, 3, 2, 5, 2, 4, 1, 5, 3, 2, 1, 5, 3, 4, 1, 3, 2, 5, 4, 3, 5, 1, 2, 4, 5, 2, 1, 3, 4, 1, 3, 2, 5, 4, 2, 3, 1, 5, 4, 5, 2, 3, 1, 4, 5, 3, 2, 1, 4, 2],
        ["Bob", 5, 4, 1, 3, 3, 2, 5, 1, 4, 3, 2, 5, 1, 4, 2, 3, 1, 5, 4, 2, 3, 5, 1, 4,
            3, 1, 2, 4, 5, 3, 1, 2, 4, 5, 2, 1, 4, 3, 5, 2, 4, 1, 3, 5, 2, 3, 4, 1, 5, 3],
        ["Charlie", 2, 4, 3, 1, 1, 3, 5, 2, 4, 1, 5, 3, 2, 4, 1, 3, 2, 5, 4, 1, 5, 4, 3,
            2, 1, 5, 3, 4, 2, 1, 4, 2, 5, 3, 1, 4, 2, 3, 5, 1, 2, 3, 5, 4, 1, 5, 3, 4, 2, 1],
        ["David", 3, 2, 5, 1, 4, 4, 3, 2, 1, 5, 3, 4, 1, 2, 5, 3, 2, 4, 5, 1, 2, 4, 5, 1,
            3, 4, 1, 5, 2, 3, 4, 2, 1, 3, 5, 2, 4, 1, 3, 5, 2, 1, 4, 5, 3, 1, 4, 2, 3, 5],
        ["Emma", 1, 4, 2, 5, 3, 5, 1, 3, 2, 4, 1, 3, 5, 4, 2, 3, 4, 1, 5, 2, 3, 4, 5, 1,
            2, 4, 5, 3, 2, 1, 4, 5, 3, 2, 1, 4, 3, 5, 1, 2, 3, 4, 5, 2, 1, 3, 5, 2, 4, 1],
        ["Eva", 3, 5, 1, 2, 4, 1, 4, 3, 2, 5, 4, 1, 2, 5, 3, 1, 4, 2, 5, 3, 2, 4, 5, 1,
            3, 2, 4, 1, 5, 3, 2, 4, 1, 5, 3, 4, 2, 5, 1, 3, 1, 2, 4, 5, 3, 2, 1, 4, 3, 5],
        ["Fiona", 1, 3, 2, 5, 2, 4, 1, 5, 3, 2, 1, 5, 3, 4, 1, 3, 2, 5, 4, 3, 5, 1, 2, 4,
            5, 2, 1, 3, 4, 1, 3, 2, 5, 4, 2, 3, 1, 5, 4, 5, 2, 3, 1, 4, 5, 3, 2, 1, 4, 2],
        ["George", 5, 4, 1, 3, 3, 2, 5, 1, 4, 3, 2, 5, 1, 4, 2, 3, 1, 5, 4, 2, 3, 5, 1, 4,
            3, 1, 2, 4, 5, 3, 1, 2, 4, 5, 2, 1, 4, 3, 5, 2, 4, 1, 3, 5, 2, 3, 4, 1, 5, 3],
        ["Hannah", 2, 4, 3, 1, 1, 3, 5, 2, 4, 1, 5, 3, 2, 4, 1, 3, 2, 5, 4, 1, 5, 4, 3, 2,
            1, 5, 3, 4, 2, 1, 4, 2, 5, 3, 1, 4, 2, 3, 5, 1, 2, 3, 5, 4, 1, 5, 3, 4, 2, 1],
        ["Ian", 3, 2, 5, 1, 4, 4, 3, 2, 1, 5, 3, 4, 1, 2, 5, 3, 2, 4, 5, 1, 2, 4, 5, 1,
            3, 4, 1, 5, 2, 3, 4, 2, 1, 3, 5, 2, 4, 1, 3, 5, 2, 1, 4, 5, 3, 1, 4, 2, 3, 5],
        ["Jane", 1, 4, 2, 5, 3, 5, 1, 3, 2, 4, 1, 3, 5, 4, 2, 3, 4, 1, 5, 2, 3, 4, 5, 1,
            2, 4, 5, 3, 2, 1, 4, 5, 3, 2, 1, 4, 3, 5, 1, 2, 3, 4, 5, 2, 1, 3, 5, 2, 4, 1],
        ["Jack", 3, 5, 1, 2, 4, 1, 4, 3, 2, 5, 4, 1, 2, 5, 3, 1, 4, 2, 5, 3, 2, 4, 5, 1,
            3, 2, 4, 1, 5, 3, 2, 4, 1, 5, 3, 4, 2, 5, 1, 3, 1, 2, 4, 5, 3, 2, 1, 4, 3, 5],
        ["Kate", 1, 3, 2, 5, 2, 4, 1, 5, 3, 2, 1, 5, 3, 4, 1, 3, 2, 5, 4, 3, 5, 1, 2, 4,
            5, 2, 1, 3, 4, 1, 3, 2, 5, 4, 2, 3, 1, 5, 4, 5, 2, 3, 1, 4, 5, 3, 2, 1, 4, 2],
        ["Liam", 2, 5, 3, 1, 4, 4, 1, 3, 2, 5, 1, 4, 3, 5, 2, 1, 3, 4, 2, 5, 1, 2, 4, 5,
            3, 1, 2, 3, 5, 4, 1, 5, 2, 4, 3, 1, 5, 2, 4, 3, 1, 2, 5, 4, 3, 1, 2, 5, 3, 4],
        ["Olivia", 4, 3, 1, 2, 5, 1, 5, 4, 3, 2, 1, 4, 2, 5, 3, 1, 4, 5, 3, 2, 1, 4, 3,
            2, 5, 1, 4, 2, 3, 5, 1, 2, 4, 5, 1, 2, 3, 5, 4, 1, 3, 2, 4, 5, 3, 1, 2, 4, 5, 3],
        ["Noah", 3, 5, 1, 4, 2, 4, 3, 5, 1, 2, 3, 1, 5, 4, 2, 3, 1, 2, 5, 4, 3, 2, 1,
            4, 5, 3, 4, 2, 5, 1, 3, 2, 4, 1, 5, 3, 4, 5, 1, 2, 4, 3, 5, 2, 1, 3, 4, 5, 2, 1],
        ["Ava", 5, 2, 1, 3, 4, 2, 4, 5, 1, 3, 2, 1, 4, 5, 3, 2, 4, 1, 5, 3, 4, 1, 2,
            5, 3, 1, 2, 4, 5, 3, 2, 1, 4, 5, 3, 4, 2, 5, 1, 3, 2, 4, 5, 1, 2, 3, 5, 4, 2, 1],
        ["Isabella", 1, 3, 5, 4, 2, 3, 4, 1, 5, 2, 3, 4, 2, 5, 1, 3, 4, 5, 1, 2, 3, 4,
            5, 1, 2, 3, 4, 1, 2, 5, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2],
        ["William", 4, 1, 5, 2, 3, 2, 1, 4, 5, 3, 1, 4, 5, 3, 2, 1, 4, 5, 3, 2, 1, 4,
            5, 3, 2, 1, 4, 5, 3, 2, 1, 4, 5, 3, 2, 1, 4, 5, 3, 2, 1, 4, 5, 3, 2, 1, 4, 5, 3, 2],
        ["Sophia", 2, 5, 3, 4, 1, 1, 2, 5, 4, 3, 1, 2, 5, 4, 3, 1, 2, 5, 4, 3, 1, 2, 5,
            4, 3, 1, 2, 5, 4, 3, 1, 2, 5, 4, 3, 1, 2, 5, 4, 3, 1, 2, 5, 4, 3, 1, 2, 5, 4, 3],
        ["James", 3, 1, 4, 5, 2, 2, 3, 1, 5, 4, 2, 3, 1, 5, 4, 2, 3, 1, 5, 4, 2, 3, 1,
            5, 4, 2, 3, 1, 5, 4, 2, 3, 1, 5, 4, 2, 3, 1, 5, 4, 2, 3, 1, 5, 4, 2, 3, 1, 5, 4],
        ["Benjamin", 5, 2, 4, 1, 3, 3, 5, 2, 4, 1, 3, 5, 2, 4, 1, 3, 5, 2, 4, 1, 3, 5,
            2, 4, 1, 3, 5, 2, 4, 1, 3, 5, 2, 4, 1, 3, 5, 2, 4, 1, 3, 5, 2, 4, 1, 3, 5, 2, 4, 1],
        ["Charlotte", 1, 4, 3, 5, 2, 2, 1, 4, 3, 5, 2, 1, 4, 3, 5, 2, 1, 4, 3, 5, 2,
            1, 4, 3, 5, 2, 1, 4, 3, 5, 2, 1, 4, 3, 5, 2, 1, 4, 3, 5, 2, 1, 4, 3, 5, 2, 1, 4, 3, 5],
        ]
# for i in range(len(data)):
#     print(data[i][0], len(data[i]))
data = np.array(data)
ids = data[:, 0]
X = data[:, 1:51].astype(float)

return_values = kmeans_custom(X, 5)[0]
print(return_values)

count_dict = {}

# Count the occurrences of each value
for num in return_values:
    if num in count_dict:
        count_dict[num] += 1
    else:
        count_dict[num] = 1

# Print the counts
for value, frequency in count_dict.items():
    print(f"Number {value} appears {frequency} times.")

# visualize_clusters_3d(pca_transform(X), kmeans_custom(X, 5)[0])
