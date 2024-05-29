import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from clustering import kmeans_custom

# load data from csv ./big5data/data-final.csv
# data = np.genfromtxt(
#     '/Users/bradyamundson/Documents/gruuperML/big5data/IPIP-FFM-data-8Nov2018 2/data-final.csv', delimiter='	', skip_header=1)
# X = data[0:27, 0:50]

# split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # create a KNN classifier
# knn = KNeighborsClassifier(n_neighbors=5)

# # train the classifier
# knn.fit(X, np.zeros(len(X)))

# plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 0], X[:, 1], c=knn.predict(X),
#             cmap='viridis', s=50, alpha=0.5)
# plt.title('K Nearest Neighbors Clustering')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.colorbar(label='Cluster')
# # plt.show(block=True)

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


kmeans = KMeans(n_clusters=5)  # Adjust the number of clusters as needed
kmeans.fit(X)
print(kmeans.labels_)

kmeans_custom = kmeans_custom(X, 5)
print(kmeans_custom[0])

study_groups = {i: [] for i in range(len(kmeans.cluster_centers_))}

for i, label in enumerate(kmeans.labels_):
    study_groups[label].append(ids[i])
print(study_groups)

study_groups_custom = {i: [] for i in range(len(kmeans_custom[1]))}

for i, label in enumerate(kmeans_custom[0]):
    study_groups_custom[label].append(ids[i])
print(study_groups_custom)

raise


# cluster_labels = kmeans.labels_

# # Now cluster_labels contains the cluster assignment for each data point (individual)

# # Optionally, you can also get the cluster centers
# cluster_centers = kmeans.cluster_centers_

# # Now you can use cluster_labels to group individuals into study groups based on clusters
# # For example, create an empty dictionary to store study groups
# study_groups = {i: [] for i in range(len(cluster_centers))}

# # Iterate through individuals and assign them to study groups based on cluster labels
# for i, label in enumerate(cluster_labels):
#     study_groups[label].append(X[i])

# print(study_groups)

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')

# colors = ['r', 'g', 'b']

# for i, group in study_groups.items():
#     pca = PCA(n_components=3)

#     # Fit PCA on the data and transform it to obtain the principal components
#     X_pca = pca.fit_transform(group)

#     # Extract x, y, and z coordinates from X_pca
#     x = X_pca[:, 0]
#     y = X_pca[:, 1]
#     z = X_pca[:, 2]

#     # Plot the data points
#     ax.scatter(x, y, z, c=colors[i], marker='o', alpha=0.5)

# # Set labels and title
# ax.set_xlabel('Principal Component 1')
# ax.set_ylabel('Principal Component 2')
# ax.set_zlabel('Principal Component 3')
# ax.set_title('PCA Visualization')

# # Show plot
# plt.show()

kmeans_pca = KMeans(n_clusters=8)  # Adjust the number of clusters as needed
pca = PCA(n_components=3)


X_pca = pca.fit_transform(X)
kmeans_pca.fit(X_pca)

cluster_labels = kmeans_pca.labels_

# Now cluster_labels contains the cluster assignment for each data point (individual)

# Optionally, you can also get the cluster centers
cluster_centers = kmeans_pca.cluster_centers_

# Now you can use cluster_labels to group individuals into study groups based on clusters
# For example, create an empty dictionary to store study groups
study_groups = {i: [] for i in range(len(cluster_centers))}

# Iterate through individuals and assign them to study groups based on cluster labels
for i, label in enumerate(cluster_labels):
    study_groups[label].append(X_pca[i])

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

for i, group in study_groups.items():
    group = np.array(group)
    x = group[:, 0]
    y = group[:, 1]
    z = group[:, 2]

    # Plot the data points
    ax.scatter(x, y, z, c=colors[i], marker='o', alpha=0.5)

# Set labels and title
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('PCA Visualization')

# Show plot
plt.show(block=True)
