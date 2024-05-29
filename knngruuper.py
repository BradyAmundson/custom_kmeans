import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn.datasets as skdata
from sklearn.model_selection import train_test_split

iris_data = skdata.load_iris()
X = iris_data.data
y = iris_data.target

mu = np.mean(X, axis=0)
X_center = X - mu

C = np.matmul(X_center.T, X_center)/X.shape[0]

S, V = np.linalg.eig(C)  # eigenvalues and eigenvectors
print(S)
print(V)

order = np.argsort(S)[::-1]
W = V[:, order][:, :3]

X_pca = np.matmul(X_center, W)

data_split = (X_pca[np.where(y == 0)],
              X_pca[np.where(y == 1)], X_pca[np.where(y == 2)])

col = ['r', 'g', 'b']
labels = ['Setosa', 'Versicolor', 'Virginica']
markers = ['o', '^', 's']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for z, c, l, m in zip(data_split, col, labels, markers):
    ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=c, label=l, marker=m)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.legend()
plt.show(block=True)

X_hat = np.matmul(X_pca, W.T) + mu
mse = np.mean((X - X_hat)**2)
print(mse)

W = V[:, order][:, :2]
Z_2 = np.matmul(X_center, W)
X_hat_2 = np.matmul(Z_2, W.T) + mu
data_split_2 = (Z_2[np.where(y == 0)],
                Z_2[np.where(y == 1)], Z_2[np.where(y == 2)])
fig = plt.figure()
ax = fig.add_subplot(111)
for z, c, l, m in zip(data_split_2, col, labels, markers):
    ax.scatter(z[:, 0], z[:, 1], c=c, label=l, marker=m)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.legend()
plt.show(block=True)

mse_2 = np.mean((X - X_hat_2)**2)
print(mse_2)


digits_data = skdata.load_digits()
X = digits_data.data
y = digits_data.target

X = np.reshape(X, (-1, 8, 8))
fig = plt.figure()
for i in range(10):
    ax = fig.add_subplot(2, 5, i+1)
    ax.imshow(X[i, :, :], cmap='gray')
    ax.axis('off')
plt.show(block=True)

X = digits_data.data
X = np.reshape(X, (-1, 64))
mu = np.mean(X, axis=0)
X_center = X - mu
C = np.matmul(X_center.T, X_center)/X.shape[0]
S, V = np.linalg.eig(C)
order = np.argsort(S)[::-1]
W = V[:, order][:, :3]
Z = np.matmul(X_center, W)

X_hat = np.matmul(Z, W.T) + mu
mse = np.mean((X - X_hat)**2)
print(mse)

# plt.plot(S[order])
# ax.set_xlabel('PC')
# ax.set_ylabel('Eigenvalue')
# plt.show(block=True)

W = V[:, order][:, :45]

Z = np.matmul(X_center, W)
X_hat = np.matmul(Z, W.T) + mu
mse = np.mean((X - X_hat)**2)
print(mse)

X_hat = np.reshape(X_hat, (-1, 8, 8))
fig = plt.figure()
for i in range(10):
    ax = fig.add_subplot(2, 5, i+1)
    ax.imshow(X_hat[i, :, :], cmap='gray')
    ax.axis('off')
plt.show(block=True)
