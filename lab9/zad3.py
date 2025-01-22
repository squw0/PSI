import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score

iris = load_iris()
data = iris.data
labels_true = iris.target

scaler = StandardScaler()
data = scaler.fit_transform(data)

num_clusters = 3  # Liczba klas
num_epochs = 100  # Liczba epok
learning_rate = 0.1  # Współczynnik uczenia

# Kohonen Clustering
weights = np.random.rand(num_clusters, data.shape[1])

for epoch in range(num_epochs):
    for i in range(data.shape[0]):
        input_vector = data[i, :]
        distances = np.linalg.norm(weights - input_vector, axis=1)
        winner_index = np.argmin(distances)
        weights[winner_index, :] += learning_rate * (input_vector - weights[winner_index, :])

cluster_assignments_kohonen = np.zeros(data.shape[0], dtype=int)
for i in range(data.shape[0]):
    input_vector = data[i, :]
    distances = np.linalg.norm(weights - input_vector, axis=1)
    cluster_assignments_kohonen[i] = np.argmin(distances)

# KMeans Clustering
def kmeans(data, num_clusters, num_iterations=100):
    centroids = data[np.random.choice(data.shape[0], num_clusters, replace=False)]
    for _ in range(num_iterations):
        distances = cdist(data, centroids, 'euclidean')
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([data[labels == j].mean(axis=0) for j in range(num_clusters)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels

cluster_assignments_kmeans = kmeans(data, num_clusters)

ari_kohonen = adjusted_rand_score(labels_true, cluster_assignments_kohonen)
ari_kmeans = adjusted_rand_score(labels_true, cluster_assignments_kmeans)

print(f"Adjusted Rand Index (Kohonen): {ari_kohonen:.2f}")
print(f"Adjusted Rand Index (KMeans): {ari_kmeans:.2f}")

plt.figure()
plt.scatter(data[:, 0], data[:, 1], c=cluster_assignments_kohonen, cmap='viridis')
plt.title('Kohonen Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()

plt.figure()
plt.scatter(data[:, 0], data[:, 1], c=cluster_assignments_kmeans, cmap='viridis')
plt.title('KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()
