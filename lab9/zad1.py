import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
data = iris.data

data_original = data.copy()
scaler = StandardScaler()
data = scaler.fit_transform(data)

num_clusters = 3 # Liczba klas
num_epochs_list = [50, 100, 200] # Liczba epok
learning_rate = 0.1 # Współczynnik uczenia

num_features = data.shape[1]

for num_epochs in num_epochs_list:
    weights = np.random.rand(num_clusters, num_features)

    for epoch in range(num_epochs):
        for i in range(data.shape[0]):
            input_vector = data[i, :]
            distances = np.linalg.norm(weights - input_vector, axis=1)
            winner_index = np.argmin(distances)
            weights[winner_index, :] += learning_rate * (input_vector - weights[winner_index, :])

    cluster_assignments = np.zeros(data.shape[0], dtype=int)
    for i in range(data.shape[0]):
        input_vector = data[i, :]
        distances = np.linalg.norm(weights - input_vector, axis=1)
        cluster_assignments[i] = np.argmin(distances)

    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=cluster_assignments, cmap='viridis')
    plt.title(f'Kohonen Clustering (num_epochs={num_epochs})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.show()
