import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Creating two linearly separable classes
X_separable, y_separable = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, 
                                               n_clusters_per_class=1, n_classes=2, random_state=42)

# Data normalization
scaler = StandardScaler()
X_separable = scaler.fit_transform(X_separable)

# Splitting the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_separable, y_separable, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Plotting the training data
plt.figure(figsize=(8, 6))
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='red', label='Class 0')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='blue', label='Class 1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Training Data')
plt.legend()
plt.grid(True)
plt.show()
