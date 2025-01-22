import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Creating two linearly separable classes
X_separable, y_separable = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, 
                                               n_clusters_per_class=1, n_classes=2, random_state=42)

# Data normalization
scaler = StandardScaler()
X_separable = scaler.fit_transform(X_separable)

# Splitting the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_separable, y_separable, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Perceptron training on the iris dataset
perceptron = Perceptron(max_iter=1000, eta0=0.1, random_state=0)
perceptron.fit(X_train, y_train)

val_accuracy = perceptron.score(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy:.2f}")

learning_rates = [0.01, 0.05, 0.1, 0.2, 0.5]
best_lr = 0
best_val_accuracy = 0

for lr in learning_rates:
    temp_model = Perceptron(max_iter=1000, eta0=lr, random_state=0)
    temp_model.fit(X_train, y_train)
    temp_val_accuracy = temp_model.score(X_val, y_val)
    print(f"Learning Rate: {lr}, Validation Accuracy: {temp_val_accuracy:.2f}")

    if temp_val_accuracy > best_val_accuracy:
        best_val_accuracy = temp_val_accuracy
        best_lr = lr

print(f"Best Learning Rate: {best_lr} with Validation Accuracy: {best_val_accuracy:.2f}")

final_model = Perceptron(max_iter=1000, eta0=best_lr, random_state=0)
final_model.fit(X_train, y_train)

test_accuracy = final_model.score(X_test, y_test)
print(f"Test Accuracy with Best Learning Rate ({best_lr}): {test_accuracy:.2f}")
