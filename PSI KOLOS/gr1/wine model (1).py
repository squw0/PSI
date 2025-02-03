import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

data = pd.read_csv('wine-dataset.txt', header=None, sep='\t')

y_wine = data.iloc[:, 0].values
X_wine = data.iloc[:, 1:].values

label_encoder = LabelEncoder()
y_wine_encoded = label_encoder.fit_transform(y_wine)

print("Shape of X_wine (features):", X_wine.shape)
print("Shape of y_wine_encoded (labels):", y_wine_encoded.shape)

X_train, X_temp, y_train, y_temp = train_test_split(X_wine, y_wine_encoded, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

perceptron = Perceptron(max_iter=5000, eta0=0.001, random_state=42)
perceptron.fit(X_train, y_train)

y_val_pred = perceptron.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy (Wine Dataset) with adjusted parameters: {val_accuracy * 100:.2f}%')

y_test_pred = perceptron.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy (Wine Dataset) with adjusted parameters: {test_accuracy * 100:.2f}%')

mask = y_wine_encoded < 2  
X_wine_binary = X_wine[mask]
y_wine_binary = y_wine_encoded[mask]

unique_classes = np.unique(y_wine_binary)
if len(unique_classes) > 1:
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X_wine_binary, y_wine_binary, test_size=0.3, random_state=42)

    if len(np.unique(y_train_bin)) > 1:

        X_train_bin = scaler.fit_transform(X_train_bin)
        X_test_bin = scaler.transform(X_test_bin)

        perceptron_bin = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
        perceptron_bin.fit(X_train_bin, y_train_bin)

        y_test_pred_bin = perceptron_bin.predict(X_test_bin)
        test_accuracy_bin = accuracy_score(y_test_bin, y_test_pred_bin)
        print(f'Test Accuracy (Binary Wine Dataset): {test_accuracy_bin * 100:.2f}%')
    else:
        print(f"Zbyt mało klas w zbiorze uczącym: znaleziono tylko {len(np.unique(y_train_bin))} klasę/klasy.")
else:
    print(f"Zbyt mało klas po filtrowaniu: znaleziono tylko {len(unique_classes)} klasę/klasy. Wybierz inne klasy.")

X_separable, y_separable = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, 
                                               n_clusters_per_class=1, n_classes=2, random_state=42)


X_train_sep, X_temp_sep, y_train_sep, y_temp_sep = train_test_split(X_separable, y_separable, test_size=0.3, random_state=42)
X_val_sep, X_test_sep, y_val_sep, y_test_sep = train_test_split(X_temp_sep, y_temp_sep, test_size=0.5, random_state=42)

X_train_sep = scaler.fit_transform(X_train_sep)
X_val_sep = scaler.transform(X_val_sep)
X_test_sep = scaler.transform(X_test_sep)

perceptron_sep = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
perceptron_sep.fit(X_train_sep, y_train_sep)

y_val_pred_sep = perceptron_sep.predict(X_val_sep)
val_accuracy_sep = accuracy_score(y_val_sep, y_val_pred_sep)
print(f'Validation Accuracy (Linearly Separable Data): {val_accuracy_sep * 100:.2f}%')

y_test_pred_sep = perceptron_sep.predict(X_test_sep)
test_accuracy_sep = accuracy_score(y_test_sep, y_test_pred_sep)
print(f'Test Accuracy (Linearly Separable Data): {test_accuracy_sep * 100:.2f}%')

# Wizualizacja sztucznie wygenerowanych danych i granicy decyzyjnej perceptronu
plt.figure(figsize=(8, 6))
plt.scatter(X_separable[y_separable == 0][:, 0], X_separable[y_separable == 0][:, 1], color='red', label='Class 0')
plt.scatter(X_separable[y_separable == 1][:, 0], X_separable[y_separable == 1][:, 1], color='blue', label='Class 1')

x_values = np.linspace(X_separable[:, 0].min(), X_separable[:, 0].max(), 100)
y_values = -(perceptron_sep.coef_[0][0] * x_values + perceptron_sep.intercept_[0]) / perceptron_sep.coef_[0][1]
plt.plot(x_values, y_values, color='black', linestyle='--', label='Decision Boundary')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Linearly Separable Data and Perceptron Decision Boundary')
plt.legend()
plt.grid(True)
plt.show()
