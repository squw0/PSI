import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np

# Zadanie 1
df = pd.read_csv('heart.csv')
print(df.head())
print(df.columns)
print(f'Liczba wierszy: {df.shape[0]}')
print(f'Liczba kolumn: {df.shape[1]}')

if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)
    print("Usuwanie kolumny id")

if 'Unnamed: 32' in df.columns:
    df.drop('Unnamed: 32', axis=1, inplace=True)
    print("Usuwanie kolumny Unnamed: 32")

nan_columns = df.columns[df.isna().any()].tolist()
if nan_columns:
    print(f'Kolumny zawierające tylko NaN: {nan_columns}')
    df.drop(columns=nan_columns, inplace=True)
else:
    print('Nie ma kolumn NaN.')

# Zadanie 2
y = df['target']
X_2d = df[['chol', 'trestbps']]
print(f'Unikalne etykiety: {y.unique()}')
print(X_2d.head())

# Zadanie 3
X_train, X_temp, y_train, y_temp = train_test_split(X_2d, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(f'Wielkosc zbioru treningowego: {X_train.shape[0]}')
print(f'Wielkosc zbioru walidacyjnego: {X_val.shape[0]}')
print(f'Wielkosc zbioru testowego: {X_test.shape[0]}')

# Zadanie 4
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
perceptron = Perceptron(max_iter=5000, eta0=0.001, random_state=42)
perceptron.fit(X_train_scaled, y_train)
print("Trening perceptronu wykonany teraz może jechać na zawody")

# Zadanie 5 
X_combined_scaled = np.vstack((X_train_scaled, X_val_scaled, X_test_scaled))
y_combined = np.hstack((y_train, y_val, y_test))
plt.figure(figsize=(10, 6))
plt.scatter(X_combined_scaled[y_combined == 0][:, 0], X_combined_scaled[y_combined == 0][:, 1], color='blue', label='Brak choroby (0)')
plt.scatter(X_combined_scaled[y_combined == 1][:, 0], X_combined_scaled[y_combined == 1][:, 1], color='red', label='Choroba serca (1)')
w1, w2 = perceptron.coef_[0]
b = perceptron.intercept_[0]
x_values = np.linspace(X_combined_scaled[:, 0].min(), X_combined_scaled[:, 0].max(), 100)
y_values = -(w1 * x_values + b) / w2
plt.plot(x_values, y_values, color='black', linestyle='--', label='Granica decyzyjna')
plt.xlabel('Cholesterol standard')
plt.ylabel('Ciśnienie krwi w spoczynku standard')
plt.title('Granica decyzyjna perceptronu - choroby serca')
plt.legend()
plt.grid(True)
plt.show()
