import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('data.csv')

print(data.head())

print(data.columns)

print(f'Liczba wierszy (próbek): {data.shape[0]}')
print(f'Liczba kolumn (cech): {data.shape[1]}')

if 'id' in data.columns:
    data.drop('id', axis=1, inplace=True)
    print("Usunięto kolumnę 'id'")

if 'Unnamed: 32' in data.columns:
    data.drop('Unnamed: 32', axis=1, inplace=True)
    print("Usunięto kolumnę 'Unnamed: 32'")

nan_columns = data.columns[data.isna().any()].tolist()
if nan_columns:
    print(f'Kolumny zawierające NaN: {nan_columns}')
    data.drop(columns=nan_columns, inplace=True)
else:
    print('Brak kolumn zawierających NaN.')

y = data['diagnosis']

X_2d = data[['radius_mean', 'texture_mean']]

print(f'Unikalne etykiety: {y.unique()}')

print(X_2d.head())

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_temp, y_train, y_temp = train_test_split(X_2d, y_encoded, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f'Rozmiar zbioru treningowego: {X_train.shape[0]}')
print(f'Rozmiar zbioru walidacyjnego: {X_val.shape[0]}')
print(f'Rozmiar zbioru testowego: {X_test.shape[0]}')

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

perceptron = Perceptron(max_iter=5000, eta0=0.001, random_state=42)
perceptron.fit(X_train_scaled, y_train)

print('Model perceptronu został wytrenowany.')

plt.figure(figsize=(10, 6))

X_combined_scaled = np.vstack((X_train_scaled, X_val_scaled, X_test_scaled))
y_combined = np.hstack((y_train, y_val, y_test))

plt.scatter(X_combined_scaled[y_combined == 0][:, 0], X_combined_scaled[y_combined == 0][:, 1],
            color='blue', marker='o', label='Benign (0)')
plt.scatter(X_combined_scaled[y_combined == 1][:, 0], X_combined_scaled[y_combined == 1][:, 1],
            color='red', marker='x', label='Malignant (1)')

a = perceptron.coef_[0][0]
b = perceptron.coef_[0][1]
c = perceptron.intercept_[0]

x_values = np.linspace(X_combined_scaled[:, 0].min(), X_combined_scaled[:, 0].max(), 100)
y_values = -(a * x_values + c) / b

plt.plot(x_values, y_values, color='black', linestyle='--', label='Granica decyzyjna')

plt.xlabel('radius_mean (standaryzowane)')
plt.ylabel('texture_mean (standaryzowane)')
plt.title('Granica decyzyjna perceptronu dla danych o raku piersi')
plt.legend()
plt.grid(True)
plt.show()

print("Granica decyzyjna dobrze oddziela klasy, jeśli większość punktów znajduje się po odpowiednich stronach linii.")