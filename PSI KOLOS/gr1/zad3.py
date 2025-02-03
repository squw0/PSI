import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Zadanie 1: Wczytanie pliku data.csv
# Wczytanie danych
data = pd.read_csv('data.csv')

# Wyświetlenie pierwszych 5 wierszy
print(data.head())

# Wyświetlenie nazw kolumn
print(data.columns)

# Sprawdzenie liczby wierszy i kolumn
print(f'Liczba wierszy (próbek): {data.shape[0]}')
print(f'Liczba kolumn (cech): {data.shape[1]}')

# Sprawdzenie, czy są kolumny zbędne lub zawierające NaN
if 'id' in data.columns:
    data.drop('id', axis=1, inplace=True)
    print("Usunięto kolumnę 'id'")

if 'Unnamed: 32' in data.columns:
    data.drop('Unnamed: 32', axis=1, inplace=True)
    print("Usunięto kolumnę 'Unnamed: 32'")

# Sprawdzenie kolumn z brakującymi danymi
nan_columns = data.columns[data.isna().any()].tolist()
if nan_columns:
    print(f'Kolumny zawierające NaN: {nan_columns}')
    data.drop(columns=nan_columns, inplace=True)
else:
    print('Brak kolumn zawierających NaN.')

# Zadanie 2: Identyfikacja kolumny z etykietą 'diagnosis'
# Zapisanie etykiet jako 'M'/'B' przed enkodowaniem
y = data['diagnosis']

# Wybranie dwóch kolumn jako cechy (X_2d)
X_2d = data[['radius_mean', 'texture_mean']]

# Wyświetlenie unikalnych etykiet
print(f'Unikalne etykiety: {y.unique()}')

# Wyświetlenie pierwszych 5 wierszy wybranych cech
print(X_2d.head())

# Zadanie 3: Użycie LabelEncoder do zamiany etykiet na 0 i 1
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Podział danych na zbiory treningowy (70%), walidacyjny (15%) i testowy (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X_2d, y_encoded, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Wyświetlenie rozmiarów zbiorów
print(f'Rozmiar zbioru treningowego: {X_train.shape[0]}')
print(f'Rozmiar zbioru walidacyjnego: {X_val.shape[0]}')
print(f'Rozmiar zbioru testowego: {X_test.shape[0]}')
